import torch
from flow_matching.utils import categorical
import math
import inspect

def generate_simplex_lattice_points(num_obj: int, num_div: int) -> torch.Tensor:
    def rec(n, H):
        if n == 1:
            return [[H]]
        points = []
        for i in range(H + 1):
            for tail in rec(n - 1, H - i):
                points.append([i] + tail)
        return points

    points = rec(num_obj, num_div)
    weight_vectors = torch.tensor(points, dtype=torch.float32) / num_div
    return weight_vectors

def select_random_weight_vector(num_obj: int, num_div: int):
    weight_vectors = generate_simplex_lattice_points(num_obj, num_div)
    idx = torch.randint(0, weight_vectors.size(0), (1,)).item()
    random_weight_vector = weight_vectors[idx]
    return random_weight_vector, weight_vectors

def z_score_norm(tensor, eps=1e-8):
    mean = tensor.mean(dim=-1, keepdim=True)
    std = tensor.std(dim=-1, unbiased=False, keepdim=True).clamp(min=eps)
    return (tensor - mean) / std

def guided_transition_scoring(x_t, u_t, w, s_models, t, importance, args):
    B, L, vocab_size = u_t.shape
    device = x_t.device
    guided_u_t = u_t.clone()  
    
    # 1. Randomly select one position per sequence.
    pos_indices = torch.randint(low=1, high=L-1, size=(B,), device=device)  # shape: (B,)
    batch_idx = torch.arange(B, device=device)
    current_tokens = x_t[batch_idx, pos_indices]  # shape: (B,)

    # 2. Build candidate tokens for each sequence and remove self-transition.
    full_cand_tokens = torch.arange(vocab_size, device=device).unsqueeze(0).expand(B, vocab_size) # (B, vocab_size)
    mask = (full_cand_tokens != current_tokens.unsqueeze(1))  # (B, vocab_size)
    # Now, cand_tokens contains only candidate tokens that differ from the current token.
    cand_tokens = torch.masked_select(full_cand_tokens, mask).view(B, vocab_size - 1)  # (B, vocab_size-1)

    # 3. Create candidate sequences by replacing the token at the selected position.
    new_x = x_t.unsqueeze(1).expand(B, vocab_size, L).clone()
    new_x = new_x[mask].view(B, vocab_size - 1, L)  # (B, vocab_size-1, L)
    new_x[batch_idx, :, pos_indices] = cand_tokens 

    new_x_flat = new_x.view(B * (vocab_size - 1), L)
    improvements_list = []
    with torch.no_grad():
        count = 0
        for i, s in enumerate(s_models):
            sig = inspect.signature(s.forward) if hasattr(s, 'forward') else inspect.signature(s)
            if 't' in sig.parameters:
                candidate_scores = s(new_x_flat, t)
                base_score = s(x_t, t)
            else:
                candidate_scores = s(new_x_flat)
                base_score = s(x_t)

            if isinstance(candidate_scores, tuple):
                for k, score in enumerate(candidate_scores):
                    improvement = candidate_scores[k].view(B, vocab_size - 1) - base_score[k].unsqueeze(1)
                    improvement = improvement.float()
                    improvement *= importance[count]
                    improvements_list.append(improvement.unsqueeze(2))
                    count += 1
            else:
                improvement = candidate_scores.view(B, vocab_size - 1) - base_score.unsqueeze(1)
                improvement = improvement.float()
                improvement *= importance[count]
                improvements_list.append(improvement.unsqueeze(2))  # (B, vocab_size-1, 1)
                count += 1

    improvement_values = torch.cat(improvements_list, dim=2) # (B, vocab_size-1, N)
    if args.is_peptide:
        improvement_values[:, :4, :] = -10 # Mask non-residue positions

    # 5. Compute ranking scores I_n
    ranks = torch.argsort(torch.argsort(improvement_values, dim=1), dim=1).float() + 1  # (B, vocab_size-1, N)
    I_n = ranks / float(vocab_size - 1)
    avg_I = I_n.mean(dim=2)
    norm_avg_I = z_score_norm(avg_I)    # (B, vocab_size-1)
    
    # 6. Compute directional score D
    D = (improvement_values * w.view(1, 1, -1)).sum(dim=2)  
    norm_D = z_score_norm(D)    # (B, vocab_size-1)

    # 7. Combine the scores
    delta_S = norm_avg_I + args.lambda_ * norm_D  # (B, vocab_size-1)

    # 9. Update the guided velocities at the selected positions.
    factor = torch.exp(args.beta * delta_S)  # (B, vocab_size-1)
    factor = torch.clamp(factor, min=-100, max=100)

    guided_u_t[batch_idx.unsqueeze(1), pos_indices.unsqueeze(1), cand_tokens] = u_t[batch_idx.unsqueeze(1), pos_indices.unsqueeze(1), cand_tokens] * factor

    # 10. For the self-transition (current token) at the selected position, 
    # set its guided velocity to be the negative sum of the updated off-diagonals.
    updated_vals = guided_u_t[batch_idx, pos_indices, :]  # (B, vocab_size)
    sum_off_diag = updated_vals.sum(dim=1) - updated_vals[batch_idx, current_tokens]
    guided_u_t[batch_idx, pos_indices, current_tokens] = -sum_off_diag

    return guided_u_t, pos_indices, cand_tokens, improvement_values, delta_S

def adaptive_hypercone_filtering(improvement_values, cand_tokens, delta_S, w, Phi, args, ema_r_t=None):
    B, num_candidates, N = improvement_values.shape
    device = improvement_values.device
    eps = 1e-8

    # Compute norms and angles.
    imp_norm = torch.norm(improvement_values.float(), dim=2)  # (B, num_candidates)
    dot_product = (improvement_values * w.view(1, 1, -1)).sum(dim=2)
    w_norm = torch.norm(w) + eps
    cos_angle = dot_product / (imp_norm * w_norm + eps)
    cos_angle = cos_angle.clamp(-1.0, 1.0)
    angles = torch.acos(cos_angle)  # (B, num_candidates)

    valid_mask = angles < math.pi / 2 
    accepted_mask = valid_mask & (angles <= Phi) # (B, num_candidates)

    # Determine the best candidate for each sequence.
    # We'll use a loop over batch items (batch size is typically moderate).
    best_candidate = torch.empty(B, dtype=torch.long, device=device)
    for i in range(B):
        # For sequence i, consider only valid candidates.
        if valid_mask[i].any():
            # There is at least one candidate with α^i < π.
            if accepted_mask[i].any():
                # At least one candidate passes the hypercone: choose the one with max delta_S among accepted.
                candidate_idx = torch.argmax(delta_S[i].masked_fill(~accepted_mask[i], float('-inf')))
            else:
                # No candidate was accepted, but some are valid. Select best candidate among valid ones.
                candidate_idx = torch.argmax(delta_S[i].masked_fill(~valid_mask[i], float('-inf')))
            best_candidate[i] = cand_tokens[i, candidate_idx]
        else:
            # No candidate is valid (all α^i >= π) → self-transition.
            best_candidate[i] = -1

    # Compute rejection rate only over valid candidates.
    rejection_rates = []
    for i in range(B):
        valid_candidates = valid_mask[i]
        total_valid = valid_candidates.sum().item()
        if total_valid > 0:
            # Among valid candidates, count how many are rejected.
            num_rejected = (valid_candidates.sum() - accepted_mask[i].sum()).item()
            rejection_rates.append(num_rejected / total_valid)
    if len(rejection_rates) > 0:
        r_t = sum(rejection_rates) / len(rejection_rates)
    else:
        # If no sequence has any valid candidate, set r_t to 0.
        r_t = 0.0

    if ema_r_t is None:
        ema_r_t = args.tau

    # Update hypercone angle and ema rejection rate only if there is at least one valid candidate in the batch.
    if valid_mask.any():
        new_ema_r_t = args.alpha_r * ema_r_t + (1 - args.alpha_r) * r_t
        new_Phi = Phi * torch.exp(torch.tensor(args.eta * (new_ema_r_t - args.tau), device=device))
        new_Phi = new_Phi.clamp(args.Phi_min, args.Phi_max).item()
    else:
        new_ema_r_t = ema_r_t
        new_Phi = Phi  # No update if no valid candidate exists.

    return best_candidate, accepted_mask, valid_mask, new_Phi, new_ema_r_t

def get_best_candidate(improvement_values, cand_tokens, delta_S):
    B, num_candidates, N = improvement_values.shape
    device = improvement_values.device
    best_candidate = torch.empty(B, dtype=torch.long, device=device)
    
    for i in range(B):
        candidate_idx = torch.argmax(delta_S[i])
        best_candidate[i] = cand_tokens[i, candidate_idx]
        
    return best_candidate

def euler_sample(x_t, pos_indices, best_candidate, guided_u_t, h):
    B, L, V = guided_u_t.shape
    device = x_t.device
    u = torch.zeros_like(guided_u_t)

    valid_mask = best_candidate != -1
    if valid_mask.any():
        valid_idx = torch.nonzero(valid_mask).squeeze(-1)
        # For these sequences, update the velocity at the selected position and candidate token.
        u[valid_idx, pos_indices[valid_idx], best_candidate[valid_idx]] = \
            guided_u_t[valid_idx, pos_indices[valid_idx], best_candidate[valid_idx]]
    
    # Compute intensity at the selected positions.
    # For sequences with no valid candidate (i.e. self-transition), intensity remains zero.
    intensity = torch.zeros(B, device=device)
    if valid_mask.any():
        intensity[valid_idx] = u[valid_idx, pos_indices[valid_idx]].sum(dim=-1)

    # According to the Euler Sampling formula, `p_jump` should be `1 - torch.exp(-h * intensity)`
    # However, since `h = 1 / T` is small, p_jump becomes tiny and slows down sampling.
    # To compensate, we scale `intensity` by T. We can do this because this is equivalent to setting `args.beta` to `T * args.beta`.
    # So for faster sampling, we just use  `1 - torch.exp(-1 * intensity)`
    p_jump = 1 - torch.exp(-1 * intensity)
    
    rand_val = torch.rand(B, device=device)

    jump_decision = (rand_val < p_jump) & valid_mask
    
    # For sequences where a jump is decided, update the token at pos_indices to best_candidate.
    x_t[jump_decision, pos_indices[jump_decision]] = best_candidate[jump_decision]

    return x_t
