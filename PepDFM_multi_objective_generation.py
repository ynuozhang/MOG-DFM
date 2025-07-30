import yaml
from tqdm import tqdm
import torch
from torch import nn
from transformers import AutoTokenizer

from models.peptide_classifiers import *

from utils.parsing import parse_guidance_args
args = parse_guidance_args()

import random
import inspect

# MOO hyperparameters
step_size = 1 / args.T
n_samples = args.n_samples
n_batches = args.n_batches
length = args.length
target = args.target_protein
vocab_size = 24
source_distribution = "uniform"
device = 'cuda:0'

name = f"multi_objective_guided_generation"

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
target_sequence = tokenizer(target, return_tensors='pt')['input_ids'].to(device)

# Load Models
solver = load_solver('./ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt', vocab_size, device)

affinity_predictor = load_affinity_predictor('./classifier_ckpt/binding_affinity_unpooled.pt', device)
affinity_model = AffinityModel(affinity_predictor, target_sequence)
hemolysis_model = HemolysisModel(device=device)
nonfouling_model = NonfoulingModel(device=device)
solubility_model = SolubilityModel(device=device)
halflife_model = HalfLifeModel(device=device)

score_models = [hemolysis_model, nonfouling_model, solubility_model, halflife_model, affinity_model]
importance = [1, 1, 1, 0.5, 0.2] # Importance weights
print(f"Importance: {importance}")


for i in range(n_batches):
    if source_distribution == "uniform":
        x_init = torch.randint(low=4, high=vocab_size, size=(n_samples, length), device=device)
    elif source_distribution == "mask":
        x_init = (torch.zeros(size=(n_samples, length), device=device) + 3).long()
    else:
        raise NotImplementedError

    zeros = torch.zeros((n_samples, 1), dtype=x_init.dtype, device=x_init.device)
    twos = torch.full((n_samples, 1), 2, dtype=x_init.dtype, device=x_init.device)
    x_init = torch.cat([zeros, x_init, twos], dim=1)

    x_1 = solver.multi_guidance_sample(args=args, x_init=x_init, 
                        step_size=step_size, 
                        verbose=True, 
                        time_grid=torch.tensor([0.0, 1.0-1e-3]),
                        score_models=score_models,
                        importance=importance)

    samples = x_1.tolist()
    samples = [tokenizer.decode(seq).replace(' ', '')[5:-5] for seq in samples]
    print(samples)

    for i, s in enumerate(score_models):
        sig = inspect.signature(s.forward) if hasattr(s, 'forward') else inspect.signature(s)
        if 't' in sig.parameters:
            candidate_scores = s(x_1, 1)
        else:
            candidate_scores = s(x_1)
        
        print(f"Score {i}: {[round(s.item(), 4) for s in candidate_scores]}")
