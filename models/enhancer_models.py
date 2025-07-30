from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import copy
import pdb

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[...]

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x): 
        return torch.sigmoid(x) * x

class CNNModel(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, alphabet_size=4, embed_dim=256, hidden_dim=256):
        """
        Args:
            embed_dim (int): Dimensionality of the token and time embeddings.
        """
        super().__init__()
        self.alphabet_size = alphabet_size
        
        self.token_embedding = nn.Embedding(self.alphabet_size, embed_dim)
        
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )

        self.swish = Swish()
        
        n = hidden_dim
        
        self.linear = nn.Conv1d(embed_dim, n, kernel_size=9, padding=4)
        
        self.blocks = nn.ModuleList([
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, padding=4),
            nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, padding=4),
            # nn.Conv1d(n, n, kernel_size=9, dilation=4, padding=16),
            # nn.Conv1d(n, n, kernel_size=9, dilation=16, padding=64),
            # nn.Conv1d(n, n, kernel_size=9, dilation=64, padding=256)
        ])
        
        self.denses = nn.ModuleList([Dense(embed_dim, n) for _ in range(5)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, n) for _ in range(5)])

        self.final = nn.Sequential(
            nn.Conv1d(n, n, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(n, self.alphabet_size, kernel_size=1)
        )


    def forward(self, x, t):
        """
        Args:
            x: Tensor of shape (B, L) containing DNA token indices.
            t: Tensor of shape (B,) containing the time steps.
        Returns:
            out: Tensor of shape (B, L, 4) with output logits for each DNA base.
        """
        x = self.token_embedding(x) # (B, L) -> (B, L, embed_dim)
        
        time_embed = self.swish(self.time_embed(t))  # (B, embed_dim)
        
        out = x.permute(0, 2, 1)    # (B, L, embed_dim) -> (B, embed_dim, L)
        out = self.swish(self.linear(out))  # (B, n, L)
        
        # Process through convolutional blocks, adding time conditioning via dense layers.
        for block, dense, norm in zip(self.blocks, self.denses, self.norms):
            # dense(embed) gives (B, n); unsqueeze to (B, n, 1) for broadcasting.
            h = self.swish(block(norm(out + dense(time_embed)[:, :, None])))
            # Residual connection if shapes match.
            if h.shape == out.shape:
                out = h + out
            else:
                out = h
        
        out = self.final(out)  # (B, 4, L)
        out = out.permute(0, 2, 1)  # (B, L, 4)
        
        # Normalization
        out = out - out.mean(dim=-1, keepdim=True)
        return out


class MLPModel(nn.Module):
    def __init__(
        self, input_dim: int = 128, time_dim: int = 1, hidden_dim=128, length=500):
        super().__init__()
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.time_embedding = nn.Linear(1, time_dim)
        self.token_embedding = torch.nn.Embedding(self.input_dim, hidden_dim)

        self.swish = Swish()

        self.main = nn.Sequential(
            self.swish,
            nn.Linear(hidden_dim * length + time_dim, hidden_dim),
            self.swish,
            nn.Linear(hidden_dim, hidden_dim),
            self.swish,
            nn.Linear(hidden_dim, hidden_dim),
            self.swish,
            nn.Linear(hidden_dim, self.input_dim * length),
        )

    def forward(self, x, t):
        '''
        x shape (B,L)
        t shape (B,)
        '''
        t = self.time_embedding(t.unsqueeze(-1))
        x = self.token_embedding(x)

        B, N, d = x.shape
        x = x.reshape(B, N * d)
        
        h = torch.cat([x, t], dim=1)
        h = self.main(h)

        h = h.reshape(B, N, self.input_dim)

        return h

class DirichletCNNModel(nn.Module):
    def __init__(self, args, alphabet_size):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.args = args
        expanded_simplex_input = args.cls_expanded_simplex and (args.mode == 'dirichlet' or args.mode == 'riemannian')
        inp_size = self.alphabet_size * (2 if expanded_simplex_input else 1)
        self.linear = nn.Conv1d(inp_size, args.hidden_dim, kernel_size=9, padding=4)
        self.time_embedder = nn.Sequential(GaussianFourierProjection(embed_dim= args.hidden_dim),nn.Linear(args.hidden_dim, args.hidden_dim))

        self.num_layers = 5 * args.num_cnn_stacks
        self.convs = [nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, padding=4),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=4, padding=16),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=16, padding=64),
                                     nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=9, dilation=64, padding=256)]
        self.convs = nn.ModuleList([copy.deepcopy(layer) for layer in self.convs for i in range(args.num_cnn_stacks)])
        self.time_layers = nn.ModuleList([Dense(args.hidden_dim, args.hidden_dim) for _ in range(self.num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(args.hidden_dim) for _ in range(self.num_layers)])
        self.final_conv = nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=1),
                                   nn.ReLU(),
                                   nn.Conv1d(args.hidden_dim, self.alphabet_size, kernel_size=1))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, seq, t):
        time_emb = F.relu(self.time_embedder(t))
        feat = seq.permute(0, 2, 1)
        feat = F.relu(self.linear(feat))

        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            if not self.args.clean_data:
                h = h + self.time_layers[i](time_emb)[:, :, None]
            h = self.norms[i]((h).permute(0, 2, 1))
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
        feat = self.final_conv(feat)
        feat = feat.permute(0, 2, 1)
        return feat