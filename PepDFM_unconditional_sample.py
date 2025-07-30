from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

from models.peptide_models import CNNModel

import random
import os

random.seed(42)

# evaluation arguments
checkpoint_path = './ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt'
vocab_size = 24
device = 'cuda:0'
source_distribution = "uniform"

probability_denoiser = CNNModel(alphabet_size=vocab_size, embed_dim=512, hidden_dim=256).to(device)
probability_denoiser.load_state_dict(torch.load(checkpoint_path, map_location=device))
probability_denoiser.eval()

# instantiate a convex path object
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)

# Sample
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)

wrapped_probability_denoiser = WrappedModel(probability_denoiser)
solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=vocab_size)

step_size = 1 / 100
n_samples = 20
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

os.makedirs('./samples/', exist_ok=True)
for length in range(6, 10):
    for _ in tqdm(range(50)):
        if source_distribution == "uniform":
            x_init = torch.randint(low=4, high=vocab_size, size=(n_samples, length), device=device)
        elif source_distribution == "mask":
            x_init = (torch.zeros(size=(n_samples, length), device=device) + 3).long()
        else:
            raise NotImplementedError

        zeros = torch.zeros((n_samples, 1), dtype=x_init.dtype, device=x_init.device)
        twos = torch.full((n_samples, 1), 2, dtype=x_init.dtype, device=x_init.device)
        x_init = torch.cat([zeros, x_init, twos], dim=1)

        sol = solver.sample(x_init=x_init, 
                            step_size=step_size, 
                            verbose=False, 
                            time_grid=torch.tensor([0.0, 1.0-1e-3]))
        sol = sol.tolist()

        samples = [tokenizer.decode(seq).replace(' ', '')[5:-5] for seq in sol]
        with open(f'./samples/{length}.txt', 'a') as f:
            for sample in samples:
                f.writelines(sample+'\n')

