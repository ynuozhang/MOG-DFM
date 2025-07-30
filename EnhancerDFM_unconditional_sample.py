import yaml
import torch

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

from models.enhancer_models import CNNModel
from models.classifier import CNNClassifier
from utils.flow_utils import upgrade_state_dict

import random

random.seed(42)

# Sampling Hyper-parameters
step_size = 1 / 800
n_samples = 10
length = 100
vocab_size = 4
device = 'cuda:1'

source_distribution = "uniform"
if source_distribution == "uniform":
    added_token = 0
elif source_distribution == "mask":
    mask_token = vocab_size  # tokens starting from zero
    added_token = 1
else:
    raise NotImplementedError

# additional mask token
vocab_size += added_token

probability_denoiser = CNNModel(alphabet_size=vocab_size, embed_dim=256, hidden_dim=256).to(device)
probability_denoiser.load_state_dict(torch.load('./ckpt/enhancer/cnn_epoch1500_lr0.001_embed256_hidden256_11.6335.ckpt', map_location=device))

# instantiate a convex path object
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)

with open('./classifier_ckpt/enhancer_class_hparams.yaml') as f:
    hparams = yaml.load(f, Loader=yaml.UnsafeLoader)

clean_cls_model = CNNClassifier(hparams['args'], alphabet_size=vocab_size, num_cls=47, classifier=True)
clean_cls_model.load_state_dict(upgrade_state_dict(torch.load('./classifier_ckpt/enhancer_class.ckpt', map_location=device)['state_dict'],prefixes=['model.']))
clean_cls_model.eval()
clean_cls_model.to(device)
for params in clean_cls_model.parameters():
    params.requires_grad = False

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)

wrapped_probability_denoiser = WrappedModel(probability_denoiser)
solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=vocab_size)

if source_distribution == "uniform":
    x_init = torch.randint(size=(n_samples, length), high=vocab_size, device=device)
elif source_distribution == "mask":
    x_init = (torch.zeros(size=(n_samples, length), device=device) + mask_token).long()
else:
    raise NotImplementedError


sol = solver.sample(x_init=x_init, 
                    step_size=step_size, 
                    verbose=False, 
                    time_grid=torch.tensor([0.0, 1.0-1e-3]))

def tensor_to_dna(tensor):
    inverse_mapping = ['A', 'C', 'G', 'T']
    return ''.join([inverse_mapping[i] for i in tensor.tolist()])

def batch_tensor_to_dna(batch_tensor):
    return [tensor_to_dna(t) for t in batch_tensor]

sequences = batch_tensor_to_dna(sol)
with open('./enhancer_sample.txt', 'w') as f:
    for seq in sequences:
        f.write(seq + '\n')

