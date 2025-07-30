import yaml
from tqdm import tqdm
import torch
from torch import nn

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

from models.enhancer_models import CNNModel
from models.classifier import CNNClassifier
from utils.flow_utils import upgrade_state_dict

from utils.parsing import parse_guidance_args
args = parse_guidance_args()

import subprocess
import inspect
import concurrent.futures

# MOO hyper-parameters
step_size = 1 / args.T
n_samples = args.n_samples
n_batches = args.n_batches
length = args.length

shape = args.target_DNA_shape
tgt_cls = args.target_enhancer_class
importance = [1, 10] # [1,10] for HelT, [1,100] for Rise
max_value = 36 # 36 for HelT, 3.4 for Rise

print(shape)
print(tgt_cls)
print(importance)
print(max_value)

vocab_size = 4
device = 'cuda:0'

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

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)

wrapped_probability_denoiser = WrappedModel(probability_denoiser)
solver = MixtureDiscreteEulerSolver(model=wrapped_probability_denoiser, path=path, vocabulary_size=vocab_size)

class SumModel(nn.Module):
    def __init__(self):
        super(SumModel, self).__init__()

    def forward(self, x):
        return x.sum(dim=1)

class CountOnesModel(nn.Module):
    def __init__(self):
        super(CountOnesModel, self).__init__()

    def forward(self, x):
        return (x == 1.0).sum(dim=1)

class EnhancerClassifier(nn.Module):
    def __init__(self, tgt_cls):
        super(EnhancerClassifier, self).__init__()
        with open('./classifier_ckpt/enhancer_class_hparams.yaml') as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)

        self.clean_cls_model = CNNClassifier(hparams['args'], alphabet_size=vocab_size, num_cls=47, classifier=True)
        self.clean_cls_model.load_state_dict(upgrade_state_dict(torch.load('./classifier_ckpt/enhancer_class.ckpt', map_location=device)['state_dict'],prefixes=['model.']))
        self.clean_cls_model.eval()
        self.clean_cls_model.to(device)
        for params in self.clean_cls_model.parameters():
            params.requires_grad = False

        self.tgt_cls = tgt_cls

    def forward(self, x, t):
        cls_prob = torch.softmax(self.clean_cls_model(x, t, return_embedding=False), dim=-1)
        cls_log_prob = torch.log(cls_prob)
        
        if t == 1:
            return torch.argmax(cls_prob, dim=-1)
        return cls_log_prob[:, self.tgt_cls]


class StructureModel(nn.Module):
    def __init__(self, device):
        super(StructureModel, self).__init__()
        alphabet = "ACGT"
        self.ascii_codes = torch.tensor([ord(c) for c in alphabet], dtype=torch.uint8).to(device)

    def decode(self, x):
        dna_chars  = self.ascii_codes[x] 
        batch_seqs = [''.join(map(chr, row.tolist())) for row in dna_chars]
        return batch_seqs

    def run_deepDNAshape(self, seq):
        global shape
        # print(shape)
        cmd = [
            "conda", "run", "-n", 'deepDNAshape',  # activate deepDNAshape conda environment for predicting the DNA shape
            "deepDNAshape",
            "--seq", seq,
            "--feature", shape,
            "--layer", '7',
        ]

        try:
            output = subprocess.check_output(cmd, text=True)     
        except FileNotFoundError:
            raise RuntimeError("`conda` command not found. Ensure Conda ≥4.6 is on your PATH.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"deepDNAshape failed:\n{e.stderr or e.stdout}") from e
        
        values = output.strip().split(' ')
        values = [float(value) for value in values]

        global max_value
        return min(sum(values) / len(values), max_value)

    def forward(self, x):
        seqs = self.decode(x)
        values_list = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self.run_deepDNAshape, seqs)

        values_list = list(results)

        return torch.tensor(values_list).to(device)


enhancer_model = EnhancerClassifier(tgt_cls=tgt_cls)
structure_model = StructureModel(device)

score_models = []
score_models.extend([enhancer_model, structure_model])

for _ in range(n_batches):
    if source_distribution == "uniform":
        x_init = torch.randint(size=(n_samples, length), high=vocab_size, device=device)
    elif source_distribution == "mask":
        x_init = (torch.zeros(size=(n_samples, length), device=device) + mask_token).long()
    else:
        raise NotImplementedError

    x_1 = solver.multi_guidance_sample(args=args, x_init=x_init, 
                        step_size=step_size, 
                        verbose=True, 
                        time_grid=torch.tensor([0.0, 1.0-1e-3]),
                        score_models=score_models,
                        importance=importance)

    print(structure_model.decode(x_1))

    for i, s in enumerate(score_models):
        sig = inspect.signature(s.forward) if hasattr(s, 'forward') else inspect.signature(s)
        if 't' in sig.parameters:
            candidate_scores = s(x_1, 1)
        else:
            candidate_scores = s(x_1)

        print(f"Score {i}: {candidate_scores}")
