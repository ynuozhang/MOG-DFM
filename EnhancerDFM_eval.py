import yaml
from tqdm import tqdm
import torch
torch.manual_seed(0)

from modules.dna_module import DNAModule
from utils.dataset import EnhancerDataset

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

from models.enhancer_models import CNNModel
from models.classifier import CNNClassifier
from utils.flow_utils import get_wasserstein_dist, upgrade_state_dict, map_t_to_alpha

import random
random.seed(42)

# evaluation arguments
checkpoint_path = './ckpt/enhancer/cnn_epoch1500_lr0.001_embed256_hidden256_11.6335.ckpt'
lr = 1e-4
epochs = 200
vocab_size = 4
embed_dim = 256
hidden_dim = 256
epsilon = 1e-3
batch_size = 256
warmup_epochs = epochs // 10
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

# test_ds = EnhancerDataset(mel_enhancer=True, split='test')
# test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=8)

def general_step(x_1, source_distribution):
    if source_distribution == "uniform":
        x_0 = torch.randint_like(x_1, high=vocab_size)
    elif source_distribution == "mask":
        x_0 = torch.zeros_like(x_1) + mask_token
    else:
        raise NotImplementedError

    # sample time
    t = torch.rand(x_1.shape[0]).to(device) * (1 - epsilon)
    t = t.to(x_1.device)
    
    # sample probability path
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    # discrete flow matching generalized KL loss
    logits = probability_denoiser(x=path_sample.x_t, t=path_sample.t)
    loss = loss_fn(logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t)

    return loss, logits, t

def mean(l):
    return sum(l) / len(l)

# probability denoiser model init
# probability_denoiser = MLPModel(input_dim=vocab_size, time_dim=1, hidden_dim=hidden_dim, length=500).to(device)
probability_denoiser = CNNModel(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
probability_denoiser.load_state_dict(torch.load(checkpoint_path, map_location=device))

# instantiate a convex path object
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)

loss_fn = MixturePathGeneralizedKL(path=path)

with open('./classifier_ckpt/enhancer_class_hparams.yaml') as f:
    hparams = yaml.load(f, Loader=yaml.UnsafeLoader)

clean_cls_model = CNNClassifier(hparams['args'], alphabet_size=vocab_size, num_cls=47, classifier=True)
clean_cls_model.load_state_dict(upgrade_state_dict(torch.load('./classifier_ckpt/enhancer_class.ckpt', map_location=device)['state_dict'],prefixes=['model.']))
clean_cls_model.eval()
clean_cls_model.to(device)
for params in clean_cls_model.parameters():
    params.requires_grad = False

val_losses = []
val_accs = []
x1s = []
preds = []
ts = []

embeds = []
embeds_gen = []

def dna_to_tensor(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    indices = [mapping[base] for base in seq]
    return torch.tensor(indices, dtype=torch.long)

# 10k sequences sampled by EnhancerDFM
with open('./enhancer_sample.txt', 'r') as f:
    sequences = f.readlines()

for sequence in tqdm(sequences, total=len(sequences)):
    x_1 = dna_to_tensor(sequence.strip())
    x_1 = x_1.unsqueeze(0).to(device)

    probability_denoiser.eval()
    with torch.no_grad():
        loss, logits, t = general_step(x_1, source_distribution)
        val_losses.append(loss)

        prediction = logits.argmax(-1)
        accuracy = (prediction == x_1).float().mean()
        val_accs.append(accuracy)

        alphas = map_t_to_alpha(t, alpha_scale=2)
        embeds.append(clean_cls_model(x_1, alphas, return_embedding=True)[1])
        embeds_gen.append(clean_cls_model(prediction, alphas, return_embedding=True)[1])

embeds = torch.cat(embeds).detach().cpu().numpy()
embeds_gen = torch.cat(embeds_gen).detach().cpu().numpy()
fbd_gen = get_wasserstein_dist(embeds_gen, embeds)

embeds_rand = torch.randint(0,4, size=embeds_gen.shape).numpy()
fbd_rand = get_wasserstein_dist(embeds_rand, embeds)
print(f"FBD Random: {fbd_rand:.4f}")

print(f"Val Loss: {mean(val_losses):.4f}, Val ACC: {mean(val_accs):.4f}, FBD_Gen: {fbd_gen:.4f}")
