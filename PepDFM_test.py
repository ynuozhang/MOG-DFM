import torch
torch.manual_seed(0)
from datasets import load_from_disk
from transformers import AutoTokenizer

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

from models.peptide_models import CNNModel
from utils import dataloader

import random

random.seed(42)

# evaluation arguments
checkpoint_path = './ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt'
vocab_size = 24
warmup_epochs = epochs // 10
device = 'cuda:1'
source_distribution = "uniform"

test_dataset = load_from_disk('./dataset/tokenized_peptide/test')

data_module = dataloader.CustomDataModule(train_dataset=None, val_dataset=None, test_dataset=test_dataset)
test_loader = data_module.test_dataloader()

# probability denoiser model init
probability_denoiser = CNNModel(alphabet_size=vocab_size, embed_dim=512, hidden_dim=256).to(device)
probability_denoiser.load_state_dict(torch.load(checkpoint_path, map_location=device))
probability_denoiser.eval()

# instantiate a convex path object
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)

loss_fn = MixturePathGeneralizedKL(path=path)

def general_step(x_1, source_distribution):
    if source_distribution == "uniform":
        x_0 = torch.randint_like(x_1, high=vocab_size)
    elif source_distribution == "mask":
        x_0 = torch.zeros_like(x_1) + 3
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

test_losses = []
test_accs = []
x1s = []
preds = []
ts = []

embeds = []
embeds_gen = []

for i, batch in enumerate(test_loader):
    x_1 = batch['input_ids']
    x_1 = x_1.to(device)

    probability_denoiser.eval()
    with torch.no_grad():
        loss, logits, t = general_step(x_1, source_distribution)
        test_losses.append(loss)

        prediction = logits.argmax(-1)
        accuracy = (prediction == x_1).float().mean()
        test_accs.append(accuracy)

print(f"Test Loss: {mean(test_losses):.4f}, Test ACC: {mean(test_accs):.4f}")
