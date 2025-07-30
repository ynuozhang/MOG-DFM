import torch
torch.manual_seed(0)
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from datasets import load_from_disk

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

from models.peptide_models import CNNModel
from utils import dataloader

import random
random.seed(42)

# training arguments
lr = 1e-4
epochs = 200
vocab_size = 24 # 20 natural amino acids + '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3
embed_dim = 512
hidden_dim = 256
epsilon = 1e-3
batch_size = 512
warmup_epochs = epochs // 10
device = 'cuda:0'
source_distribution = "uniform"

train_dataset = load_from_disk('./dataset/tokenized_peptide/train')
val_dataset = load_from_disk('./dataset/tokenized_peptide/val')

data_module = dataloader.CustomDataModule(train_dataset, val_dataset, test_dataset=None)
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

def general_step(x_1, source_distribution):
    if source_distribution == "uniform":
        x_0 = torch.randint_like(x_1, high=vocab_size)
    elif source_distribution == "mask":
        x_0 = torch.zeros_like(x_1) + 3 # '<unk>': 3
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
probability_denoiser = CNNModel(alphabet_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)

# instantiate a convex path object
scheduler = PolynomialConvexScheduler(n=2.0)
path = MixtureDiscreteProbPath(scheduler=scheduler)

optim = torch.optim.Adam(probability_denoiser.parameters(), lr=lr) 
loss_fn = MixturePathGeneralizedKL(path=path)

def warmup_fn(epoch):
    if epoch < warmup_epochs:
        return 0.1 + 0.9 * (epoch / warmup_epochs)  # Increases from 0.1 to 1
    return 1.0  # Keep the learning rate constant after warmup

warmup_scheduler = LambdaLR(optim, lr_lambda=warmup_fn)
cosine_scheduler = CosineAnnealingLR(optim, T_max=epochs - warmup_epochs, eta_min=0.1*lr)

best_val_acc = 0
best_val_loss = 999.0
best_model_path = f'./ckpt/peptide/cnn_epoch{epochs}_lr{lr}_embed{embed_dim}_hidden{hidden_dim}.ckpt'
for epoch in range(epochs):
    train_losses = []
    for i, batch in enumerate(train_loader):
        x_1 = batch['input_ids']
        x_1 = x_1.to(device)

        optim.zero_grad() 
        
        loss, _, _ = general_step(x_1, source_distribution)
        train_losses.append(loss)

        loss.backward()
        optim.step()

    val_losses = []
    val_accs = []
    x1s = []
    preds = []
    ts = []
    
    embeds = []
    embeds_gen = []
    for i, batch in enumerate(val_loader):
        x_1 = batch['input_ids']
        x_1 = x_1.to(device)

        probability_denoiser.eval()
        with torch.no_grad():
            loss, logits, t = general_step(x_1, source_distribution)
            val_losses.append(loss)

            prediction = logits.argmax(-1)
            accuracy = (prediction == x_1).float().mean()
            val_accs.append(accuracy)
            
    val_loss = mean(val_losses)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(probability_denoiser.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch} with Val Loss: {val_loss:.4f}")

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()


    print(f"Epoch {epoch}: Train Loss: {mean(train_losses):.4f}, Val Loss: {mean(val_losses):.4f}, Val ACC: {mean(val_accs):.4f}")

