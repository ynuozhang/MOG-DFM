import yaml
from utils.dataset import EnhancerDataset
import torch
torch.manual_seed(0)
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

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

# training arguments
lr = 1e-4
epochs = 1500
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

train_ds = EnhancerDataset(mel_enhancer=True, split='train')
val_ds = EnhancerDataset(mel_enhancer=True, split='valid')

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=8)

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

with open('../dirichlet-flow-matching/workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/lightning_logs/version_0/hparams.yaml') as f:
    hparams = yaml.load(f, Loader=yaml.UnsafeLoader)

clean_cls_model = CNNClassifier(hparams['args'], alphabet_size=vocab_size, num_cls=47, classifier=True)
clean_cls_model.load_state_dict(upgrade_state_dict(torch.load('../dirichlet-flow-matching/workdir/clsMELclean_cnn_dropout02_2023-12-31_12-26-28/epoch=9-step=5540.ckpt', map_location=device)['state_dict'],prefixes=['model.']))
clean_cls_model.eval()
clean_cls_model.to(device)
for params in clean_cls_model.parameters():
    params.requires_grad = False

best_fbd_gen = float('inf')
best_model_path = f'./ckpt/enhancer/cnn_epoch{epochs}_lr{lr}_embed{embed_dim}_hidden{hidden_dim}.ckpt'
for epoch in range(epochs):
    train_losses = []
    for i, (x_1, label) in enumerate(train_loader):
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
    for i, (x_1, label) in enumerate(val_loader):
        x_1 = x_1.to(device)

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

    if epoch == 0:
        embeds_rand = torch.randint(0,4, size=embeds_gen.shape).numpy()
        fbd_rand = get_wasserstein_dist(embeds_rand, embeds)
        print(f"FBD Random: {fbd_rand:.4f}")

    if fbd_gen < best_fbd_gen:
        best_fbd_gen = fbd_gen
        torch.save(probability_denoiser.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch} with FBD_Gen: {fbd_gen:.4f}")

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        cosine_scheduler.step()

    print(f"Epoch {epoch}: Train Loss: {mean(train_losses):.4f}, Val Loss: {mean(val_losses):.4f}, Val ACC: {mean(val_accs):.4f}, FBD_Gen: {fbd_gen:.4f}")
