import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from data import N_CLASSES, N_ENVS, MAKE_DATA, PLOT
from decoder_cnn import IMG_DECODE_SHAPE
from vae import VAE
from utils.enums import Task


N_COLS = 10


def sample_prior(rng, model):
    y = torch.tensor(rng.choice(N_CLASSES), dtype=torch.long, device=model.device)[None]
    e = torch.tensor(rng.choice(N_ENVS), dtype=torch.long, device=model.device)[None]
    prior_causal, prior_spurious, prior_exogenous = model.prior(y, e)
    zc_sample, zs_sample, zx_sample = prior_causal.sample(), prior_spurious.sample(), prior_exogenous.sample()
    return zc_sample, zs_sample, zx_sample


def reconstruct_x(model, z):
    batch_size = len(z)
    x_pred = model.decoder.mlp(z).reshape(batch_size, *IMG_DECODE_SHAPE)
    x_pred = model.decoder.decoder_cnn(x_pred)
    return torch.sigmoid(x_pred)


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    model.eval()
    fig, axes = plt.subplots(1, N_COLS, figsize=(N_COLS, 2))
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plot = PLOT[args.dataset]
    for col_idx in range(N_COLS):
        zc_sample, zs_sample, zx_sample = sample_prior(rng, model)
        z_sample = torch.hstack((zc_sample, zs_sample, zx_sample))
        x_sample = reconstruct_x(model, z_sample)
        plot(axes[col_idx], x_sample.squeeze().detach().cpu().numpy())
    fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig')
    os.makedirs(fig_dpath, exist_ok=True)
    plt.savefig(os.path.join(fig_dpath, 'plot_samples.png'))
    plt.close()