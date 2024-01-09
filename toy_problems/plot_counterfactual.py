import os
import pytorch_lightning as pl
import torch
from data import MAKE_DATA
from decoder_cnn import IMG_DECODE_SHAPE
from utils.enums import Task
from utils.plot import *
from vae import VAE


N_EXAMPLES = 10
N_COLS = 7


def reconstruct_x(model, z_parent, z_child):
    batch_size = len(z_parent)
    z = torch.hstack((z_parent, z_child))
    x_pred = model.decoder.mlp(z).reshape(batch_size, *IMG_DECODE_SHAPE)
    x_pred = model.decoder.decoder_cnn(x_pred)
    return torch.sigmoid(x_pred)


def main(args):
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    data_train, _, _ = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size, args.eval_batch_size)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    model.eval()
    x, y, e, parent, child = data_train.dataset[:]
    x, y, e = x.to(model.device), y.to(model.device), e.to(model.device)
    posterior_parent, posterior_child = model.encoder(x, y, e)
    z_parent, z_child = posterior_parent.loc, posterior_child.loc
    zero_mean = z_parent[parent == 0].mean(dim=0).unsqueeze(0)
    one_mean = z_parent[parent == 1].mean(dim=0).unsqueeze(0)
    fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'plot_counterfactual')
    os.makedirs(fig_dpath, exist_ok=True)
    for example_idx in range(N_EXAMPLES):
        z_parent_elem, z_child_elem, parent_elem = z_parent[[example_idx]], z_child[[example_idx]], parent[example_idx]
        z_parent_mean = one_mean if parent_elem == 0 else zero_mean
        fig, axes = plt.subplots(1, N_COLS, figsize=(2 * N_COLS, 2))
        for ax in axes:
            remove_ticks(ax)
        for col_idx in range(N_COLS):
            ratio = col_idx / N_COLS
            z_parent_perturb = ratio * z_parent_mean + (1 - ratio) * z_parent_elem
            x_pred_perturb = reconstruct_x(model, z_parent_perturb, z_child_elem)
            plot_red_green_image(axes[col_idx], x_pred_perturb.squeeze().detach().cpu().numpy())
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.pdf'))
        plt.close()