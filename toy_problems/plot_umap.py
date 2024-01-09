import os
import pytorch_lightning as pl
import umap
from data import MAKE_DATA
from sklearn.preprocessing import StandardScaler
from utils.enums import Task
from utils.plot import *
from vae import VAE


N_POINTS = 1000


def to_umap(x):
    reducer = umap.UMAP()
    x = StandardScaler().fit_transform(x)
    x = reducer.fit_transform(x)
    return x


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    data_train, _, _ = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size, args.eval_batch_size)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    model.eval()
    x, y, e, parent, child = data_train.dataset[:]
    x, y, e = x.to(model.device), y.to(model.device), e.to(model.device)
    posterior_parent, posterior_child = model.encoder(x, y, e)
    z_parent, z_child = posterior_parent.loc.detach().cpu().numpy(), posterior_child.loc.detach().cpu().numpy()

    subset_idxs = rng.choice(len(data_train.dataset), N_POINTS, replace=False)
    fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig')
    os.makedirs(fig_dpath, exist_ok=True)

    z_parent = to_umap(z_parent)
    cmap_parent = mcolors.ListedColormap(['red', 'blue'])
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    remove_ticks(ax)
    ax.scatter(
        z_parent[subset_idxs, 0],
        z_parent[subset_idxs, 1],
        c=parent[subset_idxs],
        cmap=cmap_parent
    )
    red_patch = mpatches.Patch(color='red', label=r'$y=0$')
    blue_patch = mpatches.Patch(color='blue', label=r'$y=1$')
    handles = [red_patch, blue_patch]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(fig_dpath, 'plot_parent_umap.pdf'))
    plt.close()

    z_child = to_umap(z_child)
    cmap_child = mcolors.LinearSegmentedColormap.from_list('', ['green', 'yellow', 'orange', 'red'])
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    remove_ticks(ax)
    plt.scatter(
        z_child[subset_idxs, 0],
        z_child[subset_idxs, 1],
        c=child[subset_idxs],
        cmap=cmap_child
    )
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dpath, 'plot_child_umap.pdf'))
    plt.close()