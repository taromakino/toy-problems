import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from encoder_cnn import IMG_ENCODE_SIZE, EncoderCNN
from decoder_cnn import IMG_DECODE_SHAPE, IMG_DECODE_SIZE, DecoderCNN
from torch.optim import AdamW
from torchmetrics import Accuracy
from utils.nn_utils import SkipMLP, one_hot, repeat_batch, arr_to_cov


class Encoder(nn.Module):
    def __init__(self, parent_size, child_size, h_sizes):
        super().__init__()
        self.parent_size = parent_size
        self.child_size = child_size
        self.encoder_cnn = EncoderCNN()
        self.mu_parent = SkipMLP(IMG_ENCODE_SIZE, h_sizes, parent_size)
        self.offdiag_parent = SkipMLP(IMG_ENCODE_SIZE, h_sizes, parent_size ** 2)
        self.diag_parent = SkipMLP(IMG_ENCODE_SIZE, h_sizes, parent_size)
        self.mu_child = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, child_size)
        self.offdiag_child = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, child_size ** 2)
        self.diag_child = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, child_size)

    def parent_dist(self, x):
        batch_size = len(x)
        mu = self.mu_parent(x)
        offdiag = self.offdiag_parent(x)
        offdiag = offdiag.reshape(batch_size, self.parent_size, self.parent_size)
        diag = self.diag_parent(x)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def child_dist(self, x, y, e):
        batch_size = len(x)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu_child(x, y_one_hot, e_one_hot)
        offdiag = self.offdiag_child(x, y_one_hot, e_one_hot)
        offdiag = offdiag.reshape(batch_size, self.child_size, self.child_size)
        diag = self.diag_child(x, y_one_hot, e_one_hot)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def forward(self, x, y, e, is_parent_only):
        x = self.encoder_cnn(x)
        parent_dist = self.parent_dist(x)
        if is_parent_only:
            return parent_dist
        else:
            child_dist = self.child_dist(x, y, e)
            return parent_dist, child_dist


class Decoder(nn.Module):
    def __init__(self, parent_size, child_size, h_sizes):
        super().__init__()
        self.mlp = SkipMLP(parent_size + child_size, h_sizes, IMG_DECODE_SIZE)
        self.decoder_cnn = DecoderCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).view(batch_size, *IMG_DECODE_SHAPE)
        x_pred = self.decoder_cnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, parent_size, child_size, init_sd):
        super().__init__()
        self.mu_parent = nn.Parameter(torch.zeros(parent_size))
        self.offdiag_parent = nn.Parameter(torch.zeros(parent_size, parent_size))
        self.diag_parent = nn.Parameter(torch.zeros(parent_size))
        nn.init.normal_(self.mu_parent, 0, init_sd)
        nn.init.normal_(self.offdiag_parent, 0, init_sd)
        nn.init.normal_(self.diag_parent, 0, init_sd)
        self.mu_child = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, child_size))
        self.offdiag_child = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, child_size, child_size))
        self.diag_child = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, child_size))
        nn.init.normal_(self.mu_child, 0, init_sd)
        nn.init.normal_(self.offdiag_child, 0, init_sd)
        nn.init.normal_(self.diag_child, 0, init_sd)

    def parent_dist(self, batch_size):
        mu = repeat_batch(self.mu_parent, batch_size)
        offdiag = repeat_batch(self.offdiag_parent, batch_size)
        diag = repeat_batch(self.diag_parent, batch_size)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def child_dist(self, y, e):
        mu = self.mu_child[y, e]
        cov = arr_to_cov(self.offdiag_child[y, e], self.diag_child[y, e])
        return D.MultivariateNormal(mu, cov)

    def forward(self, y, e, is_parent_only):
        parent_dist = self.parent_dist(len(y))
        if is_parent_only:
            return parent_dist
        else:
            child_dist = self.child_dist(y, e)
            return parent_dist, child_dist


class VAE(pl.LightningModule):
    def __init__(self, task, parent_size, child_size, h_sizes, prior_reg_mult, init_sd, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.prior_reg_mult = prior_reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        # q(z_c,z_s|x,y,e)
        self.encoder = Encoder(parent_size, child_size, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(parent_size, child_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(parent_size, child_size, init_sd)
        # p(y|z)
        self.classifier = nn.Linear(parent_size, 1)
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, parent_size = mu.shape
        epsilon = torch.randn(batch_size, parent_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze(-1)

    def x_loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_parent, posterior_child = self.encoder(x, y, e, False)
        z_parent = self.sample_z(posterior_parent)
        z_child = self.sample_z(posterior_child)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        z = torch.hstack((z_parent, z_child))
        log_prob_x_z = self.decoder(x, z).mean()
        # KL(q(z_c,z_s|x,y,e) || p(z_c,z_s|y,e))
        prior_parent, prior_child = self.prior(y, e, False)
        kl_parent = D.kl_divergence(posterior_parent, prior_parent).mean()
        kl_child = D.kl_divergence(posterior_child, prior_child).mean()
        kl = kl_parent + kl_child
        prior_reg = torch.norm(torch.hstack((prior_parent.loc, prior_child.loc)), dim=1).mean()
        loss = -log_prob_x_z + kl + self.prior_reg_mult * prior_reg
        return loss

    def y_loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_parent = self.encoder(x, y, e, True)
        z_parent = self.sample_z(posterior_parent)
        # E_q(z_c|x)[log p(y|z_c)]
        y_pred = self.classifier(z_parent).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x,y,e) || p(z_c,z_s|y,e))
        prior_parent = self.prior(y, e, True)
        kl_parent = D.kl_divergence(posterior_parent, prior_parent).mean()
        prior_reg = torch.norm(prior_parent.loc, dim=1).mean()
        loss = -log_prob_y_zc + kl_parent + self.prior_reg_mult * prior_reg
        return loss

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if batch_idx % 2 == 0:
            loss = self.x_loss(x, y, e)
        else:
            loss = self.y_loss(x, y, e)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, e, c, s = batch
        y_pred = self.classify(x)
        if dataloader_idx == 0:
            self.val_acc.update(y_pred, y)
        else:
            assert dataloader_idx == 1
            self.test_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())
        self.log('test_acc', self.test_acc.compute())

    def classify(self, x):
        x = self.encoder.encoder_cnn(x).flatten(start_dim=1)
        parent_dist = self.encoder.parent_dist(x)
        z_parent = parent_dist.loc
        y_pred = self.classifier(z_parent).view(-1)
        return y_pred

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        y_pred = self.classify(x)
        self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)