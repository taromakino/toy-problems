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
from utils.nn_utils import SkipMLP, one_hot, arr_to_cov, batch_block_diag


class Encoder(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.encoder_cnn = EncoderCNN()
        self.mu = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)
        self.low_rank = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size * rank)
        self.diag = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.encoder_cnn(x).view(batch_size, -1)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu(x, y_one_hot, e_one_hot)
        low_rank = self.low_rank(x, y_one_hot, e_one_hot)
        low_rank = low_rank.reshape(batch_size, 2 * self.z_size, self.rank)
        diag = self.diag(x, y_one_hot, e_one_hot)
        cov = arr_to_cov(low_rank, diag)
        return D.MultivariateNormal(mu, cov)


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = SkipMLP(2 * z_size, h_sizes, IMG_DECODE_SIZE)
        self.decoder_cnn = DecoderCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).view(batch_size, *IMG_DECODE_SHAPE)
        x_pred = self.decoder_cnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank, init_sd):
        super().__init__()
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.low_rank_causal = nn.Parameter(torch.zeros(N_ENVS, z_size, rank))
        self.diag_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.normal_(self.mu_causal, 0, init_sd)
        nn.init.normal_(self.low_rank_causal, 0, init_sd)
        nn.init.normal_(self.diag_causal, 0, init_sd)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, init_sd)
        nn.init.normal_(self.low_rank_spurious, 0, init_sd)
        nn.init.normal_(self.diag_spurious, 0, init_sd)

    def forward(self, y, e):
        mu_causal = self.mu_causal[e]
        mu_spurious = self.mu_spurious[y, e]
        mu = torch.hstack((mu_causal, mu_spurious))
        cov_causal = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        cov_spurious = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        cov = batch_block_diag(cov_causal, cov_spurious)
        return D.MultivariateNormal(mu, cov)

class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, y_mult, beta, prior_reg_mult, init_sd, lr, weight_decay, lr_infer,
            n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.beta = beta
        self.prior_reg_mult = prior_reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, init_sd)
        # p(y|z)
        self.classifier = nn.Linear(z_size, 1)
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, dim = mu.shape
        epsilon = torch.randn(batch_size, dim, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        prior_reg = (prior_dist.loc ** 2).mean()
        return log_prob_x_z, log_prob_y_zc, kl, prior_reg, y_pred

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_reg, y_pred = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.prior_reg_mult * prior_reg
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_reg, y_pred = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.prior_reg_mult * prior_reg
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log('val_kl', kl, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
        self.val_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def init_z(self, x, y_value, e_value):
        batch_size = len(x)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        posterior_dist = self.encoder(x, y, e)
        return nn.Parameter(posterior_dist.loc.detach())

    def infer_loss(self, x, y, e, z):
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
        # log q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
        log_prob_z = posterior_dist.log_prob(z)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - log_prob_z
        return loss

    def opt_infer_loss(self, x, y_value, e_value):
        batch_size = len(x)
        z_param = self.init_z(x, y_value, e_value)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        optim = AdamW([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss = self.infer_loss(x, y, e, z_param)
            loss.mean().backward()
            optim.step()
        return loss.detach().clone()

    def classify(self, x):
        loss_candidates = []
        y_candidates = []
        for y_value in range(N_CLASSES):
            for e_value in range(N_ENVS):
                loss_candidates.append(self.opt_infer_loss(x, y_value, e_value)[:, None])
                y_candidates.append(y_value)
        loss_candidates = torch.hstack(loss_candidates)
        y_candidates = torch.tensor(y_candidates, device=self.device)
        y_pred = y_candidates[loss_candidates.argmin(dim=1)]
        return y_pred

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        with torch.set_grad_enabled(True):
            y_pred = self.classify(x)
            self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)