import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from encoder_cnn import IMG_ENCODE_SIZE, EncoderCNN
from torch.optim import AdamW
from torchmetrics import Accuracy


class ERM(pl.LightningModule):
    def __init__(self, parent_size, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = EncoderCNN()
        self.classifier = nn.Sequential(
            nn.Linear(IMG_ENCODE_SIZE, parent_size),
            nn.LeakyReLU(),
            nn.Linear(parent_size, 1)
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def forward(self, x, y, e, c, s):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        y_pred = self.classifier(x).view(-1)
        return y_pred, y

    def training_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        self.train_acc.update(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        self.val_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)