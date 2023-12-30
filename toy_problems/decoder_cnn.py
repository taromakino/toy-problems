import torch.nn as nn


class DecoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.module_list(x)