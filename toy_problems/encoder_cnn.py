import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

    def forward(self, x):
        return self.module_list(x)