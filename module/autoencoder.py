import torch
import torch.nn as nn


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1024*3, 28 * 28), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(28 * 28, 1024 * 3), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

