import torch
import torch.nn as nn

# TODO: transpose conv와 upscaling 성능 비교


class ConvAETrans(torch.nn.Module):
    def __init__(self):
        super(ConvAETrans, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 10, 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(10, 5, 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(5, 10, 3, stride=3, padding=4), nn.ReLU(True),
            # nn.ConvTranspose2d(10, 3, 4, stride=2, padding=1), nn.Sigmoid(),
            # nn.ConvTranspose2d(8, 3, 3, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvAEUPSC(torch.nn.Module):
    def __init__(self):
        super(ConvAEUPSC, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(8, , 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(size=(8, 8), mode='nearest'),
            nn.Conv2d(32, 16, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Upsample(size=(16, 16), mode='nearest'),
            nn.Conv2d(16, 8, 5, stride=1, padding=1), nn.ReLU(True),
            nn.Upsample(size=(32, 32), mode='nearest'),
            nn.Conv2d(8, 3, 3, stride=1, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

