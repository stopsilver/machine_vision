# Based on https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.ipynb
import torch
from torch.autograd import Variable
from util.dataset import AnomalyDataset
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import math
from torch.utils.data import DataLoader


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape

    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)


def get_flatten_output():
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 4, 2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 128, 4, 2),
        torch.nn.LeakyReLU(),
        Flatten(),
    )
    return model(Variable(torch.rand(2,3,32,32))).size()

def get_reshape_output():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 7*7*128),
        torch.nn.ReLU(),
        Reshape((128,7,7,)),
        torch.nn.ConvTranspose2d(128, 64, 4, 2),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(64, 3, 4, 2, padding=3),
        torch.nn.Sigmoid()
    )
    return model(Variable(torch.rand(2,2))).size()


# Encoder and decoder use the DC-GAN architecture
class Encoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 64, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, padding=1),
            torch.nn.LeakyReLU(),
            Flatten(),
            torch.nn.Linear(8192, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, z_dim)
        ])

    def forward(self, x):
        # print("Encoder")
        # print(x.size())
        for layer in self.model:
            x = layer(x)
            # print(x.size())
        return x


class Decoder(torch.nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 8192),
            torch.nn.ReLU(),
            Reshape((128, 8, 8,)),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, padding=1),
            torch.nn.Sigmoid()
        ])

    def forward(self, x):
        # print("Decoder")
        # print(x.size())
        for layer in self.model:
            x = layer(x)
            # print(x.size())
        return x


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd


# Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
def convert_to_display(samples):
    print(type(samples), samples.shape)
    cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height, cnt, cnt, width])
    samples = np.transpose(samples, axes=[1, 0, 2, 3])
    samples = np.reshape(samples, [height * cnt, width * cnt])
    return samples


def train(
        dataloader,
        z_dim=2,
        n_epochs=20,
        use_cuda=True,
        print_every=1000,
        plot_every=1000
):
    model = Model(z_dim)
    if use_cuda:
        model = model.cuda()
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    i = -1
    for epoch in range(n_epochs):
        for images, labels in dataloader:
            i += 1
            optimizer.zero_grad()
            x = Variable(images, requires_grad=False)
            true_samples = Variable(
                torch.randn(200, z_dim),
                requires_grad=False
            )
            if use_cuda:
                x = x.cuda()
                true_samples = true_samples.cuda()
            z, x_reconstructed = model(x)
            mmd = compute_mmd(true_samples, z)
            nll = (x_reconstructed - x).pow(2).mean()
            loss = nll + mmd
            loss.backward()
            optimizer.step()

        print("Negative log likelihood is {:.5f}, mmd loss is {:.5f}".format(
                nll.item(), mmd.item()))

        target = images[:1]
        _input = target.cuda()
        z = model.encoder(_input)
        recon = model.decoder(z)
        recon = recon.permute(0, 2, 3, 1).contiguous().cpu().data.numpy()

        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.imshow(transforms.ToPILImage()(images[0]), interpolation='bicubic')
        fig.add_subplot(1, 2, 2)
        plt.imshow(recon[0], interpolation='bicubic')
        plt.show()

    return model


class Model(torch.nn.Module):
    def __init__(self, z_dim):
        super(Model, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed


if __name__ == '__main__':
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    #
    # model = Model(z_dim=2)
    #
    # trainset = AnomalyDataset(train=True, transform=transform)
    # train_len = len(trainset)
    # train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    #
    # testset = AnomalyDataset(train=False, transform=transform)
    # test_len = len(testset)
    # test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)
    #
    # z_dim = 2
    # model = train(train_loader, z_dim=z_dim, n_epochs=20)
    from torchvision.datasets import MNIST, FashionMNIST
    batch_size = 200
    mnist_train = torch.utils.data.DataLoader(
        MNIST("/home/jieun/Documents/machine_vision/data/MNIST", train=True, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
              ])),
        batch_size=batch_size, shuffle=True, num_workers=3,
        pin_memory=True
    )

    mnist_train = torch.utils.data.DataLoader(
        FashionMNIST("/home/jieun/Documents/machine_vision/data/FashionMNIST", train=True, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
              ])),
        batch_size=batch_size, shuffle=True, num_workers=3,
        pin_memory=True
    )