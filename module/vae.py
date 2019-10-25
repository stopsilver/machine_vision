import torch
from torch.nn.modules import PairwiseDistance


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        input_layer = 3 * 1024
        layer1 = 784
        layer2 = 40

        self.en1 = nn.Linear(input_layer, layer1)
        self.en21 = nn.Linear(layer1, layer2)
        self.en22 = nn.Linear(layer1, layer2)
        self.de1 = nn.Linear(layer2, layer1)
        self.de2 = nn.Linear(layer1, input_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.recon_loss = nn.BCELoss(reduction='sum')

    def encode(self, x):
        x = self.relu(self.en1(x))
        return self.en21(x), self.en22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.relu(self.de1(z))
        return self.sigmoid(self.de2(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_ft(self, recon_x, x, mu, logvar, show=False):
        BCE = self.recon_loss(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        if show:
            print(BCE.item(), KLD.item())
        return BCE + KLD


class convVAE(torch.nn.Module):
    def __init__(self):
        super(convVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(3, 3, 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(3, 3, 3, stride=1, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            # nn.Upsample(size=(8, 8), mode='nearest'),
            # nn.Conv2d(3, 3, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Upsample(size=(16, 16), mode='nearest'),
            nn.Conv2d(3, 3, 3, stride=1, padding=1), nn.ReLU(True),
            nn.Upsample(size=(32, 32), mode='nearest'),
            nn.Conv2d(3, 3, 3, stride=1, padding=1), nn.Sigmoid()
        )
        self.recon_loss = nn.BCELoss(reduction='sum')

    def encode(self, x):
        x = self.relu(self.en1(x))
        return self.en21(x), self.en22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.relu(self.de1(z))
        return self.sigmoid(self.de2(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_ft(self, recon_x, x, mu, logvar, show=False):
        BCE = self.recon_loss(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        if show:
            print(BCE.item(), KLD.item())
        return BCE + KLD


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms
    import torch.nn as nn
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from util.dataset import AnomalyDataset
    from datetime import datetime

    transform = transforms.Compose([
        transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = AnomalyDataset(train=True, transform=transform)
    train_len = len(trainset)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = AnomalyDataset(train=False, transform=transform)
    test_len = len(testset)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    learning_rate = 1e-3
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    model.to(device)

    losses = list()
    epochs = 10

    total_iter = 0
    start = datetime.now()

    # TRAIN
    for epoch in range(epochs):
        print('\n===> epoch %d' % epoch)
        running_loss = 0
        n_iter = 0

        for idx, data in enumerate(train_loader, 0):
            # 순전파 단계: 모델에 x를 전달하여 예상하는 y 값을 계산합니다.
            images, _ = data[:2]
            n_data = images.shape[0]
            images_input = images.view(n_data, -1)
            images_input = images_input.to(device)

            recon_batch, mu, logvar = model(images_input)

            _flag = idx == 0

            loss = model.loss_ft(recon_batch, images_input, mu, logvar, show=_flag)
            loss.backward(retain_graph=True)
            losses.append(loss.item())

            # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_iter += 1
            n_iter += 1

            running_loss += loss.item()

        epoch_loss = running_loss / train_len
        print("{} epoch, takes {}, loss: {:.10f}".format(epoch, str(datetime.now() - start), epoch_loss))

    # TEST
    reconstuction_error = list()
    total_labels = [1] * train_len + [0] * test_len

    for idx, data in enumerate(train_loader, 0):
        images, _ = data[:2]

        n_data = images.shape[0]
        images_input = images.view(n_data, -1)

        images_input = images_input.to(device)

        recon_batch, mu, logvar = model(images_input)
        sub_reconstuction_error = PairwiseDistance(2).forward(recon_batch, images_input).tolist()
        reconstuction_error.extend(sub_reconstuction_error)

        pic = recon_batch.cpu().view(-1, 3, 32, 32).detach().numpy().transpose((0, 2, 3, 1))
        if idx <= 5:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(transforms.ToPILImage()(images[0]), interpolation='bicubic')
            fig.add_subplot(1, 2, 2)
            plt.imshow(pic[0], interpolation='bicubic')
            plt.show()

    for idx, data in enumerate(test_loader, 0):
        images, _ = data[:2]

        n_data = images.shape[0]
        images_input = images.view(n_data, -1)

        images_input = images_input.to(device)

        recon_batch, mu, logvar = model(images_input)

        pic = recon_batch.cpu().view(-1, 3, 32, 32).detach().numpy().transpose((0, 2, 3, 1))
        if idx <= 5:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(transforms.ToPILImage()(images[0]), interpolation='bicubic')
            fig.add_subplot(1, 2, 2)
            plt.imshow(pic[0], interpolation='bicubic')
            plt.show()

        sub_reconstuction_error = PairwiseDistance(2).forward(recon_batch, images_input).tolist()
        reconstuction_error.extend(sub_reconstuction_error)

    di = {'total_label': total_labels, 'reconstuction_error': reconstuction_error}

    from util.validation import plot_re
    plot_re('', data=di)

    plt.plot(range(1, len(losses) + 1), losses)
    plt.show()