import torch
import torchvision.transforms as transforms
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from util.dataset import AnomalyDataset
from datetime import datetime
from module.conv_ae import ConvAEUPSC


def run():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    p = '/home/jieun/Documents/machine_vision/data'
    # trainset = AnomalyDataset(flag='MNIST', dp=p, train=True, test='5', transform=transform)
    trainset = AnomalyDataset(flag='CIFAR10', dp=p, train=True, transform=transform)
    # testset = AnomalyDataset(flag='MNIST', dp=p, train=False, test='5', transform=transform)
    testset = AnomalyDataset(flag='CIFAR10', dp=p, train=False, transform=transform)

    train_len = len(trainset)
    train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    test_len = len(testset)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    learning_rate = 1e-3

    autoencoder = ConvAEUPSC()
    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)

    losses = list()
    epochs = 20

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    autoencoder.to(device)
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
            # images = images.view(n_data, -1)

            images = images.to(device)

            y_pred = autoencoder(images)

            # 손실을 계산하고 출력합니다.
            # loss = PairwiseDistance(2).forward(y_pred, images)
            loss = criterion(y_pred, images)
            losses.append(torch.mean(loss))
            # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total_iter += 1
            n_iter += 1

            running_loss += torch.mean(loss)

        epoch_loss = running_loss / n_iter
        print("{} epoch, takes {}, loss: {:.10f}".format(epoch, str(datetime.now() - start), epoch_loss))

    # TEST
    reconstuction_error = list()
    total_labels = [1] * train_len + [0] * test_len

    for idx, data in enumerate(train_loader, 0):
        images, _ = data[:2]
        input_images = images.to(device)

        y_pred = autoencoder(input_images)

        pic = y_pred.cpu().view(-1, 3, 32, 32).detach().numpy().transpose((0, 2, 3, 1))
        # pic = y_pred.cpu().view(-1, 1, 28, 28).detach().numpy().transpose((0, 2, 3, 1))
        if idx <= 5:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(transforms.ToPILImage()(images[0]), interpolation='bicubic')
            fig.add_subplot(1, 2, 2)
            plt.imshow(pic[0], interpolation='bicubic')

            # plt.imshow(pic[0], interpolation='bicubic', cmap=plt.cm.gray_r)
            plt.savefig('/home/jieun/Documents/machine_vision/imgs/{}.png'.format('conv_ae2_normal' + str(idx)))

        sub_reconstuction_error = list()
        for i in range(len(y_pred)):
            sub_reconstuction_error.append(torch.dist(input_images[i], y_pred[i]).item())
        reconstuction_error.extend(sub_reconstuction_error)

    for idx, data in enumerate(test_loader, 0):
        images, _ = data[:2]
        input_images = images.to(device)

        y_pred = autoencoder(input_images)

        pic = y_pred.cpu().view(-1, 3, 32, 32).detach().numpy().transpose((0, 2, 3, 1))
        # pic = y_pred.cpu().view(-1, 1, 28, 28).detach().numpy().transpose((0, 2, 3, 1))
        if idx <= 5:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            plt.imshow(transforms.ToPILImage()(images[0]), interpolation='bicubic')
            fig.add_subplot(1, 2, 2)
            plt.imshow(pic[0], interpolation='bicubic')
            # plt.imshow(pic[0].reshape(28, 28), interpolation='bicubic', cmap=plt.cm.gray_r)

            plt.savefig('/home/jieun/Documents/machine_vision/imgs/{}.png'.format('conv_ae2_anomaly'+str(idx)))

        sub_reconstuction_error = list()
        for i in range(len(y_pred)):
            sub_reconstuction_error.append(torch.dist(input_images[i], y_pred[i]).item())

        reconstuction_error.extend(sub_reconstuction_error)

    di = {'total_label': total_labels, 'reconstuction_error': reconstuction_error}
    from util.validation import plot_re
    plot_re(di)

    plt.plot(range(1, len(losses) + 1), losses)
    plt.show()


if __name__ == '__main__':
    run()