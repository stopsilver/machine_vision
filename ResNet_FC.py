import torch
from sklearn.metrics import f1_score, roc_auc_score
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pickle
import os
from datetime import datetime

LOG_FILE = '191011_10.log'


class ResNet_FC(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNet_FC, self).__init__()

        self.conv_layer = nn.Sequential(*list(original_model.children())[:-2])
        for param in self.conv_layer:
            param.requires_grad = False
        self.gap_layer = nn.AvgPool2d(1, 1)
        # self.fc_layer1 = nn.Linear(512, 100)
        self.fc_layer2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        n_data = len(x)
        x = self.conv_layer(x)
        x = self.gap_layer(x).view(n_data, -1)
        # x = self.relu(self.fc_layer1(x))
        x = self.softmax(self.fc_layer2(x))
        return x


def unpickle(file):
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


class CustomizedDataset(Dataset):
    """ Diabetes dataset."""
    def __init__(self, transform=None, train=True, test='plane', partial=False):
        self.train = train  # training set or test set
        self.train_set = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.test_set = ['test_batch']
        self.transform = transform
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        test_idx = self.classes.index(test)

        self.data = []
        self.targets = []

        _path = '/home/jieun/Documents/machine_vision/data/'
        files = self.train_set if self.train else self.test_set

        # now load the picked numpy arrays
        for file_name in files:
            file_path = os.path.join(_path, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                data = entry['data']
                targets = targets = [1 if x != test_idx else 0 for x in entry['labels']]

                self.data.append(data)
                self.targets.extend(targets)

        self.len = len(self.targets)
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if partial:
            self.len = int(self.len / 10)
            self.data = self.data[:self.len]
            self.targets = self.targets[:self.len]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.len


def run():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CustomizedDataset(train=True, transform=transform, partial=True)
    train_len = len(trainset)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = CustomizedDataset(train=False, transform=transform, partial=True)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)

    resnet50 = models.resnet34(pretrained=True)
    res50_fc = ResNet_FC(resnet50)

    criterion = torch.nn.CrossEntropyLoss()
    losses = list()
    log = list()
    epochs = 50

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    res50_fc.to(device)
    total_iter = 0
    learning_rate = 0.001
    optimizer = torch.optim.SGD(res50_fc.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)
    start = datetime.now()

    # TRAIN
    for epoch in range(epochs):
        print('\n===> epoch %d' % epoch)

        running_loss = 0
        running_corrects = 0

        for idx, data in enumerate(train_loader, 0):
            # 순전파 단계: 모델에 x를 전달하여 예상하는 y 값을 계산합니다.
            images, labels = data[:2]
            images = images.to(device)
            labels = labels.to(device)

            y_pred = res50_fc(images)

            # 손실을 계산하고 출력합니다.
            loss = criterion(y_pred, labels)
            losses.append(loss.data.item())

            # 변화도를 0으로 만들고, 역전파 단계를 수행하고, 가중치를 갱신합니다.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # exp_lr_scheduler.step()
            total_iter += 1

            running_loss += loss.item() * images.size(0)
            pred = torch.max(y_pred, 1).indices
            running_corrects += torch.sum(pred == labels.data)

        epoch_loss = running_loss / train_len
        epoch_acc = running_corrects.double() /train_len

        # TEST
        org_labels = list()
        preds = list()
        test_corrects = 0

        test_len = len(testset)

        for idx, data in enumerate(test_loader, 0):
            images, labels = data[:2]
            images = images.to(device)
            labels = labels.to(device)
            loss = criterion(y_pred, labels)
            losses.append(loss.data.item())

            y_pred = res50_fc(images)
            pred = torch.max(y_pred, 1).indices

            org_labels.extend(labels)
            preds.extend(pred)

            test_corrects += torch.sum(pred == labels.data)

        test_acc = test_corrects.double() / test_len
        org_labels = [x.data.tolist() for x in org_labels]
        preds = [x.data.tolist() for x in preds]
        f1 = f1_score(org_labels, preds)
        auroc = roc_auc_score(org_labels, preds)

        print("{} epoch, takes {}, loss: {:.10f}, acc: {:.10f}".format(epoch, str(datetime.now() - start), epoch_loss, epoch_acc))
        print("test acc: {:.5f}, f1: {:.5f}, auroc: {:.5f}".format(test_acc, f1, auroc))

        di = {'epoch': epoch, 'train_acc': epoch_acc, 'test_acc': test_acc, 'train_loss': epoch_loss, 'f1': f1,
              'auroc': auroc, 'org_labels': org_labels, 'preds': preds}
        log.append(di)

    import joblib
    joblib.dump([losses, log], LOG_FILE)

    plt.plot(range(1, len(losses) + 1), losses)
    plt.show()


if __name__ == '__main__':
    run()