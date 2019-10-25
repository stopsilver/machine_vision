from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

train_set = {
    'CIFAR10': ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
    'MNIST': 'training.pt',
    'FashionMNIST': 'training.pt',
}

test_set = {
    'CIFAR10': ['test_batch'],
    'MNIST': 'test.pt',
    'FashionMNIST': 'test.pt',
}

classes = {
    'CIFAR10': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    'MNIST': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    'FashionMNIST': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
}

path = {
    'CIFAR10': 'cifar-10-batches-py',
    'MNIST': 'MNIST/processed',
    'FashionMNIST': 'FashionMNIST/processed'
}


class BinaryDataset(Dataset):
    """ Diabetes dataset."""
    def __init__(self, flag, dp, transform=None, train=True, test='plane', partial=False):
        if flag not in classes.keys():
            raise AttributeError("flag must be 'CIFAR10', 'MNIST', or 'FashionMNIST'")

        self.flag = flag
        self.train = train  # training set or test set
        self.transform = transform

        self.train_set = train_set[flag]
        self.test_set = test_set[flag]
        self.classes = classes[flag]
        _path = os.path.join(dp, path[flag])

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        test_idx = self.classes.index(test)

        self.data = []
        self.targets = []
        files = self.train_set if self.train else self.test_set

        if self.flag == 'CIFAR10':
            for file_name in files:
                file_path = os.path.join(_path, file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    data = entry['data']
                    targets = [1 if x != test_idx else 0 for x in entry['labels']]

                    self.data.append(data)
                    self.targets.extend(targets)

            self.len = len(self.targets)
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            if train:
                fp = os.path.join(_path, self.train_set)
            else:
                fp = os.path.join(_path, self.test_set)

            self.data, _targets = torch.load(fp)
            self.targets = [1 if x != test_idx else 0 for x in _targets]
            self.len = len(self.targets)

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

        if self.flag == 'CIFAR10':
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.len


class AnomalyDataset(Dataset):
    """ Diabetes dataset."""
    def __init__(self, flag, dp, transform=None, train=True, test='cat', partial=False):
        if flag not in classes.keys():
            raise AttributeError("flag must be 'CIFAR10', 'MNIST', or 'FashionMNIST'")

        self.flag = flag
        self.train = train  # training set or test set
        self.transform = transform

        self.train_set = train_set[flag]
        self.test_set = test_set[flag]
        self.classes = classes[flag]
        _path = os.path.join(dp, path[flag])

        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        test_idx = self.classes.index(test)

        self.data = []
        self.targets = []

        if self.flag == 'CIFAR10':
            files = sorted([x for x in os.listdir(_path) if '.' not in x])

            # now load the picked numpy arrays
            for file_name in files:
                file_path = os.path.join(_path, file_name)
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                    data = entry['data']
                    targets = entry['labels']

                    if train:
                        data = [x for i, x in enumerate(data) if targets[i] != test_idx]
                        targets = [x for x in targets if x != test_idx]
                    else:
                        data = [x for i, x in enumerate(data) if targets[i] == test_idx]
                        targets = [x for x in targets if x == test_idx]

                    self.data.append(data)
                    self.targets.extend(targets)

            self.len = len(self.targets)
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        else:
            files = [self.train_set, self.test_set]
            data, targets = list(), list()
            for f in files:
                _data, _targets = torch.load(os.path.join(_path, f))

                if train:
                    _data = _data[_targets != test_idx]
                    _targets = _targets[_targets != test_idx]
                else:
                    _data = _data[_targets == test_idx]
                    _targets = _targets[_targets == test_idx]
                data.append(_data)
                targets.append(_targets)

            self.data = torch.cat(data)
            self.targets = torch.cat(targets)
            self.len = len(self.data)

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

        if self.flag == 'CIFAR10':
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.len


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    p = '/home/jieun/Documents/machine_vision/data'
    # trainset = AnomalyDataset(flag='CIFAR10', dp=p, train=True, test='cat', transform=transform)
    trainset = AnomalyDataset(flag='MNIST', dp=p, train=True, test='0', transform=transform)

    # trainset = BinaryDataset(flag='CIFAR10', dp=p, train=True, test='cat', transform=transform)
    # trainset = BinaryDataset(flag='MNIST', dp=p, train=True, test='0', transform=transform)

    train_len = len(trainset)
    train_loader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2)

    for idx, data in enumerate(train_loader, 0):
        if idx == 0:
            images, _ = data[:2]
        else:
            break

