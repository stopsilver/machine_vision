from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms


class BinaryDataset(Dataset):
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

        _path = '/home/jieun/Documents/machine_vision/data/cifar-10-batches-py'
        files = self.train_set if self.train else self.test_set

        # now load the picked numpy arrays
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

        plt.imshow(self.data[0], interpolation='bicubic')
        plt.show()

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

        plt.imshow(transforms.ToPILImage()(img[0]), interpolation='bicubic')
        plt.show()

        return img, target

    def __len__(self):
        return self.len


class AnomalyDataset(Dataset):
    """ Diabetes dataset."""
    def __init__(self, transform=None, train=True, test='ship'):
        self.train = train  # training set or test set
        self.transform = transform
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        test_idx = self.classes.index(test)

        self.data = []
        self.targets = []

        _path = '/home/jieun/Documents/machine_vision/data/cifar-10-batches-py'
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


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])

    trainset = AnomalyDataset(train=True, transform=transform)
    train_len = len(trainset)
    train_loader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2)

    for idx, data in enumerate(train_loader, 0):
        if idx == 0:
            images, _ = data[:2]
        else:
            break

