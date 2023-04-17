import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Dataset


def get_dataset_dl(config):
    if not config.auto.auto:
        print('Doing {}'.format(config.DATA.name))
    if config.DATA.name == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(config.DATA.data_path, train=True, download=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=config.TRAIN.batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(config.DATA.data_path, train=False, download=False,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=config.TEST.batch_size, shuffle=True, pin_memory=True)
        return train_loader, test_loader
    elif config.DATA.name == 'FashionMNIST':
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(config.DATA.data_path, train=True, download=False,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ])),
            batch_size=config.TRAIN.batch_size, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.FashionMNIST(config.DATA.data_path, train=False, download=False,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ])),
            batch_size=config.TEST.batch_size, shuffle=True, pin_memory=True)
        return train_loader, test_loader
    else:
        raise Exception("Dataset does not exist")


class pytorch_dataset(Dataset):

    def __init__(self, data_type="train", config=None):
        if data_type == 'train':
            file_path = config.DATA.dataset_list_train.format(config.DATA.name)
        else:
            file_path = config.DATA.dataset_list_test.format(config.DATA.name)
        self.full_filenames = []
        self.labels = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            words = line.split()
            self.full_filenames.append(words[0])
            self.labels.append(int(words[1]))
        self.transform = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.full_filenames)  # size of dataset

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image)  # Apply Specific Transformation to Image
        return image, self.labels[idx]


def get_dataset(config):
    if not config.auto.auto:
        print('Doing {}'.format(config.DATA.name))
    train_dataset = pytorch_dataset(data_type="train", config=config)
    test_dataset = pytorch_dataset(data_type="test", config=config)
    return train_dataset, test_dataset
