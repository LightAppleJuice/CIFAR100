import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def get_mean_std(dataset, big_dataset=True):
    """
    Estimate mean and std for the dataset
    """
    if not big_dataset:
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        for data in loader:
            mean = data[0].mean([0, 2, 3])
            std = data[0].std([0, 2, 3])
            return mean, std
    else:
        mean = torch.zeros((3,), dtype=torch.float)
        std = torch.zeros((3,), dtype=torch.float)
        n = 0
        for num, data in enumerate(dataset):
            imgs, _ = data
            mean += imgs.sum([1, 2])
            std += torch.sum(imgs ** 2, dim=(1, 2))
            n += imgs.numel() // imgs.shape[0]
        n = float(n)
        mean = mean / n  # mean
        std = std / n - (mean ** 2)
        return mean, std.sqrt()

def show_img(img):
    plt.figure(figsize=(32, 32))
    plt.imshow(img)
    plt.show()
    print('.')


if __name__ == '__main__':
    root_path = '../data'
    small_dataset = False

    dataset = torchvision.datasets.CIFAR100(root_path, train=True, download=True,
                                            transform=torchvision.transforms.ToTensor())
    #
    mean, std = get_mean_std(dataset, big_dataset=False)
    print('Dataloader: {}, {}'.format(mean, std))
    mean, std = get_mean_std(dataset, big_dataset=True)
    print('Iteratively: {}, {}'.format(mean, std))

