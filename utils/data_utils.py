import torchvision
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Mix up implementation
    :param x:
    :param y:
    :param alpha: mixup interpolation coefficient
    :param use_cuda:
    :return:
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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

