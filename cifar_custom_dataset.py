from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
from torch import functional as F

from utils.data_utils import show_img

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets import CIFAR100


class CIFAR100(CIFAR100):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR100` Dataset.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, custom_aug=None):
        super(CIFAR100, self).__init__(root, train=train, transform=transform, target_transform=target_transform,
                 download=download)

        self.aug = custom_aug

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        show_img(img)
        if self.aug is not None:
            img = self.aug(image=np.array(img))['image']
            show_img(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target