import torchvision
import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
import random
import torch

from utils.lr_range_tester import LRFinder
from torch import optim

from torch.utils.data import DataLoader
from nn.cnn import *
from nn.resnet import *

from utils import config_parser

if __name__ == '__main__':
    root_path = './data'
    out_dir = './results'
    config_name = './configs/cnn_mfm.json'

    cfg = config_parser.parse_config(config_name)
    params = cfg.train_params

    manualSeed = 111

    experiment_name = os.path.basename(config_name).split('.')[0] + '_lr_range_test'
    out_dir = os.path.join(out_dir, experiment_name)
    print('Find log in '.format())

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    #######################################
    # 1. Data preparation
    #######################################

    cifar_mean = torch.tensor([0.5071, 0.4866, 0.4409])
    cifar_std = torch.tensor([0.2673, 0.2564, 0.2761])

    train_data = torchvision.datasets.CIFAR100(root_path, train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.RandomCrop(32, padding=4),
                                                   torchvision.transforms.RandomHorizontalFlip(),
                                                   # torchvision.transforms.ColorJitter()
                                                   torchvision.transforms.RandomRotation(10),
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(mean=cifar_mean, std=cifar_std)
                                               ]))

    test_data = torchvision.datasets.CIFAR100(root_path, train=False, download=False,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=cifar_mean, std=cifar_std)
                                              ]))

    train_loader = DataLoader(train_data, batch_size=params.batch_size_train, shuffle=True,
                              num_workers=1)
    test_loader = DataLoader(test_data, batch_size=params.batch_size_train, shuffle=False,
                             num_workers=1)

    #######################################
    # 2. Model & Trainer initialization
    #######################################
    if params.model == "CNN":
        model = CNN(n_filters=params['model_params'], n_classes=100, mfm=False)
    elif params.model == "CNN_mfm":
        model = CNN(n_filters=params['model_params'], n_classes=100)
    elif params.model == "ResNet":
        model = ResNet(BasicBlock, params.model_params, num_classes=100)

    loss_function = nn.CrossEntropyLoss()

    if params.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=1e-7, momentum=0.9, weight_decay=1e-3, nesterov=True)
    if params.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-3)

    fig, ax = plt.subplots()
    lr_finder = LRFinder(model, optimizer, loss_function)
    lr_finder.range_test(train_loader, accumulation_steps=4,
                         val_loader=test_loader, start_lr=1e-5,
                         end_lr=10, num_iter=100,
                         step_mode="exp", diverge_th=100)
    a = lr_finder.history['loss_train']
    b = min([i for i, d in enumerate(a) if d >= 1e+20 or np.isnan(d)])
    ax = lr_finder.plot(ax=ax, skip_start=0, skip_end=b)
    fig.show()
    # fig.savefig(os.path.join(out_dir, 'lr_range_test_cnn_mfm_sgd.png'))
    lr_finder.reset() # to reset the model and optimizer to their initial state
    exit()