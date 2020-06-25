import os
import shutil
import logging
import datetime
import random
import torch
import torchvision

from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from cifar_trainer import CifarTrainer
from nn.CNN import CNN


if __name__ == '__main__':
    root_path = './data'
    out_dir = './results/baseline_cnn'

    params = {
        'batch_size_train': 256,
        'batch_size_test': 16,
        'num_cuda': None,
        'learning_rate': 0.01,
        'learning_rate_step': 3,
        'num_epoch': 15,
        'model_params': [6, 16],
        'optimizer': 'SGD'
    }

    manualSeed = 111

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    # out_dir = '_'. join([out_dir, now.strftime("%m_%d_%H_%M")])
    out_dir = '_'.join([out_dir, now.strftime("%m_%d_%H")])

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger("CIFAR")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(out_dir, 'training.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Initialization")

    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    #######################################
    # 1. Data preparation
    #######################################

    train_data = torchvision.datasets.CIFAR100(root_path, train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                               ]))
    test_data = torchvision.datasets.CIFAR100(root_path, train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                              ]))

    train_loader = DataLoader(train_data, batch_size=params['batch_size_train'], shuffle=True)
    test_loader = DataLoader(test_data, batch_size=params['batch_size_train'], shuffle=False)

    #######################################
    # 2. Model & Trainer initialization
    #######################################

    model = CNN(n_filters=params['model_params'], n_classes=100)

    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)

    device = torch.device("cuda:{}".format(params['cuda_number']) if torch.cuda.is_available() else "cpu")

    trainer = CifarTrainer(model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                           snapshot_dir=os.path.join(out_dir, 'snapshots'),
                           log_dir=out_dir,
                           result_dir=out_dir,
                           device=device)

    #######################################
    # 3. Training
    #######################################

    for epoch in range(params['num_epoch']):
        logger.info('Training epoch {}/{}'.format(epoch, params['num_epoch']))
        trainer.train_epoch(train_loader, test_loader, epoch,
                            train_acc_check=100, test_acc_check=100)
        trainer.adjust_learning_rate(epoch, step=params['learning_rate_step'], base_lr=params['learning_rate'])


