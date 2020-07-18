import os
import shutil
import logging
import datetime
import random
import torchvision
import time
import numpy as np

from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import albumentations as A

from cifar_trainer import CifarTrainer
from utils import config_parser
from nn.cnn import *
from nn.resnet import *
from utils.lr_range_tester import LRFinder

from cifar_custom_dataset import CIFAR100

def lr_range_test():
    fig, ax = plt.subplots()
    lr_finder = LRFinder(model, optimizer, loss_function)
    lr_finder.range_test(train_loader, accumulation_steps=2,
                         val_loader=test_loader, start_lr=1e-5,
                         end_lr=10, num_iter=1000,
                         step_mode="exp", diverge_th=100)
    a = lr_finder.history['loss_train']
    arr = [len(a)]
    arr.extend([i for i, d in enumerate(a) if d >= 7 or np.isnan(d)])
    arr.extend([i for i, d in enumerate(lr_finder.history["loss_val"]) if d >= 7 or np.isnan(d)])
    b = min(arr)
    ax = lr_finder.plot(ax=ax, skip_start=0, skip_end=len(a)-b)
    # fig.show()
    fig.savefig(os.path.join(out_dir, 'lr_range_test_cnn_mfm_sgd.png'))
    lr_finder.reset()


if __name__ == '__main__':
    root_path = './data'
    out_dir = './results'
    config_name = './configs/cnn_mfm.json'
    run_lr_range_test = False
    run_training = True

    cfg = config_parser.parse_config(config_name)
    params = cfg.train_params

    manualSeed = 111

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    experiment_name = os.path.basename(config_name).split('.')[0] + '_lsoftmax'
    out_dir = '_'.join([os.path.join(out_dir, experiment_name), now.strftime("%m_%d_%H")])
    print('Find log in '.format())

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

    cifar_mean = torch.tensor([0.5071, 0.4866, 0.4409])
    cifar_std = torch.tensor([0.2673, 0.2564, 0.2761])

    # train_data = CIFAR100(root_path, train=True, download=False,
    #                       custom_aug=A.Compose({A.FancyPCA()}),
    #                       transform=torchvision.transforms.Compose([
    #                           torchvision.transforms.ToTensor(),
    #                           torchvision.transforms.Normalize(mean=cifar_mean, std=cifar_std)
    #                       ])
    #                       )

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
                              num_workers=params.num_workers)
    test_loader = DataLoader(test_data, batch_size=params.batch_size_train, shuffle=False,
                             num_workers=params.num_workers)

    #######################################
    # 2. Initialization
    #######################################

    device = torch.device("cuda:{}".format(params.cuda_number) if torch.cuda.is_available() else "cpu")

    softmax = 'softmax'
    if params.loss == 'LsoftmaxLoss':
        softmax = 'lsoftmax'

    if params.model == "CNN":
        model = CNN(n_filters=params['model_params'], n_classes=100, mfm=False, softmax=softmax,
                    norm_embds=False, device=device)
    elif params.model == "CNN_mfm":
        model = CNN(n_filters=params['model_params'], n_classes=100, softmax=softmax, norm_embds=False, device=device)
    elif params.model == "ResNet":
        model = ResNet(BasicBlock, params.model_params, num_classes=100, softmax=softmax, device=device)
    else:
        raise Exception('Unknown architecture. Use one of CNN, CNN_mfm, ResNet')

    model.eval()
    summary(model, input_size=(3, 32, 32))

    if params.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=5e-4)
    elif params.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4,
                              nesterov=True)
    else:
        print('Unknown optimizer. SGD is used instead')
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9, weight_decay=5e-4,
                              nesterov=True)

    loss_function = None
    if params.loss == 'CrossEntropyLoss' or params.loss == 'LsoftmaxLoss' or params.loss == 'SoftmaxLoss':
        loss_function = nn.CrossEntropyLoss()
    elif params.loss == 'NLLLoss':
        loss_function = nn.NLLLoss()
    else:
        raise Exception('Unknown type of loss')

    #######################################
    # 3. LR range test
    #######################################
    if run_lr_range_test:
        lr_range_test()
    #######################################
    # 4. Training
    #######################################
    if run_training:
        if params.lr_scheduler == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=params.learning_rate_step, gamma=0.1)
        elif params.lr_scheduler == 'MultiStepLR':
            scheduler = lr_scheduler.MultiStepLR(optimizer, [80, 120, 200, 250], gamma=0.1)
        elif params.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True,
                                                       threshold=0.0001, min_lr=0, eps=1e-04)
        elif params.lr_scheduler == 'OneCycleLR':
            change_lr_during_epoch = True
            # scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1.5e-2,
            #                                     steps_per_epoch=len(train_loader),
            #                                     div_factor=10,
            #                                     final_div_factor=100,
            #                                     epochs=params.num_epoch,
            #                                     )

            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=3e-1,
                                                steps_per_epoch=len(train_loader),
                                                div_factor=10,
                                                final_div_factor=1000,
                                                epochs=params.num_epoch)
        else:
            raise Exception('Unknown type of lr scheduler')

        trainer = CifarTrainer(model=model, optimizer=optimizer, criterion=loss_function,
                               snapshot_dir=os.path.join(out_dir, 'snapshots'),
                               log_dir=out_dir,
                               result_dir=out_dir,
                               device=device,
                               scheduler=scheduler if change_lr_during_epoch else None)

        tic = time.perf_counter()
        for epoch in range(params.num_epoch):
            tic = time.perf_counter()
            test_acc, test_mean_loss = trainer.test_model(test_loader,
                                                          iteration=epoch,
                                                          epoch=epoch, batch_idx=0,
                                                          mark='Test')

            train_acc, train_mean_loss = trainer.test_model(train_loader,
                                                            iteration=epoch,
                                                            epoch=epoch, batch_idx=0,
                                                            mark='Train')
            logger.info('Training epoch {}/{}, lr: {}'.format(epoch, params.num_epoch, scheduler.get_lr()))
            trainer.train_epoch(train_loader, test_loader, epoch,
                                train_acc_check=None, test_acc_check=None)
            if not change_lr_during_epoch:
                scheduler.step()
            toc = time.perf_counter()
            logger.info(f"Finished in {(toc - tic) / ((epoch+1) * 60):0.4f} minutes")



