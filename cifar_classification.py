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
from utils import config_parser, model_utils
from nn.cnn import *
from nn.cnn_margin_softmax import *
from nn.resnet import *
from nn.wide_resnet import *
from nn import losses
from utils.lr_range_tester import LRFinder
from six.moves import urllib

from cifar_custom_dataset import CIFAR100

proxy = urllib.request.ProxyHandler({'http': 'proxy.stc:3128', 'https': 'proxy.stc:3128'})
# construct a new opener using your proxy settings
opener = urllib.request.build_opener(proxy)
# install the openen on the module-level
urllib.request.install_opener(opener)

def lr_range_test():
    logger.info('LR_learning test')

    lr_finder = LRFinder(model, optimizer, loss_function)
    lr_finder.range_test(train_loader, accumulation_steps=1,
                         val_loader=test_loader, start_lr=1e-6,
                         end_lr=10, num_iter=500,
                         step_mode="exp", diverge_th=100)
    a = lr_finder.history['loss_train']
    arr = [len(a)]
    arr.extend([i for i, d in enumerate(a) if d >= 7 or np.isnan(d)])
    arr.extend([i for i, d in enumerate(lr_finder.history["loss_val"]) if d >= 7 or np.isnan(d)])
    b = min(arr)
    fig, ax = plt.subplots()
    ax = lr_finder.plot(ax=ax, skip_start=0, skip_end=len(a)-b)
    # fig.show()
    fig.savefig(os.path.join(out_dir, 'lr_range_test.png'))

    fig, ax = plt.subplots()
    ax = lr_finder.plot(ax=ax, skip_start=0, skip_end=len(a) - b, log_lr=False)
    fig.savefig(os.path.join(out_dir, 'lr_range_test_linear.png'))
    lr_finder.reset()


if __name__ == '__main__':
    root_path = './data'
    out_dir = './results'
    # config_name = './configs/cnn_mfm.json'
    # config_name = './configs/resnet_wide.json'
    config_name = './configs/resnet_wide_lsoftmax.json'
    # config_name = './configs/resnet18.json'
    # config_name = './configs/resnet18_pretrained.json'
    # config_name = './configs/cnn_mfm_norm_embds.json'

    run_lr_range_test = False
    run_training = True

    cfg = config_parser.parse_config(config_name)
    params = cfg.train_params

    manualSeed = 111
    # manualSeed=None

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

    experiment_name = os.path.basename(config_name).split('.')[0] + '_beta0'
    out_dir = '_'.join([os.path.join(out_dir, experiment_name), now.strftime("%m_%d_%H")])
    print('Find log in {}'.format(out_dir))

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    logger = logging.getLogger("CIFAR")
    logger.setLevel(logging.INFO)

    # logging to file
    fileHandler = logging.FileHandler(os.path.join(out_dir, 'training.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # logging to stdout
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

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
                                                   # torchvision.transforms.Resize([224, 224]), # for Imagenet pretrained
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(mean=cifar_mean, std=cifar_std)

                                               ]))

    test_data = torchvision.datasets.CIFAR100(root_path, train=False, download=False,
                                              transform=torchvision.transforms.Compose([
                                                  # torchvision.transforms.Resize([224, 224]), # for Imagenet pretrained
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(mean=cifar_mean, std=cifar_std),
                                              ]))

    train_loader = DataLoader(train_data, batch_size=params.batch_size_train, shuffle=True,
                              num_workers=params.num_workers)
    test_loader = DataLoader(test_data, batch_size=params.batch_size_train, shuffle=False,
                             num_workers=params.num_workers)

    #######################################
    # 2. Initialization
    #######################################

    device = torch.device("cuda:{}".format(params.cuda_number) if torch.cuda.is_available() else "cpu")

    if params.model == "CNN":
        model = CNN(n_filters=params['model_params'], n_classes=100, mfm=False, norm_embds=params.norm_embds)
    elif params.model == "CNN_lsoftmax":
        model = MarginSoftmaxCNN(n_filters=params['model_params'], n_classes=100, margin=params.margin, mfm=False)
    elif params.model == "CNN_mfm":
        model = CNN(n_filters=params['model_params'], n_classes=100, norm_embds=params.norm_embds)
    elif params.model == "CNN_mfm_lsoftmax":
        model = MarginSoftmaxCNN(n_filters=params['model_params'], n_classes=100, margin=params.margin)
    elif params.model == "origResNet18":
        model = torchvision.models.resnet18(num_classes=100)
    elif params.model == "ResNet18":
        model = ResNet(BasicBlock, params.model_params, num_classes=100)
    elif params.model == "WideResNet":
        model = WideResNet(WideBasicBlock, params.model_params, num_classes=100, k=params.width)
    elif params.model == "WideResNet_lsoftmax":
        model = WideResNet(WideBasicBlock, params.model_params, num_classes=100, k=params.width, margin=params.margin)
    else:
        raise Exception('Unknown architecture. Use one of CNN, CNN_mfm, ResNet')

    if run_training:
        unfreeze_in = None
        if params.use_pretrained and os.path.exists(params.use_pretrained):
            model_utils.load_model(model, params.use_pretrained)
        elif params.model == "origResNet18" and params.use_pretrained == "url":
            resnet18 = torchvision.models.resnet18(pretrained=True)
            model_utils.load_state_dict(model, resnet18.state_dict())
            model_utils.freeze_layers(model, nn.Linear)
            unfreeze_in = 5
        else:
            logger.info('No pretrained model was found.')

    model.eval()
    model.to(device)
    # summary(model, input_size=(3, 224, 224))
    summary(model, input_size=(3, 32, 32))

    # filter(lambda p: p.requires_grad, model.parameters())
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
    if params.loss == 'CrossEntropyLoss':
        loss_function = nn.CrossEntropyLoss()
    elif params.loss == 'NLLLoss':
        loss_function = nn.NLLLoss()
    elif params.loss == 'SoftCrossEntropyLoss':
        loss_function = losses.SoftCrossEntropyLoss(label_smoothing=0.1, num_classes=100)
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
        change_lr_during_epoch = False
        if params.lr_scheduler == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=params.learning_rate_step, gamma=0.1)
        elif params.lr_scheduler == 'MultiStepLR':
            # scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 100], gamma=0.1)
            scheduler = lr_scheduler.MultiStepLR(optimizer, [40, 80], gamma=0.1)
        elif params.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True,
                                                       threshold=0.0001, min_lr=0, eps=1e-04)
        elif params.lr_scheduler == 'OneCycleLR':
            change_lr_during_epoch = True
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=params.max_lr,
                                                steps_per_epoch=len(train_loader),
                                                div_factor=params.div_factor,
                                                final_div_factor=params.final_div_factor,
                                                epochs=params.num_epoch,
                                                # anneal_strategy='linear'
                                                )

        else:
            raise Exception('Unknown type of lr scheduler')

        trainer = CifarTrainer(model=model, optimizer=optimizer, criterion=loss_function,
                               snapshot_dir=os.path.join(out_dir, 'snapshots'),
                               log_dir=out_dir,
                               result_dir=out_dir,
                               device=device,
                               scheduler=scheduler,
                               change_lr_during_epoch=change_lr_during_epoch)

        tic = time.perf_counter()
        for epoch in range(params.num_epoch):
            if epoch == unfreeze_in:
                model_utils.unfreeze_layers(model)
                # optimizer.add_param_group({'params': parameters})

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

            if params.do_mixup:
                trainer.train_mixup_epoch(train_loader, test_loader, epoch,
                                    train_acc_check=None, test_acc_check=None)
            else:
                trainer.train_epoch(train_loader, test_loader, epoch,
                                    train_acc_check=None, test_acc_check=None)
            if not change_lr_during_epoch:
                scheduler.step()
            toc = time.perf_counter()
            logger.info(f"Finished in {(toc - tic) / ((epoch+1) * 60):0.4f} minutes")





