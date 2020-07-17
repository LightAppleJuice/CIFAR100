import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.mfm import MFM


class CNN(nn.Module):
    def __init__(self, n_filters, n_classes=1, input_shape=[32, 32], mfm=True):
        """
        Simple CNN model used for CIFAR100
        Architecture uses VGG blocks with smaller filter numbers
        :param n_filters: list of filters numbers used in each block (length of n_filters equals to number of blocks)
        :param n_classes:
        """
        super().__init__()
        kernel = 3
        if mfm:
            cnn_blocks = self.prepare_mfm_blocks(n_filters, kernel)
        else:
            cnn_blocks = self.prepare_blocks(n_filters, kernel)

        modules = []
        for i in range(len(cnn_blocks) - 1):
            modules.extend(cnn_blocks[i])
            # modules.append(nn.MaxPool2d(2, 2))
            modules.append(nn.AvgPool2d(2, 2))
        modules.extend(cnn_blocks[-1])

        self.cnn_layers = nn.Sequential(*modules)
        # last cnn pooling
        self.cnn_pool = nn.AvgPool2d(2, 2)
        # self.cnn_pool = nn.MaxPool2d(2, 2)

        linear_input = n_filters[-1] * input_shape[0] / pow(2, len(n_filters)) * \
                       input_shape[1] / pow(2, len(n_filters))
        if mfm:
            self.fc = MFM(int(linear_input), n_classes * 2, type='linear')
            self.dropout = nn.Dropout(p=0.7)
        else:
            self.fc = nn.Sequential(
                nn.Linear(int(linear_input), n_classes*2),
                nn.ReLU(True))
            self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(n_classes * 2, n_classes)

        self.initialize_weights()

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.cnn_pool(x)
        # x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def prepare_blocks(self, n_filters, kernel, input_channels=3):
        """
        prepare cnn blocks (similar to vgg), according to input configuration
        :param n_filters: list of number of filters used in each block
        :param kernel: kernel used in Conv2D layers
        :return: list of blocks with sequentials of stacked layers without pooling layesr
        """
        blocks = []
        for filters in n_filters:
            modules = []
            modules.append(nn.Conv2d(input_channels, filters, kernel, padding=1))
            modules.append(nn.BatchNorm2d(filters))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(filters, filters, kernel, padding=1))
            modules.append(nn.BatchNorm2d(filters))
            modules.append(nn.ReLU(inplace=True))
            input_channels = filters
            blocks.append(modules)

        return blocks

    def prepare_mfm_blocks(self, n_filters, kernel, input_channels=3):
        """
        prepare cnn blocks with maf activations, according to input configuration
        :param n_filters: list of number of filters used in each block
        :param kernel: kernel used in Conv2D layers
        :return: sequential of stacked layers
        """
        blocks = []
        for filters in n_filters:
            modules = []
            modules.append(MFM(int(input_channels), filters, type='conv', kernel_size=kernel, padding=1)),
            modules.append(MFM(filters, filters, type='conv', kernel_size=kernel, padding=1)),
            input_channels = filters
            blocks.append(modules)
        return blocks

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
