import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_filters, n_classes=1, input_shape=[32, 32]):
        """
        Simple CNN model used for CIFAR100
        Architecture uses VGG blocks with smaller filter numbers
        :param n_filters: list of filters numbers used in each block (length of n_filters equals to number of blocks)
        :param n_classes:
        """
        super().__init__()
        kernel = 3
        self.cnn_blocks = self.prepare_blocks(n_filters, kernel)
        linear_input = n_filters[-1] * input_shape[0] / pow(2, len(n_filters)) * \
                       input_shape[1] / pow(2, len(n_filters))
        self.classifier = nn.Sequential(
            nn.Linear(int(linear_input), n_classes*2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(n_classes*2, n_classes),
            # nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.cnn_blocks(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def prepare_blocks(self, n_filters, kernel, input_channels=3):
        """
        prepare cnn blocks (similar to vgg), according to input configuration
        :param n_filters: list of number of filters used in each block
        :param kernel: kernel used in Conv2D layers
        :return: sequential of stacked layers
        """
        modules = []
        for filters in n_filters:
            modules.append(nn.Conv2d(input_channels, filters, kernel, padding=1))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(filters, filters, kernel, padding=1))
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.MaxPool2d(2, 2))
            input_channels = filters

        return nn.Sequential(*modules)

    def _initialize_weights(self):
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
