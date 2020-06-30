import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, n_filters, n_classes=1):
        super().__init__()
        kernel = 5
        self.cnn_block1 = nn.Sequential(
            nn.Conv2d(3, n_filters[0], kernel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv2d(n_filters[0], n_filters[1], kernel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fc1 = nn.Linear(n_filters[1] * 5 * 5, n_classes*2)
        self.fc2 = nn.Linear(n_classes*2, n_classes)

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x