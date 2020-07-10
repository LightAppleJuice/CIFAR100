import torch
import torch.nn as nn


class MFM(nn.Module):
    """
    Max Feature Map activation function for case of conv and linear layers
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type='linear'):
        super(MFM, self).__init__()
        self.out_channels = out_channels
        if type == 'conv':
            self.filter = nn.Conv2d(in_channels, 2*out_channels,
                                    kernel_size=kernel_size, stride=stride, padding=padding)
        elif type == 'linear':
            self.filter = nn.Linear(in_channels, 2*out_channels)
        else:
            raise Exception('incorrect MFM layer type')

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

