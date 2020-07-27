import torch

from torch.nn import functional as F
from torch import nn


class AMSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin, s=30):
        super().__init__()
        assert margin > 0
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.s = s

        # Initialize parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def forward(self, x, target=None):
        eps = 1e-7

        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)

        if self.training:
            assert target is not None
            phi_theta = cos_theta - self.margin

            index = torch.zeros_like(cos_theta, dtype=torch.uint8)
            index.scatter_(1, target.data.view(-1, 1), 1)
            output = torch.where(index, phi_theta, cos_theta)

            return self.s * output
        else:
            assert target is None
            return cos_theta