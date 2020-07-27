###
# Large margin softmax implementation from https://github.com/amirhfarzaneh/lsoftmax-pytorch
###
import math

import torch
from torch import nn
from scipy.special import binom

import logging
logger = logging.getLogger("CIFAR.Lsoftmax")


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        # if Lsoftmax has difficulty converging
        # self.beta_max = 300.0
        # self.beta_min = 6.0 for cnn
        self.beta_min = 2.0
        self.beta_max = 300.0
        self.it = 0

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        # self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        # self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        # self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        # self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...

        self.register_buffer('C_m_2n', torch.Tensor(binom(margin, range(0, margin + 1, 2))))  # C_m{2n}
        self.register_buffer('cos_powers', torch.Tensor(range(self.margin, -1, -2)))# m - 2n
        self.register_buffer('sin2_powers', torch.Tensor(range(len(self.cos_powers)))) # n
        self.register_buffer('signs', torch.ones(margin // 2 + 1)) # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            # beta = max(self.beta, self.beta_min)
            beta = max(self.beta_min, self.beta_max / (1 + 0.01 * self.it))
            logger.info('Iteration {}, beta parameter: {}'.format(self.it, beta))

            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            # self.beta *= self.scale
            self.it += 1
            return logit
        else:
            assert target is None
            return input.mm(self.weight)