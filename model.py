import torch
from torch import nn
from torch.nn import functional as F

import constants


class NetBlock(nn.Module):
    def __init__(self, n_chans, batchnorm_needed, kernel_size):
        super().__init__()
        self.batchnorm_needed = batchnorm_needed
        self.conv = nn.Conv2d(n_chans, n_chans, kernel_size=kernel_size)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        if batchnorm_needed:
            self.batch_norm = nn.BatchNorm2d(num_features=n_chans)

    def forward(self, x):
        res = self.conv(x)
        if self.batchnorm_needed:
            res = self.batch_norm(res)

        res = torch.relu(res)
        res = self.pool(res)

        return res

class ConvNet(nn.Module):
    def __init__(self, n_channels=64, n_layers=3, kernel_size=5):
        super().__init__()
        self.n_chans = n_channels
        self.n_layers = n_layers

        self.conv1 = nn.Conv2d(3, self.n_chans, kernel_size)
        self.conv_hidden = nn.Sequential(*(n_layers * [NetBlock(self.n_chans)]))
        self.fc_out = nn.Linear(64, len(constants.TARGET_CATEGORIES))

    def forward(self, x):
        res = self.conv1(x)
        res = torch.relu(res)
        res = self.conv_hidden(res)
        res = torch.flatten(res)
        res = self.fc_out(res)
        res = F.log_softmax(res, dim=1)
        return res
