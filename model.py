import torch
from torch import nn
from torch.nn import functional as F

import constants


class NetBlock(nn.Module):
    def __init__(self, in_chans, out_chans, batchnorm_needed, kernel_size):
        super().__init__()
        self.batchnorm_needed = batchnorm_needed
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size)
        self.drop = nn.Dropout2d(p=0.5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if batchnorm_needed:
            self.batch_norm = nn.BatchNorm2d(num_features=out_chans)


    def forward(self, x):
        res = self.conv(x)
        if self.batchnorm_needed:
            res = self.batch_norm(res)

        res = self.drop(res)
        res = torch.relu(res)
        res = self.pool(res)

        return res


class ConvNet(nn.Module):
    def __init__(self, n_chans=16, n_layers=4, kernel_size=3, batchnorm=False):
        super().__init__()

        self.conv1 = nn.Conv2d(3, n_chans, kernel_size)

        self.conv_hidden = nn.Sequential()

        for i in range(n_layers):
            self.conv_hidden.append(NetBlock(n_chans,
                                             n_chans,
                                             kernel_size=kernel_size,
                                             batchnorm_needed=batchnorm))

        self.fc_hidden = nn.LazyLinear(128)
        self.drop = nn.Dropout1d(p=0.5)
        self.fc_out = nn.Linear(128, len(constants.TARGET_CATEGORIES))

    def forward(self, x):
        res = self.conv1(x)
        res = torch.relu(res)
        res = self.conv_hidden(res)
        res = res.view(res.size()[0], -1)
        res = torch.relu(res)
        res = self.fc_hidden(res)
        res = self.drop(res)
        res = torch.relu(res)
        res = self.fc_out(res)
        res = F.log_softmax(res, dim=1)
        return res
