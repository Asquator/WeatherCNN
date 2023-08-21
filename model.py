import torch
from torch import nn
from torch.nn import functional as F

import constants


class NetBlock(nn.Module):
    def __init__(self, in_chans, out_chans, batchnorm_needed, kernel_size):
        super().__init__()
        self.batchnorm_needed = batchnorm_needed
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        if batchnorm_needed:
            self.batch_norm = nn.BatchNorm2d(num_features=out_chans)

        self._in_chans = in_chans
        self._out_chans = out_chans

    def forward(self, x):
        res = self.conv(x)
        if self.batchnorm_needed:
            res = self.batch_norm(res)

        res = torch.relu(res)
        res = self.pool(res)

        return res


class ConvNet(nn.Module):
    def __init__(self, n_channels1=20, n_layers=5, kernel_size=3, batchnorm=False):
        super().__init__()

        self.conv1 = nn.Conv2d(3, n_channels1, kernel_size)

        self.conv_hidden = nn.Sequential()

        n_chans = n_channels1
        next_n_chans = 3 * int(n_chans / 4) + 1

        for i in range(n_layers):
            self.conv_hidden.append(NetBlock(n_chans,
                                             next_n_chans,
                                             kernel_size=kernel_size,
                                             batchnorm_needed=batchnorm))
            n_chans = next_n_chans
            next_n_chans = 3 * int(n_chans / 4) + 1

        self.fc_hidden = nn.LazyLinear(128)
        self.fc_out = nn.Linear(128, len(constants.TARGET_CATEGORIES))

    def forward(self, x):
        res = self.conv1(x)
        res = torch.relu(res)
        res = self.conv_hidden(res)
        res = res.view(res.size()[0], -1)
        res = torch.relu(res)
        res = self.fc_hidden(res)
        res = torch.relu(res)
        res = self.fc_out(res)
        res = F.log_softmax(res, dim=1)
        return res
