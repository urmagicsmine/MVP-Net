# Author zhangfandong
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg

import nn as mynn
import utils.net as net_utils

def get_gif_net(input_channels):
    """
    An implementation of the Gated Information Fusion layer.
    """
    return GIFPlainNet(input_channels)


class _GIFBaseNet(nn.Module):
    def __init__(self):
        super(_GIFBaseNet, self).__init__()

    def detectron_weight_mapping(self):
        return {}, []


class GIFPlainNet(_GIFBaseNet):
    """An implementation of the Gated Information Fusion layer.
       Introduced in Robust Deep Multi-modal Learning Based on Gated Information Fusion Network
       Params: input_channels: channles of the left or right view feature map.
    """
    def __init__(self, input_channels):
        super(GIFPlainNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels * 2, 1, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(input_channels * 2, 1, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv_fuse = nn.Conv2d(input_channels * 2, input_channels, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.sig1 = nn.Sigmoid()
        self.sig2 = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, left_x, right_x):
        cat_x = torch.cat([left_x, right_x], 1)
        weighted_left_x = self.sig1(self.conv1(cat_x)) * left_x
        weighted_right_x = self.sig2(self.conv2(cat_x)) * right_x
        concated_x = torch.cat([weighted_left_x, weighted_right_x], 1)

        return self.relu(self.conv_fuse(concated_x))

