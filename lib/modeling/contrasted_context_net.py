# Author zhangshu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg

import nn as mynn
import utils.net as net_utils

def get_cc_net(input_channels, output_channels):
    """
    Get an intance of the contrasted_context_net
    """
    return CCNet(input_channels, output_channels)


class _CCBaseNet(nn.Module):
    def __init__(self):
        super(_CCBaseNet, self).__init__()

    def detectron_weight_mapping(self):
        return {}, []


class CCNet(_CCBaseNet):
    """An implementation of the Contrasted Contest layer from CVPR2018 paper.
       Introduced in << Context contrasted feature and gated multi-scale aggregation ...>>
       Params: input_channels, output_channels
    """
    def __init__(self, input_channels, output_channels):
        super(CCNet, self).__init__()
        self.conv_local = nn.Conv2d(input_channels, output_channels, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv_context = nn.Conv2d(input_channels,
                                      output_channels, kernel_size=(3,3), stride=1, 
                                      padding=cfg.CONTRASTED_CONTEXT.DILATION_SIZE, 
                                      dilation=cfg.CONTRASTED_CONTEXT.DILATION_SIZE, 
                                      bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        local_info = self.conv_local(x)
        context_info = self.conv_context(x)

        return self.relu(local_info - context_info)

