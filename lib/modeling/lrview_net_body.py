# Author zhangfandong
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg

import nn as mynn
import utils.net as net_utils

def get_lrv_net(input_channels, output_channels, net_name):
    """
    parse net by net_name
    :param input_channels: number of input channels
    :param output_channels: number of output channels
    :param net_name:
        'plain_abcd..': plain net with stacked conv layers with kernel_size a, b, c, d, ... e.g. plain_333
        'res_x': resnet with x blocks, e.g. res_2
        TODO:
        'dense_g_n': a densenet block with growth rate g and n layers. e.g. dense_256_4
        For all networks,  a 1x1 conv layer is appended to make sure the output channels.
    :return: network
    """
    items = net_name.split('_')
    net_type = items[0]
    if net_type == 'plain':
        assert len(items) == 2
        conv_kernels = [int(x) for x in items[1]]
        return LRVPlainNet(input_channels, output_channels, conv_kernels)
    if net_type == 'res':
        assert len(items) == 2
        return LRVResNet(input_channels, output_channels, int(items[1]))
    raise NotImplementedError


def lrv_bn(input_channels):
    return nn.BatchNorm2d(input_channels)

def lrv_gn(input_channels):
    return nn.GroupNorm(net_utils.get_group_gn(input_channels), input_channels,
                     eps=cfg.GROUP_NORM.EPSILON)

def conv_bn(input_channels, output_channels, conv_kernel, padding, stride=1, bias=False):
    conv = nn.Conv2d(input_channels, output_channels,
                     kernel_size=conv_kernel, stride=stride, padding=padding, bias=bias)
    if cfg.LR_VIEW.NORM_TYPE == 'bn':
        bn = lrv_bn(output_channels)
        return [conv, bn]
    elif cfg.LR_VIEW.NORM_TYPE == 'gn':
        gn = lrv_gn(output_channels)
        return [conv, gn]
    elif cfg.LR_VIEW.NORM_TYPE == 'none':
        return [conv]
    else:
        raise NotImplementedError
        

def conv_bn_relu(input_channels, output_channels, conv_kernel, padding, stride=1, bias=False):
    layers = conv_bn(input_channels, output_channels, conv_kernel, padding, stride, bias)
    return layers + [nn.ReLU(True)]

def conv_sigmoid(input_channels, output_channels, conv_kernel, padding, stride=1, bias=False):
    conv = nn.Conv2d(input_channels, output_channels,
                     kernel_size=conv_kernel, stride=stride, padding=padding, bias=bias)
    return [conv, nn.Sigmoid()]


class _LRVBaseNet(nn.Module):
    def __init__(self):
        super(_LRVBaseNet, self).__init__()

    def detectron_weight_mapping(self):
        return {}, []


class LRVPlainNet(_LRVBaseNet):
    def __init__(self, input_channels, output_channels, conv_kernels):
        super(LRVPlainNet, self).__init__()
        layers = []
        for k in conv_kernels:
            layers += conv_bn_relu(input_channels, input_channels, k, k // 2)
        if cfg.LR_VIEW.FUSION_POLICY == 'gated_attention':
            layers += conv_sigmoid(input_channels, output_channels, 1, 0)
        else:
            layers += conv_bn_relu(input_channels, output_channels, 1, 0)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _ResBlock(nn.Module):
    def __init__(self, input_channels, bottleneck_channels):
        super(_ResBlock, self).__init__()
        self.conv1 = nn.Sequential(*conv_bn_relu(input_channels, bottleneck_channels, 1, 0))
        self.conv2 = nn.Sequential(*conv_bn_relu(bottleneck_channels, bottleneck_channels, 3, 1))
        self.conv3 = nn.Conv2d(bottleneck_channels, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        if cfg.LR_VIEW.NORM_TYPE == 'bn':
            self.bn3 = lrv_bn(input_channels)
        elif cfg.LR_VIEW.NORM_TYPE == 'gn':
            self.bn3 = lrv_gn(input_channels)
        self.relu3 = nn.ReLU(True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if cfg.LR_VIEW.NORM_TYPE in ['bn', 'gn']:
            out = self.relu3(self.bn3(x + out))
        else:
            out = self.relu3(x + out)
        return out


class LRVResNet(_LRVBaseNet):
    def __init__(self, input_channels, output_channels, num_blocks):
        super(LRVResNet, self).__init__()
        bottleneck_channels = input_channels / 4
        blocks = [_ResBlock(input_channels, bottleneck_channels) for i in range(num_blocks)]
        blocks += conv_bn_relu(input_channels, output_channels, 1, 0)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


def get_lrvasymaha_net(input_channels, output_channels, net_name):
    """
    parse net by net_name
    :param input_channels: number of input channels
    :param output_channels: number of output channels
    :param net_name:
        'plain_abcd.._kernelsize': plain net with stacked conv layers with kernel_size a, b, c, d, ... e.g. plain_333
        'res_x': resnet with x blocks, e.g. res_2
        TODO:
        'dense_g_n': a densenet block with growth rate g and n layers. e.g. dense_256_4
        For all networks,  a 1x1 conv layer is appended to make sure the output channels.
    :return: network
    """
    items = net_name.split('_')
    net_type = items[0]
    conf = {}
    conf['input_channels'] = input_channels
    conf['output_channels'] = output_channels
    conf['type'] = net_type
    conf['conv_kernels'] = [int(x) for x in items[1]]
    if len(items) == 3:
        dist_kernels = int(items[2])
    else:
        dist_kernels = 1
    return LRVAsymMahaFusionNet(input_channels, output_channels, conf, dist_kernels)


def get_fusion_net_backbone(conf):
    if conf['type'] == 'plain':
        if cfg.LR_VIEW.FUSION_POLICY == 'gated_attention':
            input_channel_num = conf['input_channels']
        else:
            input_channel_num = conf['input_channels'] * 2
        return LRVPlainNet(input_channel_num, conf['output_channels'], conf['conv_kernels'])
    return LRVResNet(conf['input_channels'], conf['output_channels'], conf['num_blocks'])


class LRVAsymMahaFusionNet(_LRVBaseNet):
    def __init__(self, input_channels, output_channels, conf, dist_kernels=1):
        """
        :param input_channels:
        :param output_channels:
        :param conf: conf.keys=[type, input_channels, output_channels, conv_kernels, num_blocks]
                    conf['type']={plain, res}
        """
        super(LRVAsymMahaFusionNet, self).__init__()
        if cfg.LR_VIEW.DIST_POLICY == 'concat':
            self.maha_conv = nn.Conv2d(2 * input_channels, input_channels, kernel_size=dist_kernels, stride=1, padding=int((dist_kernels-1) / 2.0), bias=False)
        else:
            self.maha_conv = nn.Conv2d(input_channels, input_channels, kernel_size=dist_kernels, stride=1, padding=int((dist_kernels-1) / 2.0), bias=False)
        self.model = get_fusion_net_backbone(conf)

    def forward(self, x, y):
        if cfg.LR_VIEW.DIST_POLICY == 'multiply':
            if cfg.LR_VIEW.FUSION_POLICY == 'gated_attention':
                return self.model(self.maha_conv(x * y)) * x
            else:
                return self.model(torch.cat([self.maha_conv(x * y), x], 1))
        elif cfg.LR_VIEW.DIST_POLICY == 'diff':
            if cfg.LR_VIEW.FUSION_POLICY == 'gated_attention':
                return self.model(self.maha_conv(x - y)) * x
            else:
                return self.model(torch.cat([self.maha_conv(x - y), x], 1))
        elif cfg.LR_VIEW.DIST_POLICY == 'concat':
            if cfg.LR_VIEW.FUSION_POLICY == 'gated_attention':
                return self.model(self.maha_conv(torch.cat([x, y], 1))) * x
            else:
                return self.model(torch.cat([self.maha_conv(torch.cat([x, y], 1)), x], 1))

