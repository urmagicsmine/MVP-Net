# Added by lizihao
# 20190104
# Add Z-axis supervision For DeepLesion 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
from model.utils.loss import focal_loss_all

USE_CLS = 1
USE_REG = 1
# for rcnn head
class position_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.position_cls = nn.Linear(dim_in, 3)
        self.position_reg = nn.Linear(dim_in, 1)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.position_cls.weight, std=0.001)
        init.constant_(self.position_cls.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'position_cls.weight': 'position_cls_w',
            'position_cls.bias': 'position_cls_b',
            'position_reg.weight': 'position_reg_w',
            'position_reg.bias': 'position_reg_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        position_cls = self.position_cls(x)
        position_reg = self.position_reg(x)
        return position_cls, position_reg
class position_reg_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.position_reg = nn.Linear(dim_in, 1)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.position_reg.weight, std=0.001)
        init.constant_(self.position_reg.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'position_reg.weight': 'position_reg_w',
            'position_reg.bias': 'position_reg_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        #input : res5_feat
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        #print('INPUT X ',x)
        position_reg = self.position_reg(x)
        return position_reg

class position_cls_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.position_cls = nn.Linear(dim_in, 3)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.position_cls.weight, std=0.001)
        init.constant_(self.position_cls.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'position_cls.weight': 'position_cls_w',
            'position_cls.bias': 'position_cls_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):
        #input : res5_feat
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        position_cls = self.position_cls(x)
        if not self.training:
            position_cls = F.softmax(position_cls, dim=1)
        return position_cls

def position_rcnn_losses(position_cls_pred, position_reg_pred, roidb, \
        position_inside_weights=np.array((1.0),dtype=np.float32), \
        position_outside_weights=np.array((0.01),dtype=np.float32)):
    bs = cfg.TRAIN.RPN_BATCH_SIZE_PER_IM
    device_id = position_cls_pred.get_device()
    position_reg_targets = np.zeros((len(roidb)*bs))
    position_cls_targets = np.zeros((len(roidb)*bs))
    position_cls_bins = np.array((0.58,0.72,1))
    for idx,entry in enumerate(roidb):
        position_reg_targets[idx*bs:(idx+1)*bs] = entry['z_position']
        position_cls_targets[idx*bs:(idx+1)*bs] = np.digitize(entry['z_position'], position_cls_bins)
    #print('pos', position_pred.cpu(), position_reg_targets)
    #print(position_cls_targets)
    if USE_CLS:
        position_cls_targets= Variable(torch.from_numpy(position_cls_targets.astype('int64'))).cuda(device_id)
        cls_loss = F.cross_entropy(position_cls_pred, position_cls_targets)
        cls_preds = position_cls_pred.max(dim=1)[1].type_as(position_cls_targets)
        accuracy_cls = cls_preds.eq(position_cls_targets).float().mean(dim=0)
        print('rcnn position cls acc: ',accuracy_cls, accuracy_cls.cpu())
    if USE_REG:
        position_reg_targets = Variable(torch.from_numpy(position_reg_targets.astype(np.float32))).cuda(device_id)
        position_inside_weights = Variable(torch.from_numpy(position_inside_weights)).cuda(device_id)
        position_outside_weights = Variable(torch.from_numpy(position_outside_weights)).cuda(device_id)
        reg_loss = net_utils.smooth_l1_loss(position_reg_pred, position_reg_targets, position_inside_weights, position_outside_weights)

    return cls_loss,reg_loss,accuracy_cls

def position_losses(position_cls_pred, position_reg_pred, roidb, \
        position_inside_weights=np.array((1.0),dtype=np.float32), \
        position_outside_weights=np.array((1.0),dtype=np.float32)):

    device_id = position_cls_pred.get_device()
    position_reg_targets = np.zeros((len(roidb)))
    position_cls_targets = np.zeros((len(roidb)))
    position_cls_bins = np.array((0.58,0.72,1))
    for idx,entry in enumerate(roidb):
        position_reg_targets[idx] = entry['z_position']
        position_cls_targets[idx] = np.digitize(entry['z_position'], position_cls_bins)
    #print('pos', position_pred.cpu(), position_reg_targets)
    # note: only support multi-modal now.

    # expand *3
    if cfg.LESION.MULTI_MODALITY:
        position_cls_targets = np.tile(position_cls_targets,(3))
        position_reg_targets = np.tile(position_reg_targets,(3))

    if USE_CLS:
        position_cls_targets= Variable(torch.from_numpy(position_cls_targets.astype('int64'))).cuda(device_id)
        cls_loss = 0.10*F.cross_entropy(position_cls_pred, position_cls_targets)
        cls_preds = position_cls_pred.max(dim=1)[1].type_as(position_cls_targets)
        accuracy_cls = cls_preds.eq(position_cls_targets).float().mean(dim=0)
    if USE_REG:
        position_reg_targets = Variable(torch.from_numpy(position_reg_targets.astype(np.float32))).cuda(device_id)
        position_inside_weights = Variable(torch.from_numpy(position_inside_weights)).cuda(device_id)
        position_outside_weights = Variable(torch.from_numpy(position_outside_weights)).cuda(device_id)
        reg_loss = net_utils.smooth_l1_loss(position_reg_pred, position_reg_targets, position_inside_weights, position_outside_weights)

    return cls_loss,reg_loss,accuracy_cls


# ---------------------------------------------------------------------------- #
# Position heads
# ---------------------------------------------------------------------------- #


class position_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, hidden_dim=256, num_convs=4):
        super().__init__()
        self.dim_in = dim_in
        self.num_convs = num_convs
        self.hidden_dim = hidden_dim
        #self.num_convs = 4      # 4 in fast rcnn heads
        #self.hidden_dim = 256   # FAST_RCNN.CONV_HEAD_DIM = 256
        self.position_cls = 3
        self.position_threshold = []
        module_list = []
        if 1:
            module_list.extend([
                    nn.Conv2d(dim_in, dim_in, 3, 2, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(dim_in), dim_in,
                                 eps=cfg.GROUP_NORM.EPSILON),
                    nn.ReLU(inplace=True)
                ])
            module_list.extend([
                    nn.Conv2d(dim_in, self.hidden_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(self.hidden_dim), self.hidden_dim,
                                 eps=cfg.GROUP_NORM.EPSILON),
                    nn.ReLU(inplace=True)
                ])
            module_list.extend([
                    nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(self.hidden_dim), self.hidden_dim,
                                 eps=cfg.GROUP_NORM.EPSILON),
                    nn.ReLU(inplace=True)
                ])
        else:
            module_list.extend([
                    nn.Conv2d(dim_in, self.hidden_dim, 3, 2, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(self.hidden_dim), self.hidden_dim,
                                 eps=cfg.GROUP_NORM.EPSILON),
                    nn.ReLU(inplace=True)
                ])
            for i in range(self.num_convs-1): # 4 in fast rcnn heads
                module_list.extend([
                    nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(net_utils.get_group_gn(self.hidden_dim), self.hidden_dim,
                                 eps=cfg.GROUP_NORM.EPSILON),
                    nn.ReLU(inplace=True)
                ])
        self.convs = nn.Sequential(*module_list)
        #self.dim_out = cfg.FAST_RCNN.MLP_HEAD_DIM  #1024
        self.dim_out = self.hidden_dim
        #self.fc1 = nn.Linear(self.hidden_dim * 49, self.dim_out)
        #self.fc1 = nn.Linear(self.hidden_dim, self.dim_out)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(self.num_convs):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b'
        })
        return mapping,[]

    def forward(self, x):
        batch_size = x.size(0)
        x = self.convs(x)
        #x = self.avgpool(x)
        #x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.avgpool(x).view(batch_size, -1), inplace=True)
        return x

