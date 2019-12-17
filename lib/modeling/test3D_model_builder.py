from functools import wraps
import importlib
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
from modeling.model_builder import Generalized_RCNN

logger = logging.getLogger(__name__)
#RPN_2_RCNN_3 = True

class TEST_RCNN(Generalized_RCNN):

    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        # data.shape: [n,c,cfg.LESION.NUM_SLICE,h,w]
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()
        return_dict = {}  # A dict to collect return variables
        # blob_conv[i].shape [n,c,d,h,w] for RPN, d = cfg.LESION.SLICE_NUM
        blob_conv_for_RPN,blob_conv_for_RCNN = self.Conv_Body(im_data)

        # blob.shape [n,c,h,w] for RPN
        # blob.shape [n,c,h,w] for RCNN
        #blob_conv_for_RPN = []
        #blob_conv_for_RCNN = []
        #if RPN_2_RCNN_3:
        #    for blob in blob_conv:
        #        n, c, d, h, w = blob.shape
        #        blob_ = blob[:, :, d//2, :, :]
        #        blob_conv_for_RPN.append(blob_)
        #    blob_conv_for_RCNN = blob_conv


        rpn_ret = self.RPN(blob_conv_for_RPN, im_info, roidb)

        # if self.training:
        #     # can be used to infer fg/bg ratio
        #     return_dict['rois_label'] = rpn_ret['labels_int32']

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv_for_RCNN = blob_conv_for_RCNN[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv_for_RCNN

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv_for_RCNN, rpn_ret)
            else:
                box_feat = self.Box_Head(blob_conv_for_RCNN, rpn_ret)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
            # print cls_score.shape
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred
        else:
            # TODO: complete the returns for RPN only situation
            pass

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            if cfg.MODEL.FASTER_RCNN:
            # bbox loss
                loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                    rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
                return_dict['losses']['loss_cls'] = loss_cls
                return_dict['losses']['loss_bbox'] = loss_bbox
                return_dict['metrics']['accuracy_cls'] = accuracy_cls


            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred
        return return_dict

