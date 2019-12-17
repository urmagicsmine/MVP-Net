from functools import wraps
import importlib
import logging
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
from modeling.position_heads import position_Xconv1fc_gn_head, position_cls_outputs, position_reg_outputs, position_losses, position_rcnn_losses, position_outputs
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.gated_fusion_body as gif_net_body
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
from modeling.model_builder import Generalized_RCNN, get_func
from modeling.cbam import *
logger = logging.getLogger(__name__)


class MULTI_MODALITY_RCNN(Generalized_RCNN):

    def __init__(self):
        super().__init__()
        if cfg.LESION.USE_POSITION:
            #self.Position_Head = position_Xconv1fc_gn_head(2048,1024,3)
            self.Position_Head = position_Xconv1fc_gn_head(256,1024,3)
            self.Position_Cls_Outs = position_cls_outputs(self.Position_Head.dim_out)
            self.Position_Reg_Outs = position_reg_outputs(self.Position_Head.dim_out)
        if cfg.LESION.POS_CONCAT_RCNN:
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                    self.Box_Head.dim_out + self.Position_Head.dim_out)

        self.cbam = CBAM(self.Conv_Body.dim_out*cfg.LESION.NUM_IMAGES_3DCE, 16, no_spatial=True)


    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        M = cfg.LESION.NUM_IMAGES_3DCE
        # data.shape: [n,M*3,h,w]
        n,c,h,w = data.shape
        im_data = data.view(n*M,3,h,w)
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()
        return_dict = {}  # A dict to collect return variables
        if cfg.LESION.USE_POSITION:
            blob_conv, res5_feat = self.Conv_Body(im_data)
        else:
            blob_conv = self.Conv_Body(im_data)

        # blob.shape = [n, c,h,w] for 2d
        # blob.shape = [nM,c,h,w] for 3DCE or multi-modal

        blob_conv_for_RPN = []
        blob_conv_for_RCNN = []
        # Used for MVP-Net, concat all slices before RPN.
        if cfg.LESION.CONCAT_BEFORE_RPN:
            for blob in blob_conv:
                _,c,h,w = blob.shape
                blob = blob.view(n, M*c, h, w)
                blob_cbam = self.cbam(blob)
                blob_conv_for_RPN.append(blob_cbam)
            blob_conv_for_RCNN = blob_conv_for_RPN
        # Used for Standard 3DCE, feed middle slice into RPN.
        else:
            for blob in blob_conv:
                _,c,h,w = blob.shape
                blob = blob.view(n, M*c, h, w)
                blob_conv_for_RPN.append(blob[:, (M//2)*c: (M//2+1)*c, :, :])
                blob_conv_for_RCNN.append(blob)

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

            if cfg.LESION.USE_POSITION:
                if cfg.LESION.NUM_IMAGES_3DCE == 3:
                    position_feat = blob_conv_for_RPN[0]
                    n,c,h,w =position_feat.shape
                    position_feat = position_feat.view(3,256,h,w)
                    #print(position_feat.shape)
                elif cfg.LESION.NUM_IMAGES_3DCE == 9:
                    position_feat = blob_conv_for_RPN[0][:, 3*256:6*256,:,:]
                    n,c,h,w =position_feat.shape
                    position_feat = position_feat.view(3,256,h,w)
                    #print(position_feat.shape)
                position_feat = self.Position_Head(position_feat)
                pos_cls_pred = self.Position_Cls_Outs(position_feat)
                pos_reg_pred = self.Position_Reg_Outs(position_feat)
                return_dict['pos_cls_pred'] = pos_cls_pred
                return_dict['pos_reg_pred'] = pos_reg_pred
            if cfg.LESION.POS_CONCAT_RCNN:
                try:
                    if self.training:
                        # expand position_feat as box_feat for concatenation.
                        # box_feat: (n*num_nms, 1024)
                        # position_feat: (n, 1024) expand--> position_feat_:(num_nms, n ,1024) 
                        position_feat_ = position_feat.expand(cfg.TRAIN.RPN_BATCH_SIZE_PER_IM,-1,-1)
                        # position_feat_: (num_nms ,n , 1024)  transpose--> (n, num_nms, 1024)  view-->(n*num_nms, 1024)
                        position_feat_concat = torch.transpose(position_feat_, 1, 0).contiguous().view(-1, 1024)
                    else:
                        # box_feat: (1, 1024)
                        # position_feat: (1, 1024), position_feat_:(1000 ,1024) 
                        position_feat_ = position_feat.expand(cfg.TEST.RPN_PRE_NMS_TOP_N,-1)
                        # position_feat_: (1, 1024)  transpose--> position_feat_concat (n, num_nms, 1024)
                        position_feat_concat = torch.transpose(position_feat_, 1, 0).contiguous().view_as(box_feat)
                except:
                    import pdb
                    pdb.set_trace()
                    print('position_feat', position_feat.shape)
                    print('box_feat', box_feat.shape)
                    print('position_feat_', position_feat_.shape)
                    print('position_feat_concat', position_feat_concat.shape)
                else:
                    box_feat = torch.cat([box_feat, position_feat_concat], 1)
                    cls_score, bbox_pred = self.Box_Outs(box_feat)
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

            if cfg.MODEL.FASTER_RCNN: # bbox loss
                loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                    rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
                return_dict['losses']['loss_cls'] = loss_cls
                return_dict['losses']['loss_bbox'] = loss_bbox
                return_dict['metrics']['accuracy_cls'] = accuracy_cls

            if cfg.LESION.USE_POSITION:
                pos_cls_loss,pos_reg_loss,accuracy_position = position_losses(pos_cls_pred,pos_reg_pred,roidb)
                return_dict['losses']['pos_cls_loss'] = pos_cls_loss
                return_dict['losses']['pos_reg_loss'] = pos_reg_loss
                return_dict['metrics']['accuracy_position'] = accuracy_position


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

class Fusion(nn.Module):
    def __init__(self, input_channels):
        super(Fusion, self).__init__()
        self.dim_in = input_channels
        #self.modal1_1 == nn.Conv2d(self.dim_in, 1, 1, 0)
        #self.modal2_1 == nn.Conv2d(self.dim_in, 1, 1, 0)
        #self.modal3_1 == nn.Conv2d(self.dim_in, 1, 1, 0)
        #self.modal1_2 == nn.Conv2d(self.dim_in, 3, 1, 1)
        #self.modal2_2 == nn.Conv2d(self.dim_in, 3, 1, 1)
        #self.modal3_2 == nn.Conv2d(self.dim_in, 3, 1, 1)
        self.modal1 = nn.Sequential(nn.Conv2d(self.dim_in, self.dim_in, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim_in, self.dim_in, 3,1,1),
                nn.ReLU(inplace=True))
        self.modal2 = nn.Sequential(nn.Conv2d(self.dim_in, self.dim_in, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim_in, self.dim_in, 3,1,1),
                nn.ReLU(inplace=True))
        self.modal3 = nn.Sequential(nn.Conv2d(self.dim_in, self.dim_in, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim_in, self.dim_in, 3,1,1),
                nn.ReLU(inplace=True))
        self.conv_concat = nn.Sequential(nn.Conv2d(self.dim_in * 3, self.dim_in, 1, 1, 0),
                nn.ReLU(inplace=True))

    def detectron_weight_mapping(self):
        return {}, []

    def forward(self, input_list):
        modal1 = self.modal1(input_list[0])
        modal2 = self.modal2(input_list[1])
        modal3 = self.modal3(input_list[2])
        x = torch.cat([modal1,modal2,modal3],1)
        x = self.conv_concat(x)
        #x = torch.sum(modal1,modal2,modal3)
        return x

