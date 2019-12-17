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
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import modeling.lrview_net_body as lrview_net_body
import modeling.gated_fusion_body as gif_net_body
import utils.blob as blob_utils
import utils.net as net_utils
import utils.boxes as boxes_utils
import utils.resnet_weights_helper as resnet_utils
from modeling.model_builder import Generalized_RCNN, get_func, compare_state_dict, check_inference

logger = logging.getLogger(__name__)


class Context_RCNN(Generalized_RCNN):
    """RCNN models with multi-context roi-pooling. """
    def __init__(self):
        super().__init__()

        # Position Branch
        if cfg.LESION.USE_POSITION:
            self.Position_Head = position_Xconv1fc_gn_head(2048)
            self.Position_Cls_Outs = position_cls_outputs(self.Position_Head.dim_out)
            self.Position_Reg_Outs = position_reg_outputs(self.Position_Head.dim_out)

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            # Add box head for context roi-pooling
            self.Context_Box_Head = get_func(cfg.FAST_RCNN.ROI_CONTEXT_BOX_HEAD)(
                self.RPN.dim_out, self.context_roi_feature_transform, self.Conv_Body.spatial_scale)
            # For context-enchanced roi-pooling, we need to use the fast_rcnn_disentangle_outputs instead.
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_disentangle_outputs(
                self.Box_Head.dim_out)

        self.im_info = None
        self._init_modules()


    def forward(self, data, im_info, roidb=None, **rpn_kwargs):
        self.im_info = im_info
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, **rpn_kwargs):
        im_data = data

        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()
        return_dict = {}  # A dict to collect return variables
        blob_conv,res5_feat = self.Conv_Body(im_data)

        if cfg.MODEL.LR_VIEW_ON or cfg.MODEL.GIF_ON or cfg.MODEL.LRASY_MAHA_ON:
            blob_conv = self._get_lrview_blob_conv(blob_conv)

        rpn_ret = self.RPN(blob_conv, im_info, roidb)

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
                context_box_feat, context_res5_feat = self.Context_Box_Head(blob_conv, rpn_ret)
            else:
                box_feat = self.Box_Head(blob_conv, rpn_ret)
                context_box_feat  = self.Context_Box_Head(blob_conv, rpn_ret)

            if cfg.CONTEXT_ROI_POOLING.MERGE_TYPE == 'concat':
                joint_box_feat = torch.cat([context_box_feat, box_feat], 1)
            elif cfg.CONTEXT_ROI_POOLING.MERGE_TYPE == 'sum':
                joint_box_feat = context_box_feat + box_feat

            if cfg.CONTEXT_ROI_POOLING.SHARE_FEATURE:
                cls_score, bbox_pred = self.Box_Outs(joint_box_feat, joint_box_feat)
            else:
                cls_score, bbox_pred = self.Box_Outs(joint_box_feat, box_feat)
            # print cls_score.shape
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred

            if cfg.LESION.USE_POSITION:
                position_feat = self.Position_Head(res5_feat)
                pos_cls_pred = self.Position_Cls_Outs(position_feat)
                pos_reg_pred = self.Position_Reg_Outs(position_feat)
                return_dict['pos_cls_pred'] = pos_cls_pred
                return_dict['pos_reg_pred'] = pos_reg_pred
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

            if cfg.MODEL.MASK_ON:
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                               roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                else:
                    mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                mask_pred = self.Mask_Outs(mask_feat)
                # return_dict['mask_pred'] = mask_pred
                # mask loss
                loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                return_dict['losses']['loss_mask'] = loss_mask

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


    def context_roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)
        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    context_rois = self.get_context_rois(rpn_ret[bl_rois], self.im_info, zoom_ratio=cfg.CONTEXT_ROI_POOLING.ZOOM_RATIO)
                    rois = Variable(torch.from_numpy(context_rois)).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            context_rois = self.get_context_rois(rpn_ret[blob_rois], self.im_info, zoom_ratio=cfg.CONTEXT_ROI_POOLING.ZOOM_RATIO)
            rois = Variable(torch.from_numpy(context_rois)).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    def get_context_rois(self, rois, im_info, zoom_ratio=1.1):
        """Return the rois with more context. 

          - 'im_info': 2D tensor of shape (N, 3) where the three columns encode
            the input image's [height, width, scale]. Height and width are
            for the input to the network, not the original image; scale is the
            scale factor used to scale the original image to the network input
            size.

          - 'rpn_rois': 2D tensor of shape (R, 5), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, x2, y2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
        """
        if zoom_ratio > 0:
            roi_boxes = rois[:, 1:]
            batch_indice = rois[:, 0]
            roi_boxes = boxes_utils.expand_boxes(roi_boxes, zoom_ratio)
            roi_boxes = boxes_utils.clip_boxes_to_image(roi_boxes, im_info[0], im_info[1])
            rois = np.concatenate((batch_indice, roi_boxes), axis=1)
        else:
            # Add global context, i.e. roi-pooling on the whole feature map.
            rois[:, 1] = 0
            rois[:, 2] = 0
            rois[:, 3] = 800 - 1
            rois[:, 4] = 800 - 1
            #rois[:, 3] = im_info[0][1] - 1
            #rois[:, 4] = im_info[0][0] - 1
        return rois
