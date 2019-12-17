import numpy as np
import cv2
import os

import utils.blob as blob_utils
import utils.opencv_transforms as cv_transforms
import roi_data.rpn

from core.config import cfg
from utils.ImageIO import load_16bit_png, load_multislice_16bit_png
from utils.ImageIO import get_dicom_image_blob, get_double_image_blob, get_a_img

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    if cfg.MODEL.LR_VIEW_ON or cfg.MODEL.GIF_ON or cfg.MODEL.LRASY_MAHA_ON:
        im_blob, im_scales = get_double_image_blob(roidb)
    else:
        im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid

def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        if cfg.DATA_SOURCE == 'coco':
            if cfg.LESION.LESION_ENABLED:
                im = load_multislice_16bit_png(roidb[i])
            else:
                im = cv2.imread(roidb[i]['image'])
        elif cfg.DATA_SOURCE == 'mammo':
            im = get_a_img(roidb[i])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])

        #print(roidb[i]['flipped'],cfg.TRAIN.AUGMENTATION,cfg.TRAIN.ONLINE_RANDOM_CROPPING)
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        if roidb[i]['z_flipped']:
            im = im[:, :, ::-1]
        # Generate the cropped image
        if cfg.TRAIN.ONLINE_RANDOM_CROPPING:
            x1, y1, x2, y2 = roidb[i]['cropped_bbox']
            im = im[y1:y2+1, x1:x2+1, :]

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]

        if cfg.TRAIN.AUGMENTATION:
            transform_cv = cv_transforms.Compose([
                cv_transforms.ColorJitter(brightness=0.5,
                contrast=0.25, gamma=0.5)])
        else:
            transform_cv = None
        #TODO: choose suitable pixel_means for DEEPLESION data, see also utils/blob.py:get_image_blob
        if cfg.LESION.LESION_ENABLED:
            if cfg.LESION.USE_3DCE or cfg.LESION.MULTI_MODALITY:
                pixel_means = np.tile(np.array([100]), cfg.LESION.NUM_IMAGES_3DCE * 3)
            else:
                pixel_means = np.tile(np.array([100]), cfg.LESION.SLICE_NUM)
        else:
            pixel_means = cfg.PIXEL_MEANS
        im, im_scale = blob_utils.prep_im_for_blob(
                im, pixel_means, [target_size], cfg.TRAIN.MAX_SIZE, transform_cv)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])
    # Create a blob to hold the input images [n, c, h, w] or [n, c, d, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

