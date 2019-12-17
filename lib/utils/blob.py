# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import cv2
from core.config import cfg


def get_image_blob(image, target_scale, target_max_size):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    #TODO: choose suitable pixel_means for DEEPLESION data, see also roi_data/minibatch.py:_get_image_blob
    if cfg.LESION.LESION_ENABLED:
        if cfg.LESION.USE_3DCE or cfg.LESION.MULTI_MODALITY:
            pixel_means = np.tile(np.array([100]), cfg.LESION.NUM_IMAGES_3DCE * 3)
        else:
            pixel_means = np.tile(np.array([100]), cfg.LESION.SLICE_NUM)
    else:
        pixel_means = cfg.PIXEL_MEANS
    if isinstance(image, list):
        im = image[0]
        other_im = image[1]
        processed_im = []
        im, im_scale = prep_im_for_blob(
            im, pixel_means, [target_scale], target_max_size, None)
        other_im, other_im_scale = prep_im_for_blob(
            other_im, pixel_means, [target_scale], target_max_size, None)
        processed_im.append(im[0])
        processed_im.append(other_im[0])

    else:
        processed_im, im_scale = prep_im_for_blob(
            image, pixel_means, [target_scale], target_max_size, None
        )
    # Note: processed_im might have different shape with blob. blob might be larger than
    # processed_im, or max_size
    blob = im_list_to_blob(processed_im)
    # NOTE: this height and width may be larger than actual scaled input image
    # due to the FPN.COARSEST_STRIDE related padding in im_list_to_blob. We are
    # maintaining this behavior for now to make existing results exactly
    # reproducible (in practice using the true input image height and width
    # yields nearly the same results, but they are sometimes slightly different
    # because predictions near the edge of the image will be pruned more
    # aggressively).
    # N,C,H,W for 2D input; N,C,D,H,W for 3D input.
    if cfg.LESION.USE_3D_INPUT:
        height, width = blob.shape[3], blob.shape[4]
    else:
        height, width = blob.shape[2], blob.shape[3]
    im_info = np.hstack((height, width, im_scale))[np.newaxis, :]
    return blob, im_scale, im_info.astype(np.float32)


def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent: i.e.
      - BGR channel order
      - pixel means subtracted
      - resized to the desired input size
      - float32 numpy ndarray format
      - H,W,C for 2D input , H,W,D for 3D input
    Output: 4D N,C,H,W for 2D input (5D N,C,D,H,W for 3D input).
    """
    if not isinstance(ims, list):
        ims = [ims]
    num_images = len(ims)

    if cfg.LESION.USE_3D_INPUT:
        # transform 3D Lesion data(H,W,D) to (N,C,D,H,W).

        max_shape = get_3d_max_shape([im.shape for im in ims])
        # depth axis is not padded.
        blob = np.zeros(
            (num_images, 1,max_shape[0], max_shape[1], ims[0].shape[2]), dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0, 0:im.shape[0], 0:im.shape[1], :im.shape[2]] = im
        channel_swap = (0, 1, 4, 2, 3)
        # Axis order will become: (n, c, d, h, w), eg. (1,1,9,800,800) for 9 slices
        blob = blob.transpose(channel_swap)
    else:
        max_shape = get_max_shape([im.shape[:2] for im in ims])
        if cfg.LESION.LESION_ENABLED:
            if cfg.LESION.USE_3DCE or cfg.LESION.MULTI_MODALITY:
                blob = np.zeros((num_images, max_shape[0], max_shape[1], cfg.LESION.NUM_IMAGES_3DCE * 3), dtype=np.float32)
            else:
                blob = np.zeros((num_images, max_shape[0], max_shape[1], cfg.LESION.SLICE_NUM), dtype=np.float32)
        else:
            blob = np.zeros(
                (num_images, max_shape[0], max_shape[1], 3), dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
    return blob


def get_3d_max_shape(im_shapes):
    """
    Calculate max spatial size for batching given a list of image shapes.
    Note That this function is called twice during dealing one batch,
    first in blob.get_minibatch(),H,W,D order, then in loader.collate_minibatch(),D,H,W order.
    Depth pad should be ignored.
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 3
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
        max_shape[2] = int(np.ceil(max_shape[2] / stride) * stride)
    return max_shape

def get_max_shape(im_shapes):
    """Calculate max spatial size (h, w) for batching given a list of image shapes
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
    return max_shape


def prep_im_for_blob(im, pixel_means, target_sizes, max_size, transform_cv=None):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    if transform_cv != None:
        im = transform_cv(im)
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    ims = []
    im_scales = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
        ims.append(im_resized)
        im_scales.append(im_scale)
    return ims, im_scales


def get_im_blob_sizes(im_shape, target_sizes, max_size):
    """Calculate im blob size for multiple target_sizes given original im shape
    """
    im_size_min = np.min(im_shape)
    im_size_max = np.max(im_shape)
    im_sizes = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_sizes.append(np.round(im_shape * im_scale))
    return np.array(im_sizes)


def get_target_scale(im_size_min, im_size_max, target_size, max_size):
    """Calculate target resize scale
    """
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())
