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

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml
import shutil

import torch

from core.config import cfg
# from core.rpn_generator import generate_rpn_on_dataset  #TODO: for rpn only case
# from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from datasets.mammo_json_dataset import MammoDataset
from datasets.lesion_json_dataset import LesionDataset
from datasets.mammo_dataset_evaluator import get_vis_gt_bboxes
from modeling.model_factory import GetRCNNModel
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
import utils.vis_gt as vis_gt_utils
from utils.io import save_object
from utils.timer import Timer
import os.path as osp
from utils.ImageIO import load_16bit_png,load_multislice_16bit_png
from sync_batchnorm.replicate import patch_replication_callback

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        raise NotImplementedError
        # child_func = generate_rpn_on_range
        # parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None
    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    if cfg.DATA_SOURCE == 'coco':
        dataset = JsonDataset(dataset_name)
    elif cfg.DATA_SOURCE == 'mammo':
        dataset = MammoDataset(dataset_name)
    #elif cfg.DATA_SOURCE == 'lesion':
    #    dataset = LesionDataset(dataset_name)

    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps = test_net(
            args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir
    )
    return results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


# def pad_image(img, pad=cfg.TRAIN.PADDING):
#     h, w = img.shape
#     new_img = np.zeros((h+2*pad, w+2*pad))
#     new_img[pad:-pad, pad:-pad] = img
#     return new_img
#
#
# def get_a_img(entry):
#     bbox_y1, bbox_x1, bbox_y2, bbox_x2 = entry['bbox']
#     im = cv2.imread(entry['image'], 0)
#     new_img = np.zeros((entry['height'], entry['width']))
#     pad = cfg.TRAIN.PADDING
#     y1,x1,y2,x2 = bbox_y1+pad, bbox_x1+pad,bbox_y2-pad,bbox_x2-pad
#     new_img[pad:(y2 - y1)+pad, pad:pad+(x2 - x1)] = im[y1:y2, x1:x2]
#     new_img = np.tile(new_img, (3, 1, 1))
#     new_img = np.transpose(new_img, (1, 2, 0))
#     return new_img.astype(np.uint8)
#
#
# def get_b_img(entry, im):
#     other_im = cv2.imread(entry['b_image'])
#     h, w, _ = im.shape
#     b_y1, b_x1, b_y2, b_x2 = entry['b_bbox']
#     pad = cfg.TRAIN.PADDING
#     other_im = cv2.resize(other_im[b_y1:b_y2, b_x1:b_x2, 0], dsize=(w-2*pad, h-2*pad), interpolation=cv2.INTER_CUBIC)
#     other_im = pad_image(other_im, pad=128)[:, ::-1]
#     other_im = np.tile(other_im, (3, 1, 1))
#     other_im = np.transpose(other_im, (1, 2, 0)).astype(np.uint8)
#     return other_im
#=======
def pad_image(img, pad=cfg.TRAIN.PADDING):
    h, w = img.shape
    new_img = np.zeros((h+2*pad, w+2*pad))
    if pad == 0:
        new_img = img.copy()
    else:
        new_img[pad:-pad, pad:-pad] = img
    return new_img


def get_a_img(entry):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = entry['bbox']
    im = cv2.imread(entry['image'], 0)
    if cfg.HIST_EQ and (not cfg.HIST_EQ_SYM):
        if cfg.A_HIST_EQ:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            im = clahe.apply(im)
        else:
            im = cv2.equalizeHist(im)

    pad = cfg.TRAIN.PADDING
    y1,x1,y2,x2 = bbox_y1+pad, bbox_x1+pad, bbox_y2-pad, bbox_x2-pad
    new_img = im[y1:y2, x1:x2]

    if cfg.HIST_EQ and cfg.HIST_EQ_SYM:
        if cfg.A_HIST_EQ:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            new_img = clahe.apply(new_img)
        else:
            new_img = cv2.equalizeHist(new_img)
    new_img = pad_image(new_img, pad=pad)
    new_img = np.tile(new_img, (3, 1, 1))
    new_img = np.transpose(new_img, (1, 2, 0))
    return new_img.astype(np.uint8), im.shape


def get_b_img(entry, im):
    other_im = cv2.imread(entry['b_image'])
    h, w, _ = im.shape
    b_y1, b_x1, b_y2, b_x2 = entry['b_bbox']
    pad = cfg.TRAIN.PADDING
    other_im = cv2.resize(other_im[b_y1:b_y2, b_x1:b_x2, 0], dsize=(w-2*pad, h-2*pad), interpolation=cv2.INTER_CUBIC)
    if cfg.HIST_EQ:
        if cfg.A_HIST_EQ:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            other_im = clahe.apply(other_im)
        else:
            other_im = cv2.equalizeHist(other_im)
    other_im = pad_image(other_im, pad=pad)[:, ::-1]
    other_im = np.tile(other_im, (3, 1, 1))
    other_im = np.transpose(other_im, (1, 2, 0)).astype(np.uint8)
    return other_im


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    model = initialize_model_from_cfg(args, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    if cfg.LESION.USE_POSITION:
        true_pos = 0

    if cfg.TEST.OFFLINE_MAP and (cfg.DATA_SOURCE == 'mammo' or cfg.DATA_SOURCE == 'lesion'):
        if osp.exists(os.path.join(output_dir, 'ground-truth')):
            shutil.rmtree(os.path.join(output_dir, 'ground-truth'))
        os.makedirs(os.path.join(output_dir, 'ground-truth'))
        if osp.exists(os.path.join(output_dir, 'predicted')):
            shutil.rmtree(os.path.join(output_dir, 'predicted'))
        os.makedirs(os.path.join(output_dir, 'predicted'))

    for i, entry in enumerate(roidb):
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None
        if cfg.DATA_SOURCE == 'coco':
            if cfg.LESION.LESION_ENABLED:
                im = load_multislice_16bit_png(roidb[i])
            else:
                im = cv2.imread(roidb[i]['image'])
        elif cfg.DATA_SOURCE == 'mammo':
            im, shape = get_a_img(roidb[i])

        if (cfg.MODEL.LR_VIEW_ON  or cfg.MODEL.GIF_ON or cfg.MODEL.LRASY_MAHA_ON) and cfg.DATA_SOURCE == 'mammo':
            other_im = get_b_img(entry, im)
            cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(model, [im, other_im], box_proposals, timers)
        else:
            if cfg.LESION.USE_POSITION:
                cls_boxes_i, cls_segms_i, cls_keyps_i, return_dict = im_detect_all(model, im, box_proposals, timers)
                bins = np.array((0.58,0.72,1))
                bin_pred = np.argmax(return_dict['pos_cls_pred'][0].data.cpu().numpy())
                bin_gt = np.digitize(entry['z_position'], bins)
                if bin_gt == bin_pred:
                    true_pos += 1
            else:
                cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(model, im, box_proposals, timers)

        if cfg.TEST.OFFLINE_MAP and cfg.DATA_SOURCE == 'mammo':
            gt_bboxes = get_vis_gt_bboxes(entry)
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            with open(os.path.join(output_dir, 'ground-truth', im_name + '.txt'), 'w') as w:
                for gt_bbox in gt_bboxes[0]:
                    w.write('mass %d %d %d %d\n'%(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]))
                for gt_bbox in gt_bboxes[1]:
                    w.write('mass %d %d %d %d difficult\n'%(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]))
                    #w.write('mass %d %d %d %d\n'%(gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]))
            with open(os.path.join(output_dir, 'predicted', im_name + '.txt'), 'w') as w:
                for idx in range(cls_boxes_i[1].shape[0]):
                    w.write('mass %.5f %d %d %d %d\n'%(cls_boxes_i[1][idx][4], cls_boxes_i[1][idx][0], cls_boxes_i[1][idx][1],
                            cls_boxes_i[1][idx][2], cls_boxes_i[1][idx][3]))

        if cfg.VIS:
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            if cfg.TEST.VIS_TEST_ONLY:
                vis_utils.vis_one_image(
                    im[:, :, ::-1].astype('uint8'),
                    '{:d}_{:s}'.format(i, im_name),
                    os.path.join(output_dir, 'vis'),
                    cls_boxes_i,
                    segms=cls_segms_i,
                    keypoints=cls_keyps_i,
                    thresh=cfg.VIS_TH,
                    box_alpha=0.8,
                    dataset=dataset,
                    show_class=True
                )
            else:
                if cfg.DATA_SOURCE == 'coco': #or cfg.DATA_SOURCE == 'lesion':
                    other_im_show = im[:, :, ::-1].astype('uint8')
                    if cfg.TEST.VIS_SINGLE_SLICE:
                        other_im_show = cv2.merge([other_im_show[:,:,1], other_im_show[:,:,1], other_im_show[:,:,1]])
                    gt_boxes_show = [entry['boxes'].tolist(), []]
                    gt_classes = [entry['gt_classes'].tolist(), []]
                elif cfg.DATA_SOURCE == 'mammo':
                    if (cfg.MODEL.LR_VIEW_ON  or cfg.MODEL.GIF_ON or cfg.MODEL.LRASY_MAHA_ON):
                        other_im_show = other_im[:, :, ::-1].astype('uint8')
                    else:
                        other_im_show = im[:, :, ::-1].astype('uint8')
                    gt_boxes_show = get_vis_gt_bboxes(entry)
                    gt_classes = None
                im_show = im[:, :, ::-1].astype('uint8')
                if cfg.TEST.VIS_SINGLE_SLICE:
                    im_show = cv2.merge([im_show[:,:,1], im_show[:,:,1], im_show[:,:,1]])
                vis_gt_utils.vis_one_image(
                    im_show,
                    other_im_show,
                    '{:d}_{:s}'.format(i, im_name),
                    os.path.join(output_dir, 'vis'),
                    cls_boxes_i,
                    gt_boxes_show,
                    segms=cls_segms_i,
                    thresh=cfg.VIS_TH,
                    box_alpha=0.8,
                    dataset=dataset,
                    show_class=True,
                    gt_classes = gt_classes
                )

        if cfg.DATA_SOURCE == 'mammo':
            cls_boxes_i = unalign_boxes(entry, shape, cls_boxes_i)  # cls_boxes_i[c]:array n x 5

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            if cfg.DATA_SOURCE == 'mammo':
                cls_segms_i = unalign_segms(entry, shape, cls_segms_i)
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        if i % 500 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, misc_time, eta
                )
            )
    if cfg.TEST.OFFLINE_MAP and cfg.DATA_SOURCE == 'mammo':
        os.system("python ./lib/datasets/map_evaluator.py --output_dir=%s -na -np"%output_dir)

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    if cfg.LESION.USE_POSITION:
        print('####'*10, true_pos,'/',num_images)
        print('position acc: ', float(true_pos)/num_images)
    return all_boxes, all_segms, all_keyps


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = GetRCNNModel()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)
    if cfg.TRAIN_SYNC_BN:
        # Shu:For synchorinized BN
        patch_replication_callback(model)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    if cfg.DATA_SOURCE == 'coco':
        dataset = JsonDataset(dataset_name)
    elif cfg.DATA_SOURCE == 'mammo':
        dataset = MammoDataset(dataset_name)
    elif cfg.DATA_SOURCE == 'lesion':
        dataset = LesionDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        if cfg.DATA_SOURCE == 'coco':
            roidb = dataset.get_roidb(gt=True)
        elif cfg.DATA_SOURCE == 'mammo':
            roidb = dataset.get_roidb(
                gt=True,
                proposal_file='',
                crowd_filter_thresh=0)
        #elif cfg.DATA_SOURCE == 'lesion':
        #    roidb = dataset.get_roidb(
        #       gt=True)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def __unalign_xy(x, y, offx, offy, h, w):
    new_x = min(max(0, (x + offx)), w)
    new_y = min(max(0, (y + offy)), h)
    return [int(new_x), int(new_y)]

def unalign_boxes(entry, shape, cls_boxes_i):
    """
    mapping alignment coordinates to origin coordinates
    :param bounds: bbox and segms of mass prediction
    :param offset: bbox of alighment(y1, x1, y2, x2)
    :param shape: the shape of origin image:(h, w)
    :return: coordinates in origin image
    """
    h, w = shape
    offset_y, offset_x = entry['bbox'][0:2]

    for cls in range(len(cls_boxes_i)):
        for i, item in enumerate(cls_boxes_i[cls]):
            x1, y1 = __unalign_xy(item[0], item[1], offset_x, offset_y, h, w)
            x2, y2 = __unalign_xy(item[2], item[3], offset_x, offset_y, h, w)
            cls_boxes_i[cls][i][:4] = [x1, y1, x2, y2]

    return cls_boxes_i

def unalign_segms(entry, shape, cls_segms_i):
    h, w = shape
    offset_y, offset_x = entry['bbox'][0:2]
    for cls in range(len(cls_segms_i)):
        for i, segm in enumerate(cls_segms_i[cls]):
            cls_segms_i[cls][i] = np.array([__unalign_xy(x, y, offset_x, offset_y, h, w) for (x, y) in segm])
    return cls_segms_i

def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
