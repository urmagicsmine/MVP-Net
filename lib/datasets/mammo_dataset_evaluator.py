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

"""Functions for evaluating results computed for a mammo dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import numpy as np
import os
import uuid


from core.config import cfg
from utils.myio import save_object
import utils.boxes as box_utils
from scipy import misc
import cv2
from utils.myio import read_json
import pycocotools.mask as mask_util
#from Mammogram.lib import Mammogram

logger = logging.getLogger(__name__)


def evaluate_masks(
    mammo_dataset,
    all_boxes,
    all_segms,
    output_dir,
    use_salt=True,
    cleanup=False
):
    res_file = os.path.join(
        output_dir, 'segmentations_' + mammo_dataset.name + '_' + mammo_dataset._image_set + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    segm_data = _write_mammo_segms_results_file(
        mammo_dataset, all_boxes, all_segms, res_file)
    # Only do evaluation on non-test sets (annotations are undisclosed on test)
    if cfg.TEST.MASK:
        do_segms_eval(mammo_dataset, segm_data)
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)


def if_mask_overlap(mask, predict_mask, thresh):
    predict_size = np.sum(predict_mask > 0 * 1)
    for i in range(np.amax(mask)):
        label_size = np.sum(mask==(i+1) * 1)
        intersect = np.sum((mask==(i+1)) * (predict_mask > 0) * 1)
        if intersect == 0:
            continue
        if (intersect * 1. / predict_size) > thresh :
        # if (intersect * 1. / (label_size+predict_size - intersect)) > thresh :
#         if (intersect * 1. / label_size) > thresh or (intersect * 1. / predict_size) > thresh:
            return i+1
    return 0


def compute_FP_TP_Probs(mask, predict_segms, Probs, thresh):
    max_label = np.amax(mask)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label+1):
        label = 'Label' + str(i)
        detection_summary[label] = []
    FP_counter = 0
    if max_label > 0:
        for i, predict_segm in enumerate(predict_segms):
            predict_mask = mask_util.decode(predict_segm)
            HittedLabel = if_mask_overlap(mask, predict_mask, thresh)
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP' + str(FP_counter)
                FP_summary[key] = [Probs[i], predict_segm]
                FP_counter += 1
            elif (Probs[i] > TP_probs[HittedLabel - 1]):
                label = 'Label' + str(HittedLabel)
                detection_summary[label] = [Probs[i], predict_segm]
                TP_probs[HittedLabel-1] = Probs[i]
    else:
        for i, predict_segm in enumerate(predict_segms):
            FP_probs.append(Probs[i])
            key = 'FP' + str(FP_counter)
            FP_summary[key] = [Probs[i], predict_segm]
            FP_counter += 1
    return FP_probs, TP_probs, max_label, detection_summary, FP_summary


def computeFROC(FROC_data):
    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs) / float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs) / float(sum(FROC_data[3]))
    return total_FPs, total_sensitivity, all_probs


def do_segms_eval(mammo_dataset, all_segms):
    roidb = mammo_dataset.get_roidb(
        gt=True,
        proposal_file='',
        crowd_filter_thresh=0)
    FROC_data = np.zeros((4, len(roidb)), dtype=np.object)
    FP_summary = np.zeros((2, len(roidb)), dtype=np.object)
    detection_summary = np.zeros((2, len(roidb)), dtype=np.object)
    thresh = 0.1
    for i, entry in enumerate(roidb):
        image_name = entry['file_name']
        mask, label = get_segm_mask(entry)
        segms, scores = get_segm_predicts(image_name, all_segms)
        FROC_data[0][i] = image_name
        FP_summary[0][i] = image_name
        FROC_data[0][i] = image_name
        FROC_data[1][i], FROC_data[2][i], FROC_data[3][i], detection_summary[1][i], FP_summary[1][
            i] = compute_FP_TP_Probs(mask, segms, scores, thresh)
    total_FPs, total_sensitivity, all_probs = computeFROC(FROC_data)
    for fp in [0.2, 0.4, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, max(total_FPs)]:
        index = np.where(total_FPs <= fp)[0]
        if len(index) > 0:
            recall = total_sensitivity[index[0]]
            prob = all_probs[index[0]]
    # logger.info('R@%d=%.2f%% ' % (fp, total_recall_num * 100. / total_pos_num))
            score_thresh = prob
            tp_count = 0
            tp_sum = 0
            for item in FROC_data[2]:
                tp_sum += len(item)
                if len(item) > 0:
                    for score in item:
                        if score > score_thresh:
                            tp_count += 1
            # print('%d / %d, TPR=%.2f%% ' % (tp_count, tp_sum, tp_count * 1. / tp_sum))

            fp_count = 0
            for j in range(len(FP_summary[1])):
                for name in FP_summary[1][j].keys():
                    score = FP_summary[1][j][name][0]
                    if score > score_thresh:
                        fp_count += 1
            print('Recall@%.1f=%.2f%% , thresh=%.2f, %d / %d, FPR=%.2f%%' %
                  (fp, recall, prob, fp_count, fp_count + tp_count, fp_count * 1. / (fp_count + tp_count)))


def get_iou(bb1, bb2):
    a_x1, a_y1, a_w, a_h = bb1
    b_x1, b_y1, b_w, b_h = bb2
    a_x2, a_y2 = a_x1 + a_w, a_y1 + a_h
    b_x2, b_y2 = b_x1 + b_w, b_y1 + b_h
    x_left = max(a_x1, b_x1)
    y_top = max(a_y1, b_y1)
    x_right = min(a_x2, b_x2)
    y_bottom = min(a_y2, b_y2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = a_w * a_h
    bb2_area = b_w * b_h

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_segm_mask(item, size=4096):
    unique_label = []
    x1, y1, x2, y2 = item['bbox']
    h = item['height']
    w = item['width']
    mask = np.zeros((h, w), dtype='uint8')
    i = 0
    for flat_list in item['segms']:
        points = np.array(flat_list[0], dtype=np.int32).reshape((-1, 2))
        label_bbox = cv2.boundingRect(points[:, np.newaxis, :])
        if label_bbox[2] * label_bbox[3] < size:
            continue
        overlap= False
        if label_bbox not in unique_label:
            for label in unique_label:
                if get_iou(label_bbox, label) > 0.6:
                    overlap=True
                    break
            if not overlap:
                unique_label.append(label_bbox)
                i += 1
                cv2.drawContours(mask, (points, ), 0, color=i, thickness=-1)
    return mask, unique_label


def get_segm_predicts(image_id, segms_data):
    scores = []
    segms = []
    for i, candidate in enumerate(segms_data):
        if candidate['image_id'] == image_id:
            scores.append(candidate['score'])
            segms.append(segms_data[i]['segmentation'])
    return segms, scores


def get_bbox_predicts(image_id, bboxes_data):
    scores = []
    bboxes = []
    for i, candidate in enumerate(bboxes_data):
        if candidate['image_id'] == image_id:
            scores.append(candidate['score'])
            bboxes.append(bboxes_data[i]['bbox'])
    return bboxes, scores


def _write_mammo_segms_results_file(
    mammo_dataset, all_boxes, all_segms, res_file
):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "segmentation": [...],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(mammo_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = mammo_dataset.category_to_id_map[cls]
        results.extend(_mammo_segms_results_one_category(
            mammo_dataset, all_boxes[cls_ind], all_segms[cls_ind], cat_id))
    logger.info(
        'Writing segmentation results json to: {}'.format(
            os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)
    return results


def _mammo_segms_results_one_category(mammo_dataset, boxes, segms, cat_id):
    results = []
    image_ids = mammo_dataset._image_index

    image_ids.sort()
    assert len(boxes) == len(image_ids)
    assert len(segms) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        rles = segms[i]

        if isinstance(dets, list) and len(dets) == 0:
            continue

        dets = dets.astype(np.float)
        scores = dets[:, -1]

        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'segmentation': rles[k].tolist(),
              'score': scores[k]}
              for k in range(dets.shape[0])])

    return results



def evaluate_boxes(
    mammo_dataset, all_boxes, output_dir, use_salt=True, cleanup=False
):
    res_file = os.path.join(
        output_dir, 'bbox_' + mammo_dataset.name + '_' + mammo_dataset._image_set + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    bbox_data = _write_mammo_results_file(mammo_dataset, all_boxes, res_file)
    do_bboxes_eval(mammo_dataset, bbox_data)
    do_bboxes_eval(mammo_dataset, bbox_data, thresh=0.5)
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)

def get_gt_bboxes(mammo_dataset, item):
    json_paths = mammo_dataset.label_path_from_index(item['file_name'])
    bbox_dict = {
        'mass': [],
        'exclude': []
    }
    for ann_json_path in json_paths:
        data = read_json(ann_json_path)
        mam = Mammogram(data)
        for les in mam.lesions:
            (x, y), (x1, y1) = les.get_bbox()
            w, h = x1 - x, y1 - y
            if les.type == 'mass' and les.get_malignancy_level() >= 2:
                bbox_dict['mass'].append([x, y, w, h])
            if (les.type == 'mass') or ('asymmetry' in les.type) or ('astmmetry' in les.type) or ('distortion' in les.type):
                bbox_dict['exclude'].append([x, y, w, h])
    return bbox_dict

def get_vis_gt_bboxes(item):
    unique_label = []
    exclude_label = []
    boxes = item['boxes']
    overlaps = item['gt_overlaps'].toarray()
    for box, overlap in zip(boxes, overlaps):
        if sum(box) > 0 and sum(overlap)> 0:
            unique_label.append(box)
        elif sum(box) > 0 and sum(overlap)< 0:
            exclude_label.append(box)
    #for box in item['boxes']:
    #    if box not in unique_label:
    #        np.append(box)
    return [unique_label, exclude_label]


def if_bbox_overlap(gt_bboxes, predict_bbox, thresh):
    a_x1, a_y1, a_w, a_h = predict_bbox
    a_x2 = a_x1 + a_w
    a_y2 = a_y1 + a_h
    predict_size = a_w * a_h
    hit_list = []
    for i, gt_bbox in enumerate(gt_bboxes):
        b_x1, b_y1, b_w, b_h = gt_bbox
        b_x2, b_y2 = b_x1+b_w, b_y1+b_h
        label_size = b_w * b_h
        x_left = max(a_x1, b_x1)
        y_top = max(a_y1, b_y1)
        x_right = min(a_x2, b_x2)
        y_bottom = min(a_y2, b_y2)
        if x_right < x_left or y_bottom < y_top:
            continue
        intersect = (x_right - x_left) * (y_bottom - y_top)
        if (intersect * 1. / (predict_size + label_size - intersect)) > thresh:
            hit_list.append(i+1)
    return hit_list


def compute_bbox_FP_TP_Probs(gt_bboxes, predict_bboxes, Probs, thresh):
    max_label = len(gt_bboxes['mass'])
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1, max_label+1):
        label = 'Label' + str(i)
        detection_summary[label] = []
    FP_counter = 0
    if max_label > 0:
        for i, predict_bbox in enumerate(predict_bboxes):
            HittedLabels = if_bbox_overlap(gt_bboxes['mass'], predict_bbox, thresh)
            if len(HittedLabels) == 0:
                if len(if_bbox_overlap(gt_bboxes['exclude'], predict_bbox, thresh)) == 0:
                    FP_probs.append(Probs[i])
                    key = 'FP' + str(FP_counter)
                    FP_summary[key] = [Probs[i], predict_bbox]
                    FP_counter += 1
            else:
                for HittedLabel in HittedLabels:
                    if (Probs[i] > TP_probs[HittedLabel - 1]):
                        label = 'Label' + str(HittedLabel)
                        detection_summary[label] = [Probs[i], predict_bbox]
                        TP_probs[HittedLabel-1] = Probs[i]
    else:
        for i, predict_bbox in enumerate(predict_bboxes):
            FP_probs.append(Probs[i])
            key = 'FP' + str(FP_counter)
            FP_summary[key] = [Probs[i], predict_bbox]
            FP_counter += 1
    return FP_probs, TP_probs, max_label, detection_summary, FP_summary



def do_bboxes_eval(mammo_dataset, all_bboxes, thresh = 0.25):
    print("Eval bboxes with IOU threshold of %.2f"%thresh)
    roidb = mammo_dataset.get_roidb(
        gt=True,
        proposal_file='',
        crowd_filter_thresh=0)
    FROC_data = np.zeros((4, len(roidb)), dtype=np.object)
    FP_summary = np.zeros((2, len(roidb)), dtype=np.object)
    detection_summary = np.zeros((2, len(roidb)), dtype=np.object)
    #thresh = 0.25
    for i, entry in enumerate(roidb):
        image_name = entry['file_name']
        gt_bboxes = get_gt_bboxes(mammo_dataset, entry)
        predict_bboxes, scores = get_bbox_predicts(image_name, all_bboxes)
        FROC_data[0][i] = image_name
        FP_summary[0][i] = image_name
        FROC_data[0][i] = image_name
        FROC_data[1][i], FROC_data[2][i], FROC_data[3][i], detection_summary[1][i], FP_summary[1][
            i] = compute_bbox_FP_TP_Probs(gt_bboxes, predict_bboxes, scores, thresh)
    total_FPs, total_sensitivity, all_probs = computeFROC(FROC_data)
    froc_fp = []
    froc_recall = []
    for fp in [0.2, 0.4, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, max(total_FPs)]:
        index = np.where(total_FPs <= fp)[0]
        if len(index) > 0:
            recall = total_sensitivity[index[0]]
            prob = all_probs[index[0]]
            score_thresh = prob
            tp_count = 0
            tp_sum = 0
            for item in FROC_data[2]:
                tp_sum += len(item)
                if len(item) > 0:
                    for score in item:
                        if score > score_thresh:
                            tp_count += 1

            fp_count = 0
            for j in range(len(FP_summary[1])):
                for name in FP_summary[1][j].keys():
                    score = FP_summary[1][j][name][0]
                    if score > score_thresh:
                        fp_count += 1
            froc_recall.append(recall*100)
            print('Recall@%.1f=%.2f%% , thresh=%.2f, tp: %d / %d, FPR=%.2f%%' %
                  (fp, recall * 100, prob, tp_count, tp_sum, fp_count * 100. / (fp_count + tp_count)))
    print('Mean FROC is %.2f'% np.mean(np.array(froc_recall[0:9])))



def _write_mammo_results_file(mammo_dataset, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(mammo_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = mammo_dataset.category_to_id_map[cls]
        results.extend(_mammo_results_one_category(
            mammo_dataset, all_boxes[cls_ind], cat_id))
    logger.info(
        'Writing bbox results json to: {}'.format(os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)
    return results


def _mammo_results_one_category(mammo_dataset, boxes, cat_id):
    results = []
    image_ids = mammo_dataset._image_index
    image_ids.sort()
    # if mammo_dataset._image_set == 'train':
    #     image_ids = image_ids[:400]
    assert len(boxes) == len(image_ids)
    for i, image_id in enumerate(image_ids):
        dets = boxes[i]
        if isinstance(dets, list) and len(dets) == 0:
            continue
        dets = dets.astype(np.float)
        scores = dets[:, -1]
        xywh_dets = box_utils.xyxy_to_xywh(dets[:, 0:4])
        xs = xywh_dets[:, 0]
        ys = xywh_dets[:, 1]
        ws = xywh_dets[:, 2]
        hs = xywh_dets[:, 3]
        results.extend(
            [{'image_id': image_id,
              'category_id': cat_id,
              'bbox': [xs[k], ys[k], ws[k], hs[k]],
              'score': scores[k]} for k in range(dets.shape[0])])
    return results



def if_overlap(predict, label, cutoff=.1):
    x1, y1, w1, h1 = predict
    x2, y2, w2, h2 = label
    predict_area = w1 * h1
    roi_area = w2 * h2
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if dx > 0 and dy > 0:
        inter_area = dx * dy
    else:
        return False
    return inter_area * 1.0/roi_area > cutoff or inter_area * 1.0/predict_area > cutoff

def gen_lesion_mask(name, mammo_dataset):
    image = misc.imread(mammo_dataset.image_path_from_index(name))
    mask = np.zeros(image.shape, dtype='uint8')
    pos_bounds = []
    json_file = mammo_dataset.annotation_directory + name + '.txt'
    label_data = read_json(json_file)
    for nodule in label_data['nodes']:
        if nodule['type'].lower() == 'mass':
            contours = nodule['rois'][0]['edge']
            pos_bounds.append([])
            if len(contours) == 1:
                x, y = contours[0]
                cv2.circle(mask, (x, y), 5, 255, -1)
                label_bbox = [x - 1, y - 1, 2, 2]

            else:
                cv2.drawContours(mask, (np.int32(contours), ), 0, color=255, thickness=-1)
                points = np.array(contours)
                points = points[:, np.newaxis, :]

                label_bbox = cv2.boundingRect(points)
            pos_bounds.append(label_bbox)
    return mask, pos_bounds




def evaluate_box_proposals(
    json_dataset, roidb, thresholds=None, area='all', limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        'all': 0,
        'small': 1,
        'medium': 2,
        'large': 3,
        '96-128': 4,
        '128-256': 5,
        '256-512': 6,
        '512-inf': 7}
    area_ranges = [
        [0**2, 1e5**2],    # all
        [0**2, 32**2],     # small
        [32**2, 96**2],    # medium
        [96**2, 1e5**2],   # large
        [96**2, 128**2],   # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2]]  # 512-inf
    assert area in areas, 'Unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_boxes = entry['boxes'][gt_inds, :]
        gt_areas = entry['seg_areas'][gt_inds]
        valid_gt_inds = np.where(
            (gt_areas >= area_range[0]) & (gt_areas <= area_range[1]))[0]
        gt_boxes = gt_boxes[valid_gt_inds, :]
        num_pos += len(valid_gt_inds)
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        boxes = entry['boxes'][non_gt_inds, :]
        if boxes.shape[0] == 0:
            continue
        if limit is not None and boxes.shape[0] > limit:
            boxes = boxes[:limit, :]
        overlaps = box_utils.bbox_overlaps(
            boxes.astype(dtype=np.float32, copy=False),
            gt_boxes.astype(dtype=np.float32, copy=False))
        _gt_overlaps = np.zeros((gt_boxes.shape[0]))
        for j in range(min(boxes.shape[0], gt_boxes.shape[0])):
            # find which proposal box maximally covers each gt box
            argmax_overlaps = overlaps.argmax(axis=0)
            # and get the iou amount of coverage for each gt box
            max_overlaps = overlaps.max(axis=0)
            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ind = max_overlaps.argmax()
            gt_ovr = max_overlaps.max()
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        # append recorded iou coverage level
        gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    gt_overlaps = np.sort(gt_overlaps)
    if thresholds is None:
        step = 0.05
        thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps, 'num_pos': num_pos}


