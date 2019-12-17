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

"""Functions for evaluating results computed for deeplesion dataset."""

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
from scipy import misc
import cv2
from utils.myio import read_json
from pycocotools.cocoeval import COCOeval
from utils.io import save_object
import utils.boxes as box_utils

logger = logging.getLogger(__name__)


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


def get_bbox_predicts(image_id, bboxes_data):
    scores = []
    bboxes = []
    for i, candidate in enumerate(bboxes_data):
        if candidate['image_id'] == image_id:
            scores.append(candidate['score'])
            bboxes.append(bboxes_data[i]['bbox'])
    return bboxes, scores


def evaluate_boxes(
    json_dataset, all_boxes, output_dir, use_salt=True, cleanup=False
):
    res_file = os.path.join(
        output_dir, 'bbox_' + json_dataset.name + '_results'
    )
    if use_salt:
        res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    bbox_data = _write_lesion_results_file(json_dataset, all_boxes, res_file)
    do_bboxes_eval(json_dataset, bbox_data, thresh=0.5)
    # Optionally cleanup results json file
    if cleanup:
        os.remove(res_file)

def get_gt_bboxes(entry):
    bbox_dict = {
        'lesion': [],
        'exclude': []
    }
    for box_ind in range(entry['boxes'].shape[0]):
        x1, y1, x2, y2 = entry['boxes'][box_ind, :]
        w, h = x2 - x1, y2 - y1
        bbox_dict['lesion'].append([x1, y1, w, h])
    return bbox_dict


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
    max_label = len(gt_bboxes['lesion'])
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
            HittedLabels = if_bbox_overlap(gt_bboxes['lesion'], predict_bbox, thresh)
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



def do_bboxes_eval(json_dataset, all_bboxes, thresh = 0.25):
    print("Eval bboxes with IOU threshold of %.2f"%thresh)
    roidb = json_dataset.get_roidb(
        gt=True,
        proposal_file=None,
        crowd_filter_thresh=0)
    FROC_data = np.zeros((4, len(roidb)), dtype=np.object)
    FP_summary = np.zeros((2, len(roidb)), dtype=np.object)
    detection_summary = np.zeros((2, len(roidb)), dtype=np.object)
    #thresh = 0.25
    for i, entry in enumerate(roidb):
        image_name = entry['image']
        gt_bboxes = get_gt_bboxes(entry)
        predict_bboxes, scores = get_bbox_predicts(image_name, all_bboxes)
        FROC_data[0][i] = image_name
        FP_summary[0][i] = image_name
        FROC_data[0][i] = image_name
        FROC_data[1][i], FROC_data[2][i], FROC_data[3][i], detection_summary[1][i], FP_summary[1][
            i] = compute_bbox_FP_TP_Probs(gt_bboxes, predict_bboxes, scores, thresh)
    total_FPs, total_sensitivity, all_probs = computeFROC(FROC_data)
    froc_fp = []
    froc_recall = []
    for fp in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 16.0, max(total_FPs)]:
        index = np.where(total_FPs >= fp)[0]
        if len(index) > 0:
            recall = total_sensitivity[index[-1]]
            prob = all_probs[index[-1]]
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
    print('Mean FROC is %.2f'% np.mean(np.array(froc_recall[0:6])))



def _write_lesion_results_file(json_dataset, all_boxes, res_file):
    # [{"image_id": 42,
    #   "category_id": 18,
    #   "bbox": [258.15,41.29,348.26,243.78],
    #   "score": 0.236}, ...]
    results = []
    for cls_ind, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        if cls_ind >= len(all_boxes):
            break
        cat_id = json_dataset.category_to_id_map[cls]
        results.extend(_lesion_results_one_category(
            json_dataset, all_boxes[cls_ind], cat_id))
    logger.info(
        'Writing bbox results json to: {}'.format(os.path.abspath(res_file)))
    with open(res_file, 'w') as fid:
        json.dump(results, fid)
    return results


def _lesion_results_one_category(json_dataset, boxes, cat_id):
    results = []
    image_ids = json_dataset.COCO.getImgIds()
    image_ids.sort()
    # if json_dataset._image_set == 'train':
    #     image_ids = image_ids[:400]
    print(len(boxes), len(image_ids))
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



