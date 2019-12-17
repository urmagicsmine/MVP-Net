import numpy as np
from scipy import interpolate
import os
import argparse

import _init_paths
from datasets.json_dataset import JsonDataset
from six.moves import cPickle as pickle
import pdb
np.seterr(divide='ignore',invalid='ignore')

def parse_args():

    parser = argparse.ArgumentParser(description='Test lesion FROC')
    parser.add_argument('--file', help='path to detections.pkl')
    parser.add_argument('--dataset', default='lesion_test', help='test dataset name')

    return parser.parse_args()

def IOU(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps


def num_true_positive(boxes, gts, num_box, iou_th):
    # only count once if one gt is hit multiple times
    hit = np.zeros((gts.shape[0],), dtype=np.bool)
    scores = boxes[:, -1]
    boxes = boxes[scores.argsort()[::-1], :4]

    for i, box1 in enumerate(boxes):
        if i == num_box: break
        overlaps = IOU(box1, gts)
        hit = np.logical_or(hit, overlaps >= iou_th)

    tp = np.count_nonzero(hit)

    return tp


def recall_all(boxes_all, gts_all, num_box, iou_th):
    # Compute the recall at num_box candidates per image
    nCls = len(boxes_all)
    nImg = len(boxes_all[0])
    recs = np.zeros((nCls, len(num_box)))
    nGt = np.zeros((nCls,), dtype=np.float)

    for cls in range(nCls):
        for i in range(nImg):
            nGt[cls] += gts_all[cls][i].shape[0]
            for n in range(len(num_box)):
                tp = num_true_positive(boxes_all[cls][i], gts_all[cls][i], num_box[n], iou_th)
                recs[cls, n] += tp
    recs /= nGt
    return recs


def FROC(boxes_all, gts_all, iou_th):
    # Compute the FROC curve, for single class only
    nImg = len(boxes_all)
    # img_idxs_ori : array([   0.,    0.,    0., ..., 4830., 4830., 4830.])
    img_idxs_ori = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)]).astype(int)
    boxes_cat = np.vstack(boxes_all)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs_ori[ord]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    no_lesion = 0
    for i in range(len(boxes_cat)):
        overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
        if overlaps.shape[0] == 0:
            no_lesion += 1
            nMiss += 1
        elif overlaps.max() < iou_th:
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1

        tps.append(nHits)
        fps.append(nMiss)
    nGt = len(np.vstack(gts_all))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg
    print('FROC:FP in no-lesion-images: ', no_lesion)
    return sens, fp_per_img

def sens_at_FP(boxes_all, gts_all, avgFP, iou_th):
    # compute the sensitivity at avgFP (average FP per image)
    sens, fp_per_img = FROC(boxes_all, gts_all, iou_th)
    max_fp = fp_per_img[-1]
    f = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
    valid_avgFP_end_idx = np.argwhere(np.array(avgFP) > max_fp)[0][0]
    valid_avgFP = np.hstack((avgFP[:valid_avgFP_end_idx], max_fp))
    res = f(valid_avgFP)
    return res,valid_avgFP



def get_gt_boxes(roidb):
    gt_boxes = [[] for _ in range(len(roidb))]
    for i, entry in enumerate(roidb):
        gt_boxes[i] = roidb[i]['boxes']
    return gt_boxes

def eval_FROC(dataset, all_boxes, avgFP=[0.5,1,2,3,4,8,16,32,64], iou_th=0.5):
    # all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    # only one class for lesion dataset.
    # all_boxes[1][image] = N X 5

    dataset = JsonDataset(dataset)
    roidb = dataset.get_roidb(gt = True)
    print('training sample:', len(roidb))
    gt_boxes = get_gt_boxes(roidb)

    result, valid_avgFP = sens_at_FP(all_boxes[1], gt_boxes, avgFP, iou_th)
    print('='*40)
    for recall,fp in zip(result,valid_avgFP):
        print('Recall@%.1f=%.2f%%' % (fp, recall*100))
    print('Mean FROC is %.2f'% np.mean(np.array(result)*100))
    print('='*40)

def main():
    args = parse_args()
    det_file = args.file
    dataset_name = args.dataset

    # detections['all_boxes'][cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    # only one class here.
    # all_boxes[image] = N X 5
    with open(det_file, 'rb') as f:
        detections = pickle.load(f)
    all_boxes = detections['all_boxes']

    val_fp = [0.5,1,2,3,4,8,16,32,64]
    eval_FROC(dataset_name, all_boxes, val_fp, 0.5)


if __name__ == '__main__':
    main()




