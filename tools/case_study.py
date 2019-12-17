import numpy as np
from scipy import interpolate
import os
import shutil
import cv2
import argparse

import _init_paths
from datasets.json_dataset import JsonDataset
from six.moves import cPickle as pickle
import pdb

np.seterr(divide='ignore',invalid='ignore')
# windows: origin and multi-window 1,2,3
windows = [[-1024,3071],[-174,274],[-1493,484],[-534,1425]]
score_threshold = 0.5
def parse_args():

    parser = argparse.ArgumentParser(description='Test lesion FROC')
    parser.add_argument('--baseline', default='/home/lizihao/baseline.pkl', help='path to the baseline detections.pkl')
    parser.add_argument('--ours', help='path to our detections.pkl')
    parser.add_argument('--dataset', default='lesion_test', help='test dataset name')

    return parser.parse_args()

def windowing(img, window):
	im = img.copy()
	im = (im-window[0])/(window[1]-window[0])
	return im

def draw_det_results(im, boxes, gts):
    for box in boxes:
        if box[4] < score_threshold:
            continue
        #center = (int((box[0]+box[2])/2), int((box[1]+box[3])/2))
        #axes = (int((box[2]-box[0])), int((box[3]-box[1])))
        #cv2.ellipse(im, center, axes, 0, 0, 360, (0, 0 , 255), 2)
        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 1)
    for box in gts:
        cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), 1)
    return im

def save_image(boxes1, boxes2, gts, image_path, save_dir):
    name_split = image_path.split('/')
    save_name = name_split[-2] + '__' + name_split[-1]
    # read image
    img = cv2.imread(image_path, -1)
    img = img.astype(np.float32) - 32768
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # draw detections
    ori_img = windowing(img,windows[0])
    img_0 = draw_det_results(ori_img, boxes1, gts)
    ori_img = windowing(img,windows[1])
    img_1 = draw_det_results(ori_img, boxes2, gts)
    ori_img = windowing(img,windows[2])
    img_2 = draw_det_results(ori_img, boxes2, gts)
    ori_img = windowing(img,windows[3])
    img_3 = draw_det_results(ori_img, boxes2, gts)
    img_concat = np.hstack((img_0, img_1, img_2, img_3))
    # save image
    cv2.imwrite(os.path.join(save_dir, save_name), img_concat*255)

def get_roidb_info(roidb):
    gt_boxes = [[] for _ in range(len(roidb))]
    name_list = []
    for i, entry in enumerate(roidb):
        gt_boxes[i] = roidb[i]['boxes']
        name_list.append(roidb[i]['image'])
    return gt_boxes,name_list

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

def get_hits(boxes, gt_boxes, iou_th=0.5):
    """
    Input:  pred boxes and gt boxes of an image.
    Output: num of hits(ture positive) in one image.
    """
    hit = np.zeros((gt_boxes.shape[0],), dtype=np.bool)
    for box in boxes:
        if box[4] < score_threshold:
            continue
        overlaps = IOU(box, gt_boxes)
        hits = overlaps>= iou_th
        hit = np.logical_or(hit, overlaps >= iou_th)
    tp = np.count_nonzero(hit)
    return tp

def find_diff_detections(boxes_all1, boxes_all2, roidb, save_dir, iou_th):
    gts_all,name_list = get_roidb_info(roidb)
    nImg = len(boxes_all1)
    diff = 0
    ours = 0
    baseline = 0
    print('starting ...')
    for i in range(nImg-3000):
        hits1 = get_hits(boxes_all1[i], gts_all[i], iou_th)
        hits2 = get_hits(boxes_all2[i], gts_all[i], iou_th)
        #print(hits1, hits2)
        if not hits1 == hits2:
            diff+=1
            if hits1 > hits2:
                baseline += 1
            else:
                ours += 1
                save_image(boxes_all1[i], boxes_all2[i], gts_all[i], name_list[i], save_dir)

    print(len(boxes_all1), len(boxes_all2), len(gts_all),len(name_list))
    print('starting ...')
    print('found {} different cases totally. \
            Ours better:{}, baseline better{}'.format(diff, ours, baseline))

def main():
    args = parse_args()
    det_file1 = args.baseline
    det_file2 = args.ours
    dataset_name = args.dataset
    work_dir = os.getcwd()
    save_dir = os.path.join(work_dir, '..' ,'Outputs', 'case_study')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
    # detections['all_boxes'][cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    # only one class in DeepLesion. 0-background, 1-lesion.
    # all_boxes[image] = a N*5 list.

    with open(det_file1, 'rb') as f:
        detections = pickle.load(f)
        all_boxes1= detections['all_boxes']
    with open(det_file2, 'rb') as f:
        detections = pickle.load(f)
        all_boxes2= detections['all_boxes']

    dataset = JsonDataset(dataset_name)
    roidb = dataset.get_roidb(gt = True)
    find_diff_detections(all_boxes1[1], all_boxes2[1], roidb, save_dir, iou_th=0.5)


if __name__ == '__main__':
    main()
