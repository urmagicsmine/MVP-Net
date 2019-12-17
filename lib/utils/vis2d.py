# Author __shuzhang__@9.30.18 DeepWise Inc.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np 
import sys 
import os
import os.path as osp
sys.path.insert(0, '/home/zhangshu/code/MITOK2/')
#from lib.image.draw import *
import math
import copy


def xy_2_wh(cube):
    '''
       parameters: cube in  x1 y1 x2 y2 
       return: cube in x1 y1 w h
    '''
    tmp_cube = copy.deepcopy(cube)
    tmp_cube[2:4] = (tmp_cube[2:4]  - tmp_cube[0:2]) + 1 
    return tmp_cube


def RectInfoFromCube(cube):
    '''
       parameters: cube in z_center, y_center, x_center, d, h, w order
       return: the tri-rectangle information
    '''
    tmp_cube = copy.deepcopy(cube)
    rect = xy_2_wh(tmp_cube)
    return rect[0], rect[1], rect[2], rect[3]


def draw_pred_and_gt(image_tensor, gt_bbox, pred_bbox, pred_scores=None, is_draw_bounds=True, \
              is_only_return_drawn_slices=True, gt_color=[0,255,0], pred_color=[0,0,255], text_color=[255,0,255]):
    """
    param: image_tensor: d,h,w format np.array
           gt_bbox: np.array n*6 (x1,y1,z1,x2,y2,z2)
    """

    channel, y_bound, x_bound = image_tensor.shape
    draw_imgs = {0: gray2rgb(image_tensor[0,:,:])}
    clean_imgs = copy.deepcopy(draw_imgs)
    thick = 1
    drawn_slices = []
    if pred_scores is None:
        pred_scores = np.ones(pred_bbox.shape[0])
    else:
        assert pred_bbox.shape[0] == pred_scores.shape[0], "Pred bbox and scores must have the same length!"
 
    for idx in range(gt_bbox.shape[0]):
        gtbbox = gt_bbox[idx, :]
        #print(gtbbox)
        if is_draw_bounds:
            rect_x_top, rect_y_top, rect_w, rect_h = RectInfoFromCube(gtbbox)
            drawn_slices.append(0)
            draw_rect(draw_imgs[0], rect_x_top, rect_y_top, rect_w, rect_h, gt_color, thick)
    for idx in range(pred_bbox.shape[0]):
        predbbox = pred_bbox[idx, :]
        #print(gtbbox)
        if is_draw_bounds:
            rect_x_top, rect_y_top, rect_w, rect_h = RectInfoFromCube(predbbox)
            drawn_slices.append(0)
            draw_rect(draw_imgs[0], rect_x_top, rect_y_top, rect_w, rect_h, pred_color, thick)
    drawn_slices = list(set(drawn_slices))
    if is_only_return_drawn_slices:
        return {i: draw_imgs[i] for i in drawn_slices}, {i: clean_imgs[i] for i in drawn_slices}

    return draw_imgs, clean_imgs

def draw_pred_and_gt_tensor(img_tensor, gt_bbox, save_path, pred_bbox, pred_scores=None, draw_gts=False):
    draw_imgs, clean_imgs = draw_pred_and_gt(img_tensor, gt_bbox, pred_bbox, pred_scores=pred_scores)
    for idx, img in draw_imgs.items():
        if draw_gts:
            cv2.imwrite(save_path + '.jpg', np.hstack((img, clean_imgs[idx])))
        else:
            cv2.imwrite(save_path + '.jpg', img)

