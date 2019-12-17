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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'
ANN_FN_MERGE = 'merged_annotation_file'
NORM_IM_DIR = 'norm_im_dir'
DICOM_DIR = 'dicom_dir'
BBOX_DIR = 'bbox_dir'
BBOX_DIR_DCM = 'bbox_dir_dcm'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'
DATA_LIST = 'data_list'
PID_LIST = 'pid_list'

# Available datasets
DATASETS = {
    'lungbinary_1031_train': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/binary_lesion_1031_train.json'
    },
    'lungbinary_1031_valid': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/binary_lesion_1031_val.json'
    },
    'lungbinary_1031_test': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/binary_lesion_1031_test.json'
    },
    'lungbinary_1031_repeat': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/binary_lesion_1031_repeat.json'
    },
    'lungtop3_1031_train': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/top3_lesion_1031_train.json'
    },
    'lungtop3_1031_valid': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/top3_lesion_1031_val.json'
    },
    'lungtop3_1031_test': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/top3_lesion_1031_test.json'
    },
    'lungtop3_1031_repeat': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/top3_lesion_1031_repeat.json'
    },
    'lungsub18_1031_train': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/sub18_lesion_1031_train.json'
    },
    'lungsub18_1031_valid': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/sub18_lesion_1031_val.json'
    },
    'lungsub18_1031_test': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/sub18_lesion_1031_test.json'
    },
    'lungsub18_1031_repeat': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/sub18_lesion_1031_repeat.json'
    },
    'lungsuper8_1031_train': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/super8_lesion_1031_train.json'
    },
    'lungsuper8_1031_valid': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/super8_lesion_1031_val.json'
    },
    'lungsuper8_1031_test': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/super8_lesion_1031_test.json'
    },
    'lungsuper8_1031_repeat': {
        IM_DIR:
            _DATA_DIR + '/lung_det/rgb_slice_norm1.0/',
        ANN_FN:
            _DATA_DIR + '/lung_det/config/coco_json/super8_lesion_1031_repeat.json'
    },
    'lesion_train': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/annotation/deeplesion_train.json'
            #_DATA_DIR + '/DeepLesion/DL_info.csv'
    },
    'lesion_valid': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/DL_info.csv'
    },
    'lesion_test': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/annotation/deeplesion_test.json'
            #_DATA_DIR + '/DeepLesion/DL_info.csv'
    },
    'lesion_train_augment': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/annotation_augment/deeplesion_train.json'
    },
    'lesion_test_augment': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/annotation_augment/deeplesion_test.json'
    },
    'lesion_train_175_275': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/annotation_175_275/deeplesion_train.json'
    },
    'lesion_test_175_275': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/annotation_175_275/deeplesion_test.json'
    },
    'lesion_test_other': {
        IM_DIR:
            _DATA_DIR + '/DeepLesion/Images_png/Images_png',
        ANN_FN:
            _DATA_DIR + '/DeepLesion/annotation_other/deeplesion_test.json'
            #_DATA_DIR + '/DeepLesion/DL_info.csv'
    },
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_drop10_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/drop10_instances_train2017.json',
    },
    'coco_2017_drop10_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_d10n10_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/noise_ann/drop10noise10_instances_train2017.json',
    },
    'coco_2017_d00n20_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/noise_ann/drop00noise20_instances_train2017.json',
    },
    'coco_2017_d40n00_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/noise_ann/drop40noise00_instances_train2017.json',
    },
    'truck_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/one_cat_anns',
    },
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json'
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'keypoints_coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    },
    'domestic': {
        IM_DIR:
            '/home/HDD1/mammogram_data/domestic/norm_roi_pad/image/',
        ANN_FN:
            '/home/HDD1/mammogram_data/domestic/norm_roi_pad/new_cluster3_v2/json/',
        DATA_LIST:
            '/home/HDD1/mammogram_data/domestic_mass/config/mass_{}_name_list.txt',
        PID_LIST:
            '/home/HDD1/mammogram_data/domestic_mass/config/mass_{}_pid_list.txt'
    },
    'gansu1': {
        IM_DIR:
            '/data2/mammogram/gansu_b1_1k/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/gansu_b1_1k/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/gansu_b1_1k/dicom/',
        BBOX_DIR:
            '/data2/mammogram/gansu_b1_1k/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/gansu_b1_1k/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/gansu_b1_1k/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/gansu_b1_1k/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/gansu_b1_1k/config/set/{}_pid_list.txt'
    },
    'gansu2': {
        IM_DIR:
            '/data2/mammogram/gansu_b2/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/gansu_b2/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/gansu_b2/dicom/',
        BBOX_DIR:
            '/data2/mammogram/gansu_b2/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/gansu_b2/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/gansu_b2/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/gansu_b2/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/gansu_b2/config/set/{}_pid_list.txt'
    },
    'gansu3': {
        IM_DIR:
            '/data2/mammogram/gansu_b3/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/gansu_b3/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/gansu_b3/dicom/',
        BBOX_DIR:
            '/data2/mammogram/gansu_b3/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/gansu_b3/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/gansu_b3/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/gansu_b3/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/gansu_b3/config/set/{}_pid_list.txt'
    },
    'gansu4': {
        IM_DIR:
            '/data2/mammogram/gansu_b4/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/gansu_b4/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/gansu_b4/dicom/',
        BBOX_DIR:
            '/data2/mammogram/gansu_b4/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/gansu_b4/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/gansu_b4/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/gansu_b4/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/gansu_b4/config/set/{}_pid_list.txt'
    },
    'hologicmix1': {
        IM_DIR:
            '/data2/mammogram/hologic_mix1_1k/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/hologic_mix1_1k/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/hologic_mix1_1k/dicom/',
        BBOX_DIR:
            '/data2/mammogram/hologic_mix1_1k/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/hologic_mix1_1k/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/hologic_mix1_1k/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/hologic_mix1_1k/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/hologic_mix1_1k/config/set/{}_pid_list.txt'
    },
    'nanjing1': {
        IM_DIR:
            '/data2/mammogram/nanjing_b1/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/nanjing_b1/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/nanjing_b1/dicom/',
        BBOX_DIR:
            '/data2/mammogram/nanjing_b1/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/nanjing_b1/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/nanjing_b1/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/nanjing_b1/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/nanjing_b1/config/set/{}_pid_list.txt'
    },
    'nanjing2': {
        IM_DIR:
            '/data2/mammogram/nanjing_b2/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/nanjing_b2/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/nanjing_b2/dicom/',
        BBOX_DIR:
            '/data2/mammogram/nanjing_b2/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/nanjing_b2/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/nanjing_b2/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/nanjing_b2/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/nanjing_b2/config/set/{}_pid_list.txt'
    },
    'by31': {
        IM_DIR:
            '/data2/mammogram/by3_b1/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/by3_b1/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/by3_b1/dicom/',
        BBOX_DIR:
            '/data2/mammogram/by3_b1/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/by3_b1/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/by3_b1/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/by3_b1/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/by3_b1/config/set/{}_pid_list.txt'
    },
    'shougang1': {
        IM_DIR:
            '/data2/mammogram/shougang_b1_2/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/shougang_b1_2/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/shougang_b1_2/dicom/',
        BBOX_DIR:
            '/data2/mammogram/shougang_b1_2/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/shougang_b1_2/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/shougang_b1_2/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/shougang_b1_2/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/shougang_b1_2/config/set/{}_pid_list.txt'
    },
    'zhongri1': {
        IM_DIR:
            '/data2/mammogram/zhongri_b1_2/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/zhongri_b1_2/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/zhongri_b1_2/dicom/',
        BBOX_DIR:
            '/data2/mammogram/zhongri_b1_2/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/zhongri_b1_2/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/zhongri_b1_2/annotation/json/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/zhongri_b1_2/annotation/json_merge_mass/iomin_0.5/origin/',
        PID_LIST:
            '/data2/mammogram/zhongri_b1_2/config/set/{}_pid_list.txt'
    },
    'gansu2filter': {
        IM_DIR:
            '/data2/mammogram/gansu_b2/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/gansu_b2/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/gansu_b2/dicom/',
        BBOX_DIR:
            '/data2/mammogram/gansu_b2/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/gansu_b2/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/gansu_b2/annotation/json_checked/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/gansu_b2/annotation/json_checked/origin/',
        PID_LIST:
            '/data2/mammogram/gansu_b2/config/set/{}_pid_list.txt'
    },
    'by31filter': {
        IM_DIR:
            '/data2/mammogram/by3_b1/image/origin/',
        NORM_IM_DIR:
            '/data2/mammogram/by3_b1/image/norm_window/origin/',
        DICOM_DIR:
            '/data2/mammogram/by3_b1/dicom/',
        BBOX_DIR:
            '/data2/mammogram/by3_b1/image/origin_bbox/',
        BBOX_DIR_DCM:
            '/data2/mammogram/by3_b1/image/dcm_bbox/',
        ANN_FN:
            '/data2/mammogram/by3_b1/annotation/json_checked/origin/',
        ANN_FN_MERGE:
            '/data2/mammogram/by3_b1/annotation/json_checked/origin/',
        PID_LIST:
            '/data2/mammogram/by3_b1/config/set/{}_pid_list.txt'
    }
}
