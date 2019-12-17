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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from glob import glob
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse
import json
import cv2

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API

import utils.boxes as box_utils
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_FN,ANN_FN_MERGE
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR, NORM_IM_DIR, BBOX_DIR, DICOM_DIR, BBOX_DIR_DCM
from .dataset_catalog import PID_LIST
from utils.myio import read_json
from pycocotools import mask

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MammoDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name):

        logger.debug('Creating: {}'.format(name))
        self.name = name.split('_')[0]
        self._image_set = name.split('_')[-1]

        assert self.name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(self.name)
        assert os.path.exists(DATASETS[self.name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[self.name][IM_DIR])
        assert os.path.exists(DATASETS[self.name][ANN_FN]), \
            'Annotation file \'{}\' not found'.format(DATASETS[self.name][ANN_FN])
        if cfg.USE_MERGED_ANN:
            assert os.path.exists(DATASETS[self.name][ANN_FN_MERGE]), \
                'Annotation file \'{}\' not found'.format(DATASETS[self.name][ANN_FN_MERGE])
        # assert os.path.exists(DATASETS[self.name][DATA_LIST].format(self._image_set)), \
        #     'List file \'{}\' not found'.format(DATASETS[self.name][DATA_LIST].format(self._image_set))

        assert os.path.exists(DATASETS[self.name][PID_LIST].format(self._image_set)), \
            'List file \'{}\' not found'.format(DATASETS[self.name][PID_LIST].format(self._image_set))
        if cfg.TRAIN.NORM_DATA:
            self.image_directory = DATASETS[self.name][NORM_IM_DIR]
        else:
            self.image_directory = DATASETS[self.name][IM_DIR]
        if cfg.USE_MERGED_ANN:
            self.annotation_directory = DATASETS[self.name][ANN_FN_MERGE]
        else:
            self.annotation_directory = DATASETS[self.name][ANN_FN]
        if cfg.USE_DCM_BBOX:
            self.bbox_directory = DATASETS[self.name][BBOX_DIR_DCM]
        else:
            self.bbox_directory = DATASETS[self.name][BBOX_DIR]

        # self.id_list = DATASETS[self.name][DATA_LIST].format(self._image_set)
        self.id_list = DATASETS[self.name][PID_LIST].format(self._image_set)
        cats = {'mass': 1}
        self.classes = tuple(['__background__'] + [c for c in cats])
        self.num_classes = len(self.classes)

        self.debug_timer = Timer()
        # Set up dataset classes
        self.category_to_id_map = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.png'
        self._label_ext = '.json'
        if 'gansu' or 'nanjing' in self.name:
            self._label_ext = '.json'
        self._image_index = self._load_image_set_index()

        if cfg.MODEL.NUM_CLASSES != -1:
            assert cfg.MODEL.NUM_CLASSES == self.num_classes, \
                "number of classes should equal when using multiple datasets"
        else:
            cfg.MODEL.NUM_CLASSES = self.num_classes
        # Added by shuzhang
        self.keypoints = None

    def _load_image_set_index(self):
        """
        Load image paths.
        """
        image_set_file = self.id_list
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        image_index = []
        with open(image_set_file) as f:
            patient_index = [x.strip() for x in f.readlines()]
            for pat in patient_index:
                if self.name != 'inbreast':
                    image_index.extend(glob(self.image_directory + pat + '*.png'))
                else:
                    image_index.extend(glob(self.image_directory + '*_' + pat + '*.png'))

        image_index = [index.split('/')[-1].split('.png')[0] for index in image_index]
        valid_index = []
        for index in image_index:
            if self.label_path_from_index(index):
                valid_index.append(index)
        print("Sample Num: ", len(valid_index))
        return valid_index

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['width', 'height', 'bbox', 'b_bbox', 'b_image', 'boxes', 'segms', 'gt_classes', 'seg_areas', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map']
        return keys

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self.image_directory,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path
    def label_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        label_path = os.path.join(self.annotation_directory,
                                  index + self._label_ext)
        if os.path.exists(label_path):
            return [label_path]
        else:
            label_paths = glob(os.path.join(self.annotation_directory,
                                      index +'*'+ self._label_ext))
            bbox_paths = glob(os.path.join(self.bbox_directory,
                                      index +'*'+ '.txt'))
            if len(label_paths) == 0 or len(bbox_paths)==0:
                return None
            else:
                return label_paths

    def get_roidb(
            self,
            gt=False,
            proposal_file='',
            crowd_filter_thresh=0
        ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self._image_index
        image_ids.sort()
        if cfg.DEBUG:
            image_ids = image_ids[:10]
        roidb = self._prep_roidb_entry(image_ids)

        if gt:
            # Include ground-truth object annotations
            cache_filepath = os.path.join(self.cache_path, self.name+'_' + self._image_set +'_gt_roidb.pkl')
            logger.debug(
                'cache_filepath {}'.
                    format(cache_filepath)
            )

            if os.path.exists(cache_filepath):
                self.debug_timer.tic()
                # with open(cache_filepath, 'rb') as fid:
                #     roidb = pickle.load(fid)
                self._add_gt_from_cache(roidb, cache_filepath)

                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                # roidb = [self._load_mammo_annotation(index)
                #          for index in image_ids]
                for entry in roidb:
                    self._load_mammo_annotation(entry)
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                if not cfg.DEBUG:
                    with open(cache_filepath, 'wb') as fp:
                        pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', cache_filepath)
        _add_class_assignments(roidb)
        return roidb


    def _prep_roidb_entry(self, image_ids):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset

        roidb = []
        for index in image_ids:
            entry = {}
            im_path = self.image_path_from_index(index)

            assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)

            entry['image'] = im_path
            entry['file_name'] = index
            # import SimpleITK as sitk
            # import os
            # dicom_dir = '/data2/mammogram/gansu_b1_1k/dicom/'
            # dicom_path = os.path.join(dicom_dir, pid, studyid, seriesid, '_'.join([side, view, instanceid])) + '.dcm'
            # image_ = sitk.ReadImage(dicom_path.encode('utf-8'))

            # if other_side
            # entry['other_bbox'] = (bbox_y1, bbox_x1, bbox_y2, bbox_x2)
            # entry['other_image'] = other_im_path
            entry['height'] = 0#image.shape[0]
            entry['width'] = 0#image.shape[1]
            entry['bbox'] = 0
            entry['b_bbox'] = 0
            entry['b_image'] = ''
            entry['dataset'] = self

            # Make file_name an abs path
            entry['flipped'] = False
            # Empty placeholders
            entry['boxes'] = np.empty((0, 4), dtype=np.float32)
            entry['segms'] = []
            entry['gt_classes'] = np.empty((0), dtype=np.int32)
            entry['seg_areas'] = np.empty((0), dtype=np.float32)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(
                np.empty((0, self.num_classes), dtype=np.float32)
            )
            entry['is_crowd'] = np.empty((0), dtype=np.bool)
            # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
            # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
            entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
            roidb.append(entry)
        return roidb

    def _load_mammo_annotation(self, entry):
        index = entry['file_name']
        bbox_data = read_json(os.path.join(self.bbox_directory, index+'.txt'))
        bbox_y1, bbox_x1, bbox_y2, bbox_x2 = bbox_data['a_bbox']
        entry['b_bbox'] = bbox_data['b_bbox']
        entry['b_image'] = os.path.join(self.image_directory, bbox_data['b_index']+self._image_ext)
        pad = cfg.TRAIN.PADDING
        bbox_y1, bbox_x1, bbox_y2, bbox_x2 = bbox_y1-pad, bbox_x1-pad, bbox_y2+pad, bbox_x2+pad
        entry['bbox'] = (bbox_y1, bbox_x1, bbox_y2, bbox_x2)
        entry['height'] = bbox_y2 - bbox_y1 # image.shape[0]
        entry['width'] = bbox_x2 - bbox_x1 # image.shape[1]

        boxes = []
        valid_segms = []
        gt_classes = []
        gt_overlaps = []
        seg_areas = []
        box_to_gt_ind_map = []
        is_crowd = []

        cls = 1
        id = 0

        json_paths = self.label_path_from_index(index)
        for json_path in json_paths:
            data = read_json(json_path)
            for ix, nodule in enumerate(data['nodes']):
                if nodule['type'].lower() == 'mass':
                    points = nodule['rois'][0]['edge']
                    if len(points) > 3:
                        #if isinstance(points[0][0], unicode):
                        #    flat_list = [max(0, int(float(item))) for sublist in points for item in sublist]
                        #else:
                        flat_list = [max(0, int(item)) for sublist in points for item in sublist]


                        label_array = np.array(flat_list, dtype=np.int32).reshape((-1, 2))
                        label_array[:, 0] -= bbox_x1  # x
                        label_array[:, 1] -= bbox_y1  # y
                        label_array[:, 0][label_array[:, 0] > entry['width']] = entry['width'] - 1
                        label_array[:, 1][label_array[:, 1] > entry['height']] = entry['height'] - 1
                        label_array[label_array < 0] = 0

                        x1, y1, w, h = cv2.boundingRect(label_array[:, np.newaxis, :])
                        x2 = x1 + w - 1
                        y2 = y1 + h - 1
                        seg_area = (w + 1) * (h + 1)
                        boxes.append([x1, y1, x2, y2])
                        seg_areas.append(seg_area)
                        flat_list = [item for sublist in label_array for item in sublist]
                        valid_segms.extend([[flat_list]])
                        gt_classes.append(cls)
                        gt_overlaps.append([0, 1.0])
                        box_to_gt_ind_map.append(id)
                        is_crowd.append(False)
                        id += 1
                elif cfg.MULTIPLE_CLASS_FG and ('astmmetry' in nodule['type'].lower() or 'asymmetry' in nodule['type'].lower() or 'distortion' in nodule['type'].lower()):
                    points = nodule['rois'][0]['edge']
                    if len(points) > 3:
                        #if isinstance(points[0][0], unicode):
                        #    flat_list = [max(0, int(float(item))) for sublist in points for item in sublist]
                        #else:
                        flat_list = [max(0, int(item)) for sublist in points for item in sublist]


                        label_array = np.array(flat_list, dtype=np.int32).reshape((-1, 2))
                        label_array[:, 0] -= bbox_x1  # x
                        label_array[:, 1] -= bbox_y1  # y
                        label_array[:, 0][label_array[:, 0] > bbox_x2 - bbox_x1] = entry['width'] - 1
                        label_array[:, 1][label_array[:, 1] > bbox_y2 - bbox_y1] = entry['height'] - 1
                        label_array[label_array < 0] = 0

                        x1, y1, w, h = cv2.boundingRect(label_array[:, np.newaxis, :])
                        x2 = x1 + w - 1
                        y2 = y1 + h - 1
                        seg_area = (w + 1) * (h + 1)
                        boxes.append([x1, y1, x2, y2])
                        seg_areas.append(seg_area)
                        flat_list = [item for sublist in label_array for item in sublist]
                        valid_segms.extend([[flat_list]])
                        gt_classes.append(cls)
                        gt_overlaps.append([-1.0, -1.0])
                        box_to_gt_ind_map.append(id)
                        if cfg.TRAIN.IGNORE_ON:
                            is_crowd.append(True)
                        else:
                            is_crowd.append(False)
                        id += 1

        if len(boxes) == 0:
            boxes.append([0,0,0,0])
            gt_classes.append(1)
            gt_overlaps.append([0, 1.0])
            box_to_gt_ind_map.append(id)
            seg_areas.append(0)
            is_crowd.append(False)

        entry['boxes'] = np.append(entry['boxes'], np.array(boxes, dtype=np.float32), axis=0)
        entry['segms'].extend(valid_segms)
        entry['gt_classes'] = np.append(entry['gt_classes'], np.array(gt_classes, dtype=np.int32))
        entry['seg_areas'] = np.append(entry['seg_areas'], np.array(seg_areas, dtype=np.float32))
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), np.array(gt_overlaps, dtype=np.float32), axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], np.array(is_crowd, dtype=np.bool))
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], np.array(box_to_gt_ind_map, dtype=np.int32)
        )

    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        print(len(roidb), len(cached_roidb))

        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            width, height, bbox, b_bbox, b_image, boxes, segms, gt_classes, seg_areas, gt_overlaps, is_crowd, \
                box_to_gt_ind_map = values[:12]
            entry['height'] = height
            entry['width'] = width
            entry['bbox'] = bbox
            entry['b_bbox'] = b_bbox
            entry['b_image'] = b_image
            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'].extend(segms)
            # To match the original implementation:
            # entry['boxes'] = np.append(
            #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )

def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


# save max-ious into gt_overlaps
def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )

# mark gt_overlaps to -1 for anchors overlap crowd areas over crowd_thresh
# gt_overlaps is a N*C matrix, originally set to 0
def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = mask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)

# summary max of gt_overlaps to vector max_overlaps and vector max_classes
def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]

if __name__ == '__main__':
    dataset = 'gansu'
    mammo_dataset = MammoDataset(dataset + '_valid')
    roidb = mammo_dataset.get_roidb(
        gt=True,
        proposal_file='',
        crowd_filter_thresh=0
    )


