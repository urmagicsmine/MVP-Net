"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'gansu1':
        cfg.TEST.DATASETS = ('gansu1_valid',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'test':
        cfg.TEST.DATASETS = ('gansu1_test',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'by31':
        cfg.TEST.DATASETS = ('by31_valid',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'by31_train':
        cfg.TEST.DATASETS = ('by31_train',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'gansu2':
        cfg.TEST.DATASETS = ('gansu2_valid',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'gansu3':
        cfg.TEST.DATASETS = ('gansu3_valid',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'nanjing1':
        cfg.TEST.DATASETS = ('nanjing1_valid',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'data5':
        cfg.TEST.DATASETS = ('gansu1_train', 'gansu2_train', 'gansu3_train', 'nanjing1_train', #'by31_train',
                             'gansu1_valid', 'gansu2_valid', 'gansu3_valid', 'nanjing1_valid', #'by31_valid',
                             'gansu1_test', 'gansu2_test', 'gansu3_test', 'nanjing1_test', 'by31_test', 'by31_valid','by31_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'data10':
        cfg.TEST.DATASETS =( 
                             'gansu1_valid', 'gansu2_valid', 'gansu3_valid', 'nanjing1_valid', 'nanjing2_valid', 'zhongri1_valid', 'shougang1_valid',
                             'gansu1_test', 'gansu2_test', 'gansu3_test', 'nanjing1_test', 'nanjing2_test', 'zhongri1_test', 'shougang1_test',
                             'gansu1_train', 'gansu2_train', 'gansu3_train', 'nanjing1_train', 'nanjing2_train', 'zhongri1_train', 'shougang1_train',
                             'gansu4_valid', 'gansu4_test', 'gansu4_train', 'hologicmix1_valid', 'hologicmix1_test', 'hologicmix1_train',
                             'by31_test', 'by31_valid','by31_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'dataall':
        cfg.TEST.DATASETS = (
                             'gansu1_valid', 'gansu2_valid', 'gansu3_valid', 'nanjing1_valid', 'nanjing2_valid', 'zhongri1_valid', 'shougang1_valid',
                             'gansu1_test', 'gansu2_test', 'gansu3_test', 'nanjing1_test', 'nanjing2_test', 'zhongri1_test', 'shougang1_test',
                             'gansu1_train', 'gansu2_train', 'gansu3_train', 'nanjing1_train', 'nanjing2_train', 'zhongri1_train', 'shougang1_train',
                             'by31_test', 'by31_valid','by31_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'gansu1_all':
        cfg.TEST.DATASETS = ('gansu1_train', 'gansu1_valid', 'gansu1_test')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'gansu2_all':
        cfg.TEST.DATASETS = ('gansu2_valid', 'gansu2_test', 'gansu2_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'gansu3_all':
        cfg.TEST.DATASETS = ('gansu3_valid', 'gansu3_test', 'gansu3_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'nanjing1_all':
        cfg.TEST.DATASETS = ('nanjing1_valid', 'nanjing1_test', 'nanjing1_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'by31_all':
        cfg.TEST.DATASETS = ('by31_valid', 'by31_test', 'by31_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'gansu_all':
        cfg.TEST.DATASETS = ('gansu1_valid', 'gansu2_valid', 'gansu3_valid', 'nanjing1_valid', 'by31_valid')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'data8':
        cfg.TEST.DATASETS = ('gansu1_valid', 'gansu2_valid', 'gansu3_valid', 'nanjing1_valid', 'nanjing2_valid', 'by31_valid')#, 'zhongri1_valid', 'shougang1_valid', 'gansu2filter_valid', 'by31filter_valid')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'train8':
        cfg.TEST.DATASETS = ('gansu2filter_valid', 'by31filter_valid', 'gansu1_valid', 'gansu2_valid', 'gansu3_valid', 'nanjing1_valid', 'nanjing2_valid', 'by31_valid', 'zhongri1_valid', 'shougang1_valid', 'gansu1_train','by31_train')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'test':
        cfg.TEST.DATASETS = ('gansu1_test',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lesion_test':
        cfg.TEST.DATASETS = ('lesion_test',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lesion_test_175_275':
        cfg.TEST.DATASETS = ('lesion_test_175_275',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lesion_test_augment':
        cfg.TEST.DATASETS = ('lesion_test_augment',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lesion_test_other':
        cfg.TEST.DATASETS = ('lesion_test_other',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lesion_valid':
        cfg.TEST.DATASETS = ('lesion_valid',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lesion_train':
        cfg.TEST.DATASETS = ('lesion_train',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lb1031_test':
        cfg.TEST.DATASETS = ('lungbinary_1031_test', 'lungbinary_1031_valid')
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == 'lt1031_test':
        cfg.TEST.DATASETS = ('lungtop3_1031_test', 'lungtop3_1031_valid')
        cfg.MODEL.NUM_CLASSES = 4
    elif args.dataset == 'ls1031_test':
        cfg.TEST.DATASETS = ('lungsub18_1031_test', 'lungsub18_1031_valid')
        cfg.MODEL.NUM_CLASSES = 19
    elif args.dataset == 'lsu1031_test':
        cfg.TEST.DATASETS = ('lungsuper8_1031_test', 'lungsuper8_1031_valid')
        cfg.MODEL.NUM_CLASSES = 9
    elif args.dataset == 'lsu1031_train':
        cfg.TEST.DATASETS = ('lungsuper8_1031_train',)
        cfg.MODEL.NUM_CLASSES = 9
    elif args.dataset == 'lb1031_train':
        cfg.TEST.DATASETS = ('lungbinary_1031_train',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)
