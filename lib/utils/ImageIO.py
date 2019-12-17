import numpy as np
import cv2
import os

import utils.blob as blob_utils
import utils.opencv_transforms as cv_transforms
import utils.boxes as boxes_utils

from core.config import cfg

def load_multislice_16bit_png(roidb):
    imname = roidb['image']
    slice_intv = roidb['slice_intv']
    windows = roidb['windows']

    # Load 16bit single channel image
    # return: 3D slices D,H,W 

    def get_slice_name(img_path, delta=0):
        if delta == 0:
            return img_path
        delta = int(delta)
        name_slice_list = img_path.split(os.sep)
        slice_idx = int(name_slice_list[-1][:-4])
        img_name = '%03d.png' % (slice_idx + delta)
        full_path = os.path.join('/', *name_slice_list[:-1], img_name)

        # if the slice is not in the dataset, use its neighboring slice
        while not os.path.exists(full_path):
            #print 'file not found:', img_name
            delta -= np.sign(delta)
            img_name = '%03d.png' % (slice_idx + delta)
            full_path = os.path.join('/', *name_slice_list[:-1], img_name)
            if delta == 0:
                break
        return full_path

    def _load_data(img_name, delta=0):
        img_name = get_slice_name(img_name, delta)
        if img_name not in data_cache.keys():
            data_cache[img_name] = cv2.imread(img_name, -1)
            if data_cache[img_name] is None:
                print('file reading error:', img_name, os.path.exists(img_name))
                assert not data_cache[img_name] == None
        return data_cache[img_name]

    data_cache = {}

    im_cur = cv2.imread(imname, -1)
    #TODO: get_mask()

    # For 3DCE Framework, needed slice num is M*3(M denotes NUM_IMAGES_3DCE in the paper)
    if cfg.LESION.USE_3DCE:
        num_slice = int(cfg.LESION.NUM_IMAGES_3DCE * 3)
    # When use multi-modality,use 3DCE backbone,while slice_num = 3.
    else:
        #num_slice = int(cfg.LESION.SLICE_NUM)
        num_slice = int(cfg.LESION.NUM_IMAGES_3DCE)

    if cfg.LESION.SLICE_INTERVAL == 0 or np.isnan(slice_intv) or slice_intv < 0:
        ims = [im_cur] * num_slice  # only use the central slice
    else:
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = float(cfg.LESION.SLICE_INTERVAL) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
            for p in range((num_slice-1)//2):
                im_prev = _load_data(imname, - rel_pos * (p + 1))
                im_next = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
            #when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
            if num_slice%2 == 0:
                im_next = _load_data(imname, rel_pos * (p + 2))
                ims = ims + [im_next]
        else:
            for p in range((num_slice-1)//2):
                intv1 = rel_pos*(p+1)
                slice1 = _load_data(imname, - np.ceil(intv1))
                slice2 = _load_data(imname, - np.floor(intv1))
                im_prev = a * slice1 + b * slice2  # linear interpolation

                slice1 = _load_data(imname, np.ceil(intv1))
                slice2 = _load_data(imname, np.floor(intv1))
                im_next = a * slice1 + b * slice2
                ims = [im_prev] + ims + [im_next]
            #when num_slice is even number,got len(ims) with num_slice-1. Add 1 slice.
            if num_slice%2 == 0:
                intv1 = rel_pos*(p+2)
                slice1 = _load_data(imname, np.ceil(intv1))
                slice2 = _load_data(imname, np.floor(intv1))
                im_next = a * slice1 + b * slice2
                ims = ims + [im_next]

    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    im = im.astype(np.float32, copy=False)-32768  # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit
    #im = multi_windowing(im[:,:,9:18])
    if cfg.LESION.USE_SPECIFIC_WINDOWS:
        im = windowing(im, windows)
    elif cfg.LESION.MULTI_MODALITY:
        im = multi_windowing(im[:,:,:])
    else:
        im = windowing(im, cfg.WINDOWING)
        #im = windowing(im, [-175,275])
    return im

def load_16bit_png(imname):
    # Load 16bit single channel image
    img = cv2.imread(imname, -1)
    if cfg.LESION.THREE_SLICES:
        slice_ind = int(imname.split(os.sep)[-1].split('.')[0])
        imname_prev = os.sep.join(
            imname.split(os.sep)[:-1]) + os.sep + '%03d.png' % \
            (max(slice_ind - cfg.LESION.SLICE_INTERVAL, 0))
        imname_next = os.sep.join(
            imname.split(os.sep)[:-1]) + os.sep + '%03d.png' % \
            (max(slice_ind + cfg.LESION.SLICE_INTERVAL, 0))
        if not os.path.exists(imname_prev):
            imname_prev = imname
        if not os.path.exists(imname_next):
            imname_next = imname
        img_prev = cv2.imread(imname_prev, -1)
        img_next = cv2.imread(imname_next, -1)
        ims = [img_prev, img, img_next]
    else:
        ims = [img] * 3
    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    im = im.astype(np.float32, copy=False)-32768  # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit
    im = windowing(im, cfg.WINDOWING)

    return im

def multi_windowing(im):
    windows = [[-174,274],[-1493,484],[-534,1425]]
    #h,w,c = im.shape
    #assert im.shape[2] == 9,'im.shape != 9.'
    #assert c%3==0,'channel cannot devided by 3'
    if cfg.LESION.NUM_IMAGES_3DCE == 2:
        im_win1 = windowing(im, windows[0])
        im_win2 = windowing(im, windows[1])
        im = np.concatenate((im_win1, im_win2),axis=2)
    else:
    #elif cfg.LESION.NUM_IMAGES_3DCE == 3:
        im_win1 = windowing(im, windows[0])
        im_win2 = windowing(im, windows[1])
        im_win3 = windowing(im, windows[2])
        im = np.concatenate((im_win1, im_win2, im_win3),axis=2)
    #elif cfg.LESION.NUM_IMAGES_3DCE == 9:
        #im_win1 = windowing(im, windows[0])
        #im_win2 = windowing(im, windows[1])
        #im_win3 = windowing(im, windows[2])
        #im = np.concatenate((im_win1, im_win2, im_win3),axis=2)
    return im

def windowing(im, win):
    # Scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1

def get_a_img(entry):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = entry['bbox']
    im = cv2.imread(entry['image'], 0)
    if cfg.HIST_EQ and (not cfg.HIST_EQ_SYM):
        if cfg.A_HIST_EQ:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            im = clahe.apply(im)
        else:
            im = cv2.equalizeHist(im)

    # new_img = np.zeros((entry['height'], entry['width']))
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
    return new_img.astype(np.uint8)

def pad_image(img, pad=cfg.TRAIN.PADDING):
    h, w = img.shape
    new_img = np.zeros((h+2*pad, w+2*pad))
    if pad == 0:
        new_img = img.copy()
    else:
        new_img[pad:-pad, pad:-pad] = img
    return new_img

def get_double_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    pad = cfg.TRAIN.PADDING
    for i in range(num_images):

        # Process A image
        im = get_a_img(roidb[i])
        other_im = cv2.imread(roidb[i]['b_image'])
        # Process B image
        h, w, _ = im.shape
        b_y1, b_x1, b_y2, b_x2 = roidb[i]['b_bbox']
        if cfg.TRAIN.AUG_LRV_BBOX:
            b_x1, b_y1, b_x2, b_y2 = boxes_utils.aug_align_box((b_x1,b_y1,b_x2,b_y2), (other_im.shape[1], other_im.shape[0]), pad=0)
        tmp = other_im[b_y1:b_y2, b_x1:b_x2, 0]
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
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            other_im = other_im[:, ::-1, :]

        # generate the cropped image
        if cfg.TRAIN.ONLINE_RANDOM_CROPPING:
            x1, y1, x2, y2 = roidb[i]['cropped_bbox']
            im = im[y1:y2+1, x1:x2+1, :]
            other_im = other_im[y1:y2+1, x1:x2+1, :]

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        if cfg.TRAIN.AUGMENTATION:
            transform_cv = cv_transforms.Compose([
                                                  cv_transforms.ColorJitter(brightness=0.5,
                                                                   contrast=0.25, gamma=0.5)])
        else:
            transform_cv = None
        # TODO: add augmentation
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE, transform_cv)
        other_im, other_im_scale = blob_utils.prep_im_for_blob(
            other_im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE, transform_cv)

        im_scales.append(im_scale[0])
        processed_ims.append(im[0])
        processed_ims.append(other_im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

def get_dicom_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):

        # Segment the breast area
        #y1,x1,y2,x2 = roidb[i]['bbox']

        im = cv2.imread(roidb[i]['image'])#[y1:y2, x1:x2] #(h, w)
        other_im = cv2.imread(roidb[i]['other_image'])
        # im = np.tile(image, (3, 1, 1))
        # im = np.transpose(im, (1, 2, 0))
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
            other_im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        transform_cv = cv_transforms.Compose([
                                              cv_transforms.ColorJitter(brightness=0.5,
                                                                        contrast=0.25)])
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, transform_cv, [target_size], cfg.TRAIN.MAX_SIZE)
        other_im, other_im_scale = blob_utils.prep_im_for_blob(
            other_im, cfg.PIXEL_MEANS, transform_cv, [target_size], cfg.TRAIN.MAX_SIZE)

        im_scales.append(im_scale[0])
        processed_ims.append(im[0])
        processed_ims.append(other_im[0])

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    return blob, im_scales

