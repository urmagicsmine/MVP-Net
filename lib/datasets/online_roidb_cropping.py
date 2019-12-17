from core.config import cfg
import numpy as np
import scipy
import copy
import math

def online_cropping_roidb_v1(roidb):
    """
    modify the roidb inline.
    add key: 'cropped_bbox' [x1,y1,x2,y2] for cropped image
    two-step mam crop:
    step1: create a minimum bounding box to contain all the mam
           and make it at least with shape [Min_Width, Min_Height]
    step2: create a random bounding box range from current box to the full image
    """
    Min_Width = cfg.TRAIN.ONLINE_RANDOM_CROPPING_WIDTH_MIN
    Min_Height = cfg.TRAIN.ONLINE_RANDOM_CROPPING_HEIGHT_MIN
    prob = cfg.TRAIN.ONLINE_RANDOM_CROPPING_PROBABILITY
    len_statistic = []
    for entry in roidb:
        if entry['seg_areas'][0] == 0:
            continue
        # height, width, gt bbox and segms
        height, width = entry['height'], entry['width']
        boxes, segms = entry['boxes'], entry['segms']
        # boxes: array N*4 [x1, y1, x2, y2]
        # segms: list, 1*1*N, each is [x,y,x,y,x,y...]
         
        # crop the entry with a probability 0.5
        probability = np.random.rand()
        if probability > prob:
            entry['cropped_bbox'] = [0, 0, width-1, height-1]
            continue
        
        min_x, max_x = boxes[:,0].min(), boxes[:,2].max() 
        min_y, max_y = boxes[:,1].min(), boxes[:,3].max()
        
        mass_width = max_x - min_x + 1
        mass_height = max_y - min_y + 1
        # two step to augment the data
        # step1: generate the min box and enlarge it to the Min_Width and Min_Height
        if mass_width < Min_Width:
            min_x = math.max(0, min_x - (Min_Width - mass_width)/2)
            max_x = min(max_x + (Min_Width - mass_width)/2, width-1)
        if mass_height < Min_Height:
            min_y = max(0, min_y - (Min_Height - mass_height)/2)
            max_y = min(max_y + (Min_Height - mass_height)/2, height-1)
        # step2: random sample the image in the full image
        if min_x > 0:
            x_left_extend = np.random.randint(0, min_x+1, 1)[0]
        else:
            x_left_extend = 0
        if width-max_x > 0:
            x_right_extend = np.random.randint(0, width-max_x+1, 1)[0]
        else:
            x_right_extend = 0
        if min_y > 0:
            y_up_extend = np.random.randint(0, min_y+1, 1)[0]
        else:
            y_up_extend = 0
        if height-max_y > 0:
            y_down_extend = np.random.randint(0, height-max_y+1, 1)[0]
        else:
            y_down_extend = 0
        if x_left_extend * x_right_extend * y_up_extend * y_down_extend == 0:
            print(entry['image'])
            print([min_x, min_y, max_x, max_y, height, width])
        x1 = min_x - x_left_extend
        y1 = min_y - y_up_extend
        x2 = max_x + x_right_extend
        y2 = max_y + y_down_extend
        # update the annotations
        entry['height'] = y2 - y1 + 1
        entry['width'] = x2 - x1 + 1
        entry['boxes'][:, 0] = entry['boxes'][:, 0] - x1
        entry['boxes'][:, 2] = entry['boxes'][:, 2] - x1
        entry['boxes'][:, 1] = entry['boxes'][:, 1] - y1
        entry['boxes'][:, 3] = entry['boxes'][:, 3] - y1
        segms_new = []
        for seg in segms:
            seg = np.array(seg)
            seg[:, 0:-1:2] = seg[:, 0:-1:2] - x1
            seg[:, 1:-1:2] = seg[:, 1:-1:2] - y1
            seg = seg.tolist()
            segms_new.extend([seg])
        entry['segms'] = segms_new
        entry['cropped_bbox'] = [int(x1), int(y1), int(x2), int(y2)]
        # sumary 
        len_statistic.extend([x_left_extend, x_right_extend, y_up_extend, y_down_extend])
    # output the summary of len_statistic
    print(np.min(len_statistic), np.max(len_statistic), np.std(len_statistic))

def online_cropping_roidb_v2(roidb):
    """
    add key: 'cropped_bbox' [x1,y1,x2,y2] for cropped image
    two-step mam crop:
    step1: random generate a bounding box within the constricts area in the full image
    step2: for each mass, if it is larger than the max with and height, igore it 
           for each mass, random put it in the generate bounding box, then move the box to the full image
    """
    Min_Width = cfg.TRAIN.ONLINE_RANDOM_CROPPING_WIDTH_MIN
    Max_Width = cfg.TRAIN.ONLINE_RANDOM_CROPPING_WIDTH_MAX
    Min_Height = cfg.TRAIN.ONLINE_RANDOM_CROPPING_HEIGHT_MIN
    Max_Height = cfg.TRAIN.ONLINE_RANDOM_CROPPING_HEIGHT_MAX
    prob = cfg.TRAIN.ONLINE_RANDOM_CROPPING_PROBABILITY
    len_useless = 0
    ret_roidb = []
    for entry in roidb:
        if entry['seg_areas'][0] == 0:
            continue
        # height, width, gt bbox and segms
        height, width = entry['height'], entry['width']
        boxes, segms = entry['boxes'], entry['segms']
        # boxes: array N*4 [x1, y1, x2, y2]
        # segms: list, 1*1*N, each is [x,y,x,y,x,y...]
        # normalize the boundary pixel into [0, height-1] [0, width-1]
        
        # crop the entry with a probability 0.5
        probability = np.random.rand()
        if probability > prob:
            entry['cropped_bbox'] = [int(0), int(0), int(width-1), int(height-1)]
            continue

        # for each gt box, create a independent bounding box
        for box_gt, seg_gt in zip(boxes, segms):
            box_gt = box_gt.reshape((1, 4))
            min_x, max_x = box_gt[0, 0], box_gt[0, 2]
            min_y, max_y = box_gt[0, 1], box_gt[0, 3]
            # the height and width of a mass
            mass_width = max_x - min_x + 1
            mass_height = max_y - min_y + 1
            if mass_height >= Max_Height or mass_width >= Max_Width:
                print("ignore the mass with shape: (%d, %d)"%(int(mass_height), int(mass_width)))
                continue
            # generate a bounding box which could cover the mass
            if width <= Min_Width:
                box_cropped_width = width
            else:
                if max(mass_width, Min_Width) > min(width, Max_Width): # handle unusual samples
                    continue
                if max(mass_width, Min_Width) == min(width, Max_Width): # the mass width == width
                    box_cropped_width = min(width, Max_Width)
                else:
                    box_cropped_width = np.random.randint(max(mass_width, Min_Width), min(width, Max_Width))
            if height <= Min_Height:
                box_cropped_height = height
            else:
                if max(mass_height, Min_Height) > min(height, Max_Height): # handling unusual samples 
                    continue
                if max(mass_height, Min_Height) == min(height, Max_Height): # the mass height == height
                    box_cropped_height = min(height, Max_Height)
                else:
                    box_cropped_height = np.random.randint(max(mass_height, Min_Height), min(height, Max_Height))
            # put the mass into the cropped box
            if box_cropped_width - mass_width < 0: # handling some unusuall samples
                mass_shift_x = 0
            else:
                mass_shift_x = np.random.randint(0, box_cropped_width - mass_width + 1)
            if box_cropped_height - mass_height < 0: # handing some unusuall samples
                mass_shift_y = 0
            else:
                mass_shift_y = np.random.randint(0, box_cropped_height - mass_height + 1)
            # compute the true coordinate of the cropped box
            cropped_min_x = min_x - mass_shift_x
            cropped_min_y = min_y - mass_shift_y
            cropped_max_x = cropped_min_x + box_cropped_width - 1
            cropped_max_y = cropped_min_y + box_cropped_height - 1
            # move the cropped box to the full image if it exceeds the boundary
            if cropped_min_x < 0:
                cropped_max_x = cropped_max_x - cropped_min_x
                cropped_min_x = 0
            if cropped_min_y < 0:
                cropped_max_y = cropped_max_y - cropped_min_y
                cropped_min_y = 0
            if cropped_max_x > width - 1:
                cropped_min_x = cropped_min_x - (cropped_max_x - width)
                cropped_max_x = width - 1
            if cropped_max_y > height - 1:
                cropped_min_y = cropped_min_y - (cropped_max_y - height)
                cropped_max_y = height - 1
            # keep those masses that exists in the cropped bounding box 
            entry_new = copy.deepcopy(entry)
            entry_new['height'] = int(cropped_max_y - cropped_min_y + 1)
            entry_new['width'] = int(cropped_max_x - cropped_min_x + 1)
            entry_new['cropped_bbox'] = [int(cropped_min_x), int(cropped_min_y),
                                         int(cropped_max_x), int(cropped_max_y)]
            cropped_bbox = [cropped_min_x, cropped_min_y, cropped_max_x, cropped_max_y]
            iops = compute_intersect_over_gt(cropped_bbox, entry_new['boxes'])
            
            new_boxes = np.empty((0, 4), dtype=np.float32)
            new_segms = [] # new_valid_segms
            new_seg_areas = []
            new_gt_overlaps = None
            new_gt_classes = []
            new_is_crowd = []
            new_box_to_gt_ind_map = [] 
            new_max_overlaps = []
            new_max_classes = []
             
            for index, iop in enumerate(iops[:, 0]):
                if iop >= cfg.TRAIN.ONLINE_RANDOM_CROPPING_IOP_THRESHOLD:
                    # keep the record
                    x1, y1, x2, y2 = entry_new['boxes'][index, 0], entry_new['boxes'][index, 1], entry_new['boxes'][index, 2], entry_new['boxes'][index, 3]
                    x1 = x1 - cropped_min_x 
                    y1 = y1 - cropped_min_y
                    x2 = x2 - cropped_min_x
                    y2 = y2 - cropped_min_y
                    if x1 < 0:
                        x1 = 0
                    if x2 > entry_new['width'] - 1:
                        x2 = entry_new['width'] - 1
                    if y1 < 0:
                        y1 = 0
                    if y1 > entry_new['height'] - 1:
                        y1 = entry_new['height'] - 1
                    # new boxes
                    new_boxes = np.append(new_boxes, np.array([x1, y1, x2, y2]).reshape((1,4)), axis=0)
                    # the segmentaion 
                    seg = np.array(entry_new['segms'][index])
                    seg[:, 0:-1:2] = seg[:, 0:-1:2] - cropped_min_x
                    seg[:, 1:-1:2] = seg[:, 1:-1:2] - cropped_min_y
                    for k, v in enumerate(seg[0, :]):
                        if k % 4 == 0 or k % 4 == 2:
                            if v < 0:
                                seg[0, k] = 0
                            if v > entry_new['width'] - 1:
                                seg[0, k] = entry_new['width'] - 1
                        else:
                            if v < 0:
                                seg[0, k] = 0
                            if v > entry_new['height'] - 1:
                                seg[0, k] = entry_new['height'] - 1
                    # new segms_new
                    new_segms.extend([seg.astype(np.int32).tolist()])
                    # new seg_areas
                    new_seg_areas.append((x2-x1+1)*(y2-y1+1))
                    # new gt_classes
                    new_gt_classes.append(entry_new['gt_classes'][index])
                    # new max_overlaps
                    new_max_overlaps.append(entry_new['max_overlaps'][index])
                    # new max_classes
                    new_max_classes.append(entry_new['max_classes'][index])
                    # new gt_overlaps
                    _, C = entry_new['gt_overlaps'].shape
                    if new_gt_overlaps is None:
                        new_gt_overlaps = entry_new['gt_overlaps'].toarray()[index, :].reshape(1,C)
                    else:
                        new_gt_overlaps = np.append(
                            new_gt_overlaps, entry_new['gt_overlaps'].toarray()[index, :].reshape(1,C), axis=0)
                    # new is_crowd:
                    new_is_crowd.append(entry_new['is_crowd'][index]) 
                    # new box_to_gt_ind_map
                    #if new_box_to_gt_ind_map is None:
                    #    new_box_to_gt_ind_map = np.array(entry_new['box_to_gt_ind_map'][index], dtype=np.int32)
                    #else:
                    #    new_box_to_gt_ind_map = np.append(
                    #        new_box_to_gt_ind_map, np.array(entry_new['box_to_gt_ind_map'][index], dtype=np.int32))
            entry_new['segms'] = new_segms
            entry_new['boxes'] = new_boxes.astype(np.float32)
            entry_new['gt_classes'] = np.array(new_gt_classes, dtype=np.int32)
            entry_new['gt_overlaps'] = scipy.sparse.csr_matrix(new_gt_overlaps, dtype=np.float32)
            entry_new['is_crowd'] = np.array(new_is_crowd, dtype=np.bool)
            # entry_new['box_to_gt_ind_map'] = new_box_to_gt_ind_map
            entry_new['box_to_gt_ind_map'] = np.array(range(len(new_gt_classes)), dtype=np.int32)
            entry_new['max_overlaps'] = np.array(new_max_overlaps)
            entry_new['max_classes'] = np.array(new_max_classes)
            if new_boxes.shape[0] == 0:
                len_useless = len_useless + 1
                continue
            ret_roidb.append(entry_new)
    print('missing masses: %d'%(len_useless))
    return ret_roidb

def compute_intersect_over_gt(box1, box2):
    """
    Inputs:
        box1: cropped_box
        box2: gt mass boxes
    Outpus:
        iops
    """
    ixmin = np.maximum(box1[0], box2[:, 0])
    iymin = np.maximum(box1[1], box2[:, 1])
    ixmax = np.minimum(box1[2], box2[:, 2])
    iymax = np.minimum(box1[3], box2[:, 3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    iops = inters / ((box2[:, 2] - box2[:, 0] + 1.0) * (box2[:, 3] - box2[:, 1] + 1.0))
    return iops.reshape((-1,1))
