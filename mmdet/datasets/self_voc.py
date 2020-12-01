import copy
import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class SELFVOC(CustomDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def load_annotations(self, ann_file):
        # load image list from file
        lines = mmcv.list_from_file(self.ann_file)

        data_infos = []

        for line in lines:
            annos = line.strip().split(" ")

            img_name, width, height, bboxes =\
                     annos[1], int(annos[2]), int(annos[3]), [int(each) for each in annos[4:]]
                     
            data_info = dict(filename=img_name, width=width, height=height)

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            for x in range(len(bboxes) // 5):
                gt_labels.append(bboxes[x*5])
                gt_bboxes.append(bboxes[x*5+1:x*5+5])
            
            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)
        return data_infos