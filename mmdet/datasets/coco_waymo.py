import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class WaymoDataset(CocoDataset):

    CLASSES = ('UNKNOWN', 'VEHICLE', 'PEDESTRIAN','SIGN', 'CYCLIST')

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                if label < len(self.cat_ids):
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data['image_id'] = img_id
                        data['bbox'] = self.xyxy2xywh(bboxes[i])
                        data['score'] = float(bboxes[i][4])
                        data['category_id'] = self.cat_ids[label]
                        json_results.append(data)
                else:
                    if len(bboxes)!=0:
                        print(label,bboxes)
        return json_results