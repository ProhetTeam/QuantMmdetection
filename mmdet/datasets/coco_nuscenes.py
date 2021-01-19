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
class NuscenesDataset(CocoDataset):

    CLASSES = ('vehicle.truck', 'vehicle.construction', 'vehicle.motorcycle',
             'movable_object.barrier', 'vehicle.car', 'human.pedestrian.adult',
             'movable_object.trafficcone', 'human.pedestrian.construction_worker',
             'movable_object.debris', 'static_object.bicycle_rack', 'vehicle.trailer',
             'vehicle.bicycle', 'vehicle.bus.rigid', 'animal', 'human.pedestrian.stroller',
             'human.pedestrian.police_officer', 'movable_object.pushable_pullable',
             'human.pedestrian.personal_mobility', 'vehicle.bus.bendy',
             'vehicle.emergency.police', 'human.pedestrian.child',
             'vehicle.emergency.ambulance', 'vehicle.ego', 'human.pedestrian.wheelchair')