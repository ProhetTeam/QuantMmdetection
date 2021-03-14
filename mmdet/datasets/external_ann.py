import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict
from pycocotools.coco import COCO,_isArrayLike
import time
import mmcv
import numpy as np
from .coco import CocoDataset
from .builder import DATASETS
from collections import defaultdict
from refile import smart_open
from .pipelines import Compose
import sys
import json
import os,psutil
import copy

class GetAnnsById:
    def __init__(self, ann_dict, ann_path='s3://waymo-extracted/ann_by_img/'):
        '''ann_dict: dict {annotation_id(str) : image_nori_id(str) or ann nori id}'''
        self.ann_dict={}
        self.ann_path=ann_path
        for key,value in ann_dict.items():
            self.ann_dict[eval(key)]=value
    
    def __getitem__(self,key):
        if self.ann_path == 'nori://':
            ann = json.load(smart_open(self.ann_path+self.ann_dict[key],'rb'))
        else:
            ann = json.load(smart_open(self.ann_path+self.ann_dict[key]+'.json'))
        return ann[f'{key}']

    def keys(self):
        return self.ann_dict.keys()

class GetAnnsByImg:
    def __init__(self, ann_dict, ann_path='s3://waymo-extracted/ann_by_img/'):
        '''ann_dict: dict {image_id(str) : image_nori_id(str) or ann nori id}'''
        self.ann_dict={}
        self.ann_path=ann_path
        for key,value in ann_dict.items():
            self.ann_dict[eval(key)]=value
    
    def __getitem__(self,key):
        if self.ann_path == 'nori://':
            ann = json.load(smart_open(self.ann_path+self.ann_dict[key],'rb'))
        else:
            ann = json.load(smart_open(self.ann_path+self.ann_dict[key]+'.json'))
        anns = []
        for annid,value in ann.items():
            if annid != 'images':
                anns.append(value)
        return anns
    
    def keys(self):        
        return self.ann_dict.keys()

class GetImgsInList:
    def __init__(self, img_dict, ann_path='s3://waymo-extracted/ann_by_img/',use_nori=True):
        self.img_dict = img_dict
        self.ann_path = ann_path
        self.use_nori = use_nori
        self.ids = list(self.img_dict.keys())
        self.iterpos = 0
        self.imgidtolistindex = {}
        for i,id in enumerate(self.ids):
            self.imgidtolistindex[f'{id}']=i

    def __iter__(self):
        return self    
    
    def __next__(self):
        if self.iterpos < len(self.ids):
            self.iterpos += 1
            return self.__getitem__(self.iterpos-1)
        else:
            raise StopIteration

    def getkey(self,imgid):
        return self.imgidtolistindex[f'{imgid}']

    def valid(self,v_ids):
        self.ids = [str(id) for id in v_ids]
        valid_inds = []
        for i in v_ids:
            valid_inds.append(self.getkey(i))
        self.imgidtolistindex = {}
        for i,id in enumerate(self.ids):
            self.imgidtolistindex[f'{id}']=i
        return valid_inds
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self,key):
        if self.ann_path == 'nori://':
            ann = json.load(smart_open(self.ann_path+self.img_dict[self.ids[key]],'rb'))
        else:
            ann = json.load(smart_open(self.ann_path+self.img_dict[self.ids[key]]+'.json'))
        img=ann['images']
        if not self.use_nori:
            img['filename']=img['file_name']
            return img
        else:
            img['filename']=img['nori_id']
            return img

class GetImgsById:
    def __init__(self, img_dict, ann_path='s3://waymo-extracted/ann_by_img/'):
        '''img_dict: dict {image_id(str) : image_nori_id(str) or ann nori id}'''
        self.img_dict={}
        self.ann_path=ann_path
        for key,value in img_dict.items():
            self.img_dict[eval(key)]=value
    
    def __getitem__(self,key):
        if self.ann_path == 'nori://':
            ann = json.load(smart_open(self.ann_path+self.img_dict[key],'rb'))
        else:
            ann = json.load(smart_open(self.ann_path+self.img_dict[key]+'.json'))
        return ann['images']
    
    def keys(self):
        return self.img_dict.keys()

class ExternalAnn(COCO):
    
    def __init__(self, annotation_file=None, ):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing
        annotations.
        :param annotation_file (str): location of annotation file

        edit annotation file as follows:
        {
            ['images']:[]
            ['annotations']:[]
            ['categories']:COCO['categories']
            ['img_index']:img_index_path (str) or img_index (dict)
            ['ann_index']:ann_index_path (str) or ann_index (dict)
            ['catToImgs']:catToImgs_path (str) or catToImgs (dict)
            ['ann_path']:ann_path_prefix (str)
            ['valid_inds']:[valid_image_ids] (list(int))
        } 
        img_index_path (str): path of index file or dict{f'{img_id}':ann_path(str)}
        ann_index_path (str): path of index file or dict{f'{annotation_id}':ann_path(str)}
        catToImgs_path (str): path of index file or dict{f'{category_id}':[img_ids]}
        (used json.dump & json.load so keys are str)
        ann_path_prefix (str): 'nori://' or folder path(str)
        [valid_image_ids] (list(int)): [imgIDs] for imgs that have annotation
        dict{f'{category_id}':[img_ids]}
        Extracted annotations：
        dict{
            'images':COCO['images'][i],
            f'{annotation_id}':COCO['annotations'][x] if COCO['annotations'][x]['image_id']=i
            }
        A convertion sample can be found at tools/convert_datasets/FromCocoToExtann.py 
        :return:
        """
        # load dataset
        self.dataset, self.cats, self.imgs = dict(), dict(
        ), dict()
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            assert type(
                dataset
            ) == dict, 'annotation file format {} not supported'.format(
                type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()
        self.img_ann_map = self.imgToAnns
    
    
    def createIndex(self):
        # create index

        print('creating index...')
        tic = time.time()
        cats = {}
                
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat
        
        if 'ann_path' in self.dataset:
            self.ann_path = self.dataset['ann_path']

        # create class members
        if isinstance(self.dataset['ann_index'],str):            
            with smart_open(self.dataset['ann_index']) as file:
                ann_nori_dict = json.load(file)
        elif isinstance(self.dataset['ann_index'],dict):
            ann_nori_dict = self.dataset['ann_index']

        if isinstance(self.dataset['img_index'],str):
            with smart_open(self.dataset['img_index']) as file:
                img_nori_dict = json.load(file)
        elif isinstance(self.dataset['img_index'],dict):
            img_nori_dict = self.dataset['img_index']
        
        if isinstance(self.dataset['catToImgs'],str):
            with smart_open(self.dataset['catToImgs']) as file:
                catToImgs = json.load(file)
        elif isinstance(self.dataset['catToImgs'],dict):
            catToImgs = self.dataset['catToImgs']

        if 'ann_path' in self.dataset:
            self.anns = GetAnnsById(ann_nori_dict, ann_path=self.ann_path)
            self.imgToAnns = GetAnnsByImg(img_nori_dict, ann_path=self.ann_path)
            self.imgs = GetImgsById(img_nori_dict, ann_path=self.ann_path)
            self.dataset['images'] = GetImgsInList(img_nori_dict, ann_path=self.ann_path)
        else:
            self.anns = GetAnnsById(ann_nori_dict)
            self.imgToAnns = GetAnnsByImg(img_nori_dict)
            self.imgs = GetImgsById(img_nori_dict)
            self.dataset['images'] = GetImgsInList(img_nori_dict)
        self.catToImgs = { eval(cat_id): catToImgs[cat_id] for cat_id in catToImgs.keys() }
        print('index created (t={:0.2f}s)'.format(time.time() - tic))
        print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3 , 'GB rss after create index')
        self.cats = cats
        self.cat_img_map = self.catToImgs

    def getAnnIds(self, imgIds=[],catIds=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that
        filter
        :param imgIds  (int array)     : get anns for given imgs

        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        if len(imgIds) == len(catIds) == 0:
            return list(self.anns.keys())
        else:
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds
                    if imgId in self.imgToAnns.keys()
                ]
                anns = list(itertools.chain.from_iterable(lists))
                if len(catIds) == 0:
                    return [ann['id'] for ann in anns]
                else:
                    return list(ann['id'] for ann in anns 
                    if self.anns[ann['id']]['category_id'] in catIds)
            else:
                return list(id for id in self.anns.keys() 
                if self.anns[id]['category_id'] in catIds)


    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

@DATASETS.register_module()
class ExternalCocoDataset(CocoDataset):
    CLASSES = ('UNKNOWN', 'VEHICLE', 'PEDESTRIAN','SIGN', 'CYCLIST')
    def __init__(self,
                 ann_file,
                 pipeline,
                 use_nori = True,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        self.use_nori = use_nori
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None
                    or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root,
                                              self.proposal_file)
        # load annotations (and proposals)
        self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            v_ids = self.coco.dataset['valid_inds']
            valid_inds=self.data_infos.valid(v_ids)
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)
        

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):
        """Load "images" and "categories" from COCO style annotation file.
        Annotations will get from files with image nori_id

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list-like-object[dict]: Annotation info from COCO api.
        """
        from .utils import smart_open_map
        
        self.coco = smart_open_map(ExternalAnn, ann_file)
        self.cat_ids = [cat['id'] for cat in self.coco.dataset['categories']]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = list(self.coco.imgs.keys())

        with smart_open(self.coco.dataset['img_index']) as file:
            img_nori_dict = json.load(file)
        if self.use_nori == False:
            if 'ann_path' in self.coco.dataset:
                data_infos = GetImgsInList(img_nori_dict,ann_path=self.coco.dataset['ann_path'],use_nori=False)
            else:
                data_infos = GetImgsInList(img_nori_dict,use_nori=False)
        else:
            if 'ann_path' in self.coco.dataset:
                data_infos = GetImgsInList(img_nori_dict,ann_path=self.coco.dataset['ann_path'])
            else:
                data_infos = GetImgsInList(img_nori_dict)

        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1
    
    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return [ann['category_id'] for ann in ann_info]
    
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
                        print('abnormal bbox in result,class id:', label, '\n', bboxes)
        return json_results

