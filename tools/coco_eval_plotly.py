
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import argparse
import os
import logging
import cv2
try:
    import nori2 as nori
except ImportError:
    raise ImportError('Please install nori2')

import plotly.express as px
import plotly.offline as of
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import defaultdict
of.offline.init_notebook_mode(connected=True)

def drwa_prs(coco_res: COCOeval, save_path = './test_new.html', use_line3d = False):
    #fig = go.Figure()
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{"type": "Scatter"}, {"type": "Scatter"}]])

    aind = 0
    mind = -1
    s = coco_res.eval['precision']
    scores = coco_res.eval['scores']

    thres_num = s.shape[0]
    category_num = s.shape[2]
    rec_thres = coco_res.params.recThrs

    r""" Draw all pr curve """ 
    for category_idx, category in enumerate(coco_res.params.catIds):
        name = coco_res.cocoDt.cats[category]['name']
        r""" Each Category and Each iou threshold PR curve and Score """
        for thres_idx, iou_thres in enumerate(coco_res.params.iouThrs):
            y_pre = s[thres_idx, :, category_idx, aind, mind]
            y_score = scores[thres_idx, :, category_idx, aind, mind]
            fig.add_trace(
                go.Scatter(x=rec_thres, y=y_pre, name=name + ":" + "iou_{:.2f}".format(iou_thres), mode='lines'),
                row=1, col=1)
            fig.add_trace(
                go.Scatter(x=rec_thres, y=y_score, name=name + ":" + "score_{:.2f}".format(iou_thres), mode='lines'),
                row=1, col=2)

        r""" Each Category MAP"""
        prec =  s[:, :, category_idx, aind, mind]
        prec = np.mean(prec, axis = 0)
        cat_map = np.mean(prec[prec > -1])
        fig.add_trace(
            go.Scatter(x=rec_thres, y=prec, name=name + ":MAP_({:.3f})".format(cat_map), mode='lines'),
            row=1, col=1)

    r""" Summary plot """
    all_categories_names = [cat['name'] for _, cat in coco_res.cocoDt.cats.items()]
    all_vis = [False for _ in  range((thres_num * 2 + 1) * category_num)]
    for i in range(category_num):
        all_vis[i * (thres_num * 2 + 1) + 2 * thres_num] = True 
    button_all = dict(label = 'All',
                    method = 'update',
                    args = [{'visible':  all_vis,
                            'title': 'All',
                            'showlegend':True}])

    def create_layout_button(category_name):
        vis = [False for _ in  range((thres_num * 2 + 1) * category_num)]
        try:
            find_idx = all_categories_names.index(category_name)
            for i in range(find_idx * (thres_num * 2 + 1), (find_idx + 1) * (thres_num * 2 + 1)):
                vis[i] = True
        except:
            assert("This category is not here".format(category_name))
        return dict(label = category_name,
                    method = 'update',
                    args = [{'visible': vis,
                             'title': category_name,
                             'showlegend': True}])

    fig.update_layout(
        updatemenus=[go.layout.Updatemenu(
            active = 0,
            buttons = [button_all] + 
                        list(map(lambda column: create_layout_button(column), all_categories_names))
            )
     ])
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title='Recall',
        yaxis_title='Precsion'
    )

    fig.write_html(save_path) 

def draw_badcase(coco_res: COCOeval, top_k: int, fp_dict, fn_dict, fp_dir, fn_dir, img_root = ""):
    cats = coco_res.cocoGt.cats
    def drow_box(img_info:dict, anns):
        red = (0, 0, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255) #黄色，忽略框
        try:
            img = cv2.imread(os.path.join(img_root, img_info['file_name']))
            assert(img is None)
        except:
            nori_fetch = nori.Fetcher()
            img_bytes = nori_fetch.get(img_info['nori_id']) 
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        for idx in range(len(anns)):
            x1 = int(anns[idx]['bbox'][0])
            y1 = int(anns[idx]['bbox'][1])
            x2 = int(anns[idx]['bbox'][0] + anns[idx]['bbox'][2])
            y2 = int(anns[idx]['bbox'][1] + anns[idx]['bbox'][3])
            cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color = red, thickness=1)
            try:
                info = cats[anns[idx]['category_id']]['name'] +':{:.3}'.format(anns[idx]['score'])
            except:
                info = cats[anns[idx]['category_id']]['name'] 
            cv2.putText(img, info, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, .6, red, 1, 2)
        return img 

    r"""1. Draw fp """
    for key_idx, key in enumerate(list(fp_dict)[0:top_k]):
        (imgid, catid), anns = key, fp_dict[key]
        img_info = coco_res.cocoDt.imgs[imgid]
        img = drow_box(img_info, anns)
        save_path = os.path.join(fp_dir, cats[catid]['name'], format(key_idx, '06d') + "_" + img_info['file_name'])
        cv2.imwrite(save_path, img)
    
    r"""2. Draw fn """
    for key_idx, key in enumerate(list(fn_dict)[0:top_k]):
        (imgid, catid), anns = key, fn_dict[key]
        img_info = coco_res.cocoGt.imgs[imgid]
        img = drow_box(img_info, anns)
        save_path = os.path.join(fn_dir, cats[catid]['name'], format(key_idx, '06d') + "_" + img_info['file_name'])
        cv2.imwrite(save_path, img)

def save_badcase(coco_res: COCOeval, 
                 iou_thres = 0.5, 
                 score_thres = 0.3,
                 top_k = -1, 
                 save_path = './',
                 img_root = None):

    r""" 1. Make dirs 
        fp: false positive, two parts: iou <= iou_thres, rank by score
        fn: false negative, two parts: detection has no match, rank by size
    """
    def make_dir(path_dir:str):
        if not os.path.exists(path_dir):
            try:
                os.makedirs(path_dir, exist_ok = True)
            except:
                print("No access to make dir")
        return path_dir
    fp_dir = make_dir(os.path.join(save_path, "badcase/fp"))
    fn_dir = make_dir(os.path.join(save_path, 'badcase/fn'))
    for badcase_dir in [fp_dir, fn_dir]:
        for _, cat in coco_res.cocoDt.cats.items():
            make_dir(os.path.join(badcase_dir, cat['name']))

    r""" 2. Save badcase """
    p = coco_res.params
    catIds = p.catIds if p.useCats else [-1]
    setK = set(catIds)
    setA = set(map(tuple, p.areaRng))
    setI = set(p.imgIds)

    k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
    a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
    i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
    I0 = len(p.imgIds)
    A0 = len(p.areaRng)

    fp_dict = defaultdict(list)
    fn_dict = defaultdict(list)
    for k, k0 in enumerate(k_list):
        Nk = k0*A0*I0
        Na = 0
        for Ni in i_list:
            E = coco_res.evalImgs[Nk + Na + Ni]
            if E is None:
                continue
            dtScores = np.array(E['dtScores'])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind='mergesort')
            dtScoresSorted = dtScores[inds]

            iou_thres_idx = np.where(coco_res.params.iouThrs == iou_thres)[0]
            dtm  = np.concatenate(E['dtMatches'][iou_thres_idx])[inds]
            gtm  = np.concatenate(E['gtMatches'][iou_thres_idx])
            dtIg = np.concatenate(E['dtIgnore'])[inds]
            sorted_dtIds = np.array(E['dtIds'])[inds]
            gtIds = np.array(E['gtIds'])
            gtIg = E['gtIgnore']
            npig = np.count_nonzero(gtIg==0 )
            if npig == 0:
                continue
            tps = np.logical_and(               dtm,  np.logical_not(dtIg))
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
            for idx_fp, val in enumerate(fps):
                if val == True and \
                    coco_res.cocoDt.anns[sorted_dtIds[idx_fp]]['score'] <= score_thres: # low score fp
                    fps[idx_fp] = False
            if True in fps:
                for idx_fp, val in enumerate(fps):
                    if val == True:
                        fp_dict[E['image_id'], E['category_id']].append(coco_res.cocoDt.anns[sorted_dtIds[idx_fp]])

            if 0 in gtm:
                for idx_fn, val in enumerate(gtm):
                    if val == 0:
                        fn_dict[E['image_id'], E['category_id']].append(coco_res.cocoGt.anns[gtIds[idx_fn]])
    
    sort_fp_dict = {k: v for k, v in sorted(fp_dict.items(), key=lambda item: -item[1][0]['score'])}
    sort_fn_dict = {k: v for k, v in sorted(fn_dict.items(), key=lambda item: -item[1][0]['area'])} 
    draw_badcase(coco_res, top_k, sort_fp_dict, sort_fn_dict, fp_dir, fn_dir, img_root)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-json', type=str, default='./gt.json', help='detection json file')
    parser.add_argument('--pred-json', type=str, default='./dt.json', help='prediction json file')
    parser.add_argument('--save-dir', type=str, default=None, help='prediction json file')
 
    opt = parser.parse_args()
    logger = logging.getLogger(__name__)

    anno = COCO(opt.gt_json)  # init annotations api
    r""" Reading Detection JSON 
        Formation: [{"image_id": 42, 
                    "category_id": 18, 
                    "bbox": [258.15, 41.29, 348.26, 243.78], 
                    "score": 0.236}, ...]
    """
    if type(opt.pred_json) == str:
        with open(opt.pred_json) as f:
            det_anns = json.load(f)
    else:
        det_anns = pred_json

    r""" Replace nori id with image id """
    if len(det_anns) != 0 and not det_anns[0]['image_id'] in set(anno.imgs):
        nori_imgid_dict = {}
        new_det_anns = []
        for ele in anno.imgs:
            nori_imgid_dict[anno.imgs[ele]['nori_id']] = ele
        for ele in det_anns:
            ele['image_id'] = nori_imgid_dict[ele['image_id']] 

    pred = anno.loadRes(det_anns)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.params.iouThrs = np.arange(0.5, 0.95, 0.2)
    eval.params.recThrs = np.arange(0, 1, 0.2)
    eval.params.imgIds = eval.params.imgIds[0:1000]
    #eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
    eval.evaluate()
    eval.accumulate()
    logger.info(f'Hyperparameters {eval.summarize()}')
    
    if opt.save_dir == None:
        opt.save_dir = os.path.dirname(opt.pred_json)
    save_badcase(eval, save_path = opt.save_dir)
    drwa_prs(eval, save_path = os.path.join(opt.save_dir, 'coco_eval.html'))