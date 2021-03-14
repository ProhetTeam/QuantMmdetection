import json
import mmcv
import argparse
from refile import smart_open, smart_path_join, smart_listdir
import boto3
import nori2 as nori
import os
import sys
from tqdm import tqdm


def ExtractAnn(orig_coco, ds_name):
    local = '/tmp/extracted_anns/'
    if not os.path.exists(local):
        os.makedirs(local)
    new_file={}
    file_list = []
    valid_ind = []
    for i in orig_coco['images']:
        new_file[i['id']]={'images':i}
    for i in orig_coco['annotations']:
        if i['image_id'] not in new_file:
            continue
        new_file[i['image_id']][i['id']] = i 
    for i in new_file.values():
        i_id=i['images']['id']
        filepath=local+f'{i_id}'+'.json'
        file_list.append(filepath)
        if len(i) != 1:
            valid_ind.append(i_id)
        with open(filepath,'w+') as dump_file:
            json.dump(i, dump_file)            
        if (i['images']['id']+1) % 10000 == 0:
            show=i['images']['id']+1
            print(f'No. {show} img+anno dumped')
    map_file=local+ds_name+'map.json'
    map_path=os.path.split(map_file)[0]
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    with open(map_file,'w+') as nori_list:
        nori_list.write(str(file_list))
    return [valid_ind,file_list]
    
def callback(nid, e):
    if e != None:
        print(e)
        return
    res.append(nid) 

def UploadAnnToNori(nori_path, ds_name):
    local = '/tmp/extracted_anns/'
    map_file = local+ds_name+'.json'
    res = []
    if 's3' in nori_path:
        nw = nori.remotewriteopen(nori_path+'/'+ds_name+'.nori')
    else:
        nw = nori.open(nori_path+'/'+ds_name+'.nori', 'w')
    map_dict = {}
    
    files=smart_listdir(local)
    for file in tqdm(files):
        if file.endswith('.json'):
            imgpath = smart_path_join(local, file)
            filedata = smart_open(imgpath, "rb").read()
            map_dict[file] = nw.async_put(callback,filedata,filename=file)
    nw.close()
    nw.join()
    map_path=os.path.split(map_file)[0]
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    json.dump(map_dict, open(map_file, "w"))
    return map_dict

def GenerateNoriIndex(nori_path, ds_name):
    nori_file=smart_path_join(nori_path+'/'+ds_name+'.nori/')
    reader=nori.NoriReader(nori_file)
    counter = 0
    index = {}
    for i in reader.scan():
        index[i[2]['filename']]=i[0]
        counter += 1
        if counter == 1:
            print(i[0],i[2]['filename'])
        if counter % 1000 == 0:
            print (counter,'index build')
    nori_map_file = '/data/data_map/map2/'+ds_name+'.json'
    nori_map_path=os.path.split(nori_map_file)[0]
    if not os.path.exists(nori_map_path):
        os.makedirs(nori_map_path)
    with open('/data/data_map/map2/'+ds_name+'.json','w+') as file:
        json.dump(index,file)
    cmd = 'nori speedup '+nori_file+' --on --replica=2'
    print(cmd)
    os.system(cmd)
    return index

def CocoToExtAnn(orig_coco, out_file, nori_path, index_path, ds_name):
    new_json = {}
    new_json["images"] = []
    new_json["annotations"] = []
    new_json['categories'] = orig_coco['categories']
    ext = ExtractAnn(orig_coco, ds_name)
    new_json['valid_inds'] = ext[0]
    map_dict = UploadAnnToNori(nori_path, ds_name)    
    index = GenerateNoriIndex(nori_path, ds_name)
    local = '/tmp/extracted_anns/'
    img_index, ann_index, catToImgs = {}, {}, {}
    for i in orig_coco['images']:
        ann =f'{i["id"]}' + '.json'
        if ann not in index:
            continue
        img_index[i['id']] = index[ann]
    for i in orig_coco['annotations']:
        ann =f'{i["image_id"]}' + '.json'
        if ann not in index:
            continue
        ann_index[i['id']] = index[ann]
        if i['category_id'] not in catToImgs:
            catToImgs[i['category_id']]=[]
        catToImgs[i['category_id']].append(i['image_id'])
    img_index_path = smart_path_join( index_path, ds_name, 'img_index.json')
    ann_index_path = smart_path_join( index_path, ds_name, 'ann_index.json')
    catToImgs_path = smart_path_join( index_path, ds_name, 'catToImgs.json')
    with smart_open(img_index_path,'w') as data:
        data.write(json.dumps(img_index))
    with smart_open(ann_index_path,'w') as data:
        data.write(json.dumps(ann_index))
    with smart_open(catToImgs_path,'w') as data:
        data.write(json.dumps(catToImgs))
    '''
    new_json['img_index'] = img_index_path
    new_json['ann_index'] = ann_index_path
    new_json['catToImgs'] = catToImgs_path
    '''
    new_json['img_index'] = img_index
    new_json['ann_index'] = ann_index
    new_json['catToImgs'] = catToImgs

    new_json['ann_path'] = 'nori://'
    with smart_open(index_path+ds_name+'/extann.json','w') as data:
        data.write(json.dumps(new_json))
    if len(out_file) != 0:
        with smart_open(out_file,'w') as data:
            data.write(json.dumps(new_json))
    print('Convert complete.')
    files=smart_listdir(local)
    for file in tqdm(files):
        if file.endswith('.json'):
            os.remove(local+file)

def main():
    with smart_open(args.orig_coco) as file:
        orig_coco = json.load(file)
    CocoToExtAnn(orig_coco,args.out_index,args.out_anns_path,args.index_path,args.ds_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_coco", default='/home/wangningzi/waymo-od/data/waymo_train_full_nori.json',
                        help="the input coco json file, e.g. /data/instances_train2017_nori.json")
    parser.add_argument("--out_index", default='',
                        help="the local output json file, e.g. /data/dataset/ext/coco_detection_train2017_nori.json")
    parser.add_argument("--out_anns_path", default='s3://wangnz-testbmk/extracted_anns',
                        help="the extracted anns nori on oss, e.g. s3://public/")
    parser.add_argument("--index_path", default='s3://wangnz-testbmk/dataset_index/',
                        help="the extracted anns index on oss, e.g. s3://public/")
    parser.add_argument("--ds_name", default='waymo/train_full',
                        help="the name of dataset, e.g. COCO2017/train")
    args = parser.parse_args()
    main()
