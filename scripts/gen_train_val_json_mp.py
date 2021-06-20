import json
import os
import random
import time
from glob import glob
from itertools import groupby

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
import multiprocessing

_MAX_INSTANCE_PER_FRAME = 6

root_dir = '/'

train_file = os.path.join(root_dir, 'tcdata/PreRoundData/ImageSets/train.txt')
img_root_path = os.path.join(root_dir, 'tcdata/PreRoundData/JPEGImages')
ann_root_path = os.path.join(root_dir, 'tcdata/PreRoundData/Annotations')
save_train_file = os.path.join(root_dir, 'user_data/annotations/instances_train_sub.json')


def gen_info(train_name):
    # id = 0
    video_dir = os.path.join(img_root_path, train_name)
    ann_dir = os.path.join(ann_root_path, train_name)
    frames = list(sorted(glob(os.path.join(video_dir, '*.jpg'))))
    masks = list(sorted(glob(os.path.join(ann_dir, '*.png'))))
    if len(frames) != len(masks):
        # 2 videos lack of annotations of last frames and 6 videos have an additional mask.
        # print('unbalanced annotations!')
        frames_names = set(map(lambda x: x[:-4], os.listdir(video_dir)))
        masks_names = set(map(lambda x: x[:-4], os.listdir(ann_dir)))
        names = frames_names & masks_names
        frames = sorted(list(map(lambda x: os.path.join(video_dir, x + '.jpg'), names)))
        masks = sorted(list(map(lambda x: os.path.join(ann_dir, x + '.png'), names)))
    assert len(frames) == len(masks)
    length = len(frames)
    valid = 0  # current maximum persons in visiting video sequences
    video_ann = []
    for idx, (frame_path, mask_path) in enumerate(zip(frames, masks)):
        mask = fortranarrayImg(mask_path)
        h, w = mask.shape
        for i in range(1, _MAX_INSTANCE_PER_FRAME):
            submask = mask.copy()
            submask[mask != i] = 0
            submask[submask == i] = 1
            submask = np.asfortranarray(submask)
            if np.count_nonzero(submask) != 0:
                if i > valid:
                    # id += 1
                    valid = i
                    ann_sample = {'height': h, 'width': w, 'length': length, 'category_id': 1,
                                  'segmentations': [], 'bboxes': [], 'video_id': train_name, 'iscrowd': 0, 'id': 0,
                                  'areas': []}  # id must be resorted!
                    for j in range(idx):
                        ann_sample['segmentations'].append(None)
                        ann_sample['bboxes'].append(None)
                        ann_sample['areas'].append(None)
                    video_ann.append(ann_sample)
                rle = mask_utils.encode(submask)
                area = mask_utils.area(rle).tolist()
                bounding_box = mask_utils.toBbox(rle).tolist()
                # rle_dict = binary_mask_to_rle(submask)
                # compressed_rle = mask_utils.frPyObjects(rle_dict, rle_dict.get('size')[0], rle.get('size')[1])
                # dec = mask_utils.decode(compressed_rle)
                video_ann[i - 1]['segmentations'].append(binary_mask_to_rle(submask))
                video_ann[i - 1]['bboxes'].append(bounding_box)
                video_ann[i - 1]['areas'].append(area)
            else:
                if i > valid:
                    break
                video_ann[i - 1]['segmentations'].append(None)
                video_ann[i - 1]['bboxes'].append(None)
                video_ann[i - 1]['areas'].append(None)
    # convert to relative path
    save_file_names = list(map(lambda x: os.path.relpath(x, img_root_path), frames))
    video_info = {"height": h, "width": w, "length": length, "date_captured": '', "license": 1, "coco_url": "",
         "flickr_url": "", "id": train_name, "file_names": save_file_names}
    print(train_name, 'is ok')
    return video_info, video_ann


def fortranarrayImg(save_path):
    return np.asfortranarray(Image.open(save_path), dtype=np.uint8)


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

with open(train_file, 'r') as f:
    names = sorted(f.read().split('\n')[:-1])

# random.shuffle(train_names)
# mid = int(0.8 * len(train_names))
# train_names, val_names = sorted(train_names[:mid]), sorted(train_names[mid:])

js = dict()
js['licenses'] = ['yf&ly']
js['info'] = {'authors': 'yf&ly'}
js['videos'] = []
js['annotations'] = []
js['categories'] = [{"supercategory": "object", "id": 1, "name": "person"}]
id = 0
start = time.time()
with multiprocessing.Pool(96) as pool:
    res = list(pool.imap(gen_info, names))

# pre_id = 0
idx = 0
for video_info, video_ann in res:
    # assert video_info['id'] > pre_id
    # pre_id = video_info['id']
    js['videos'].append(video_info)
    for ann in video_ann:
        idx += 1
        ann['id'] = idx
    js['annotations'].extend(video_ann)

json.dump(js, open(save_train_file, 'w'), indent=4, sort_keys=False)
print("cost time: {} s".format(time.time()-start))

'''
train_names = []
val_names = []
for name in names:
    if int(name) % 5 == 0:
        val_names.append(name)
    else:
        train_names.append(name)

train_names, val_names = sorted(train_names), sorted(val_names)
print('training samples:', len(train_names))
print('validate samples:', len(val_names))

save_train_file = 'data/annotations/instances_train_sub.json'
save_val_file = 'data/annotations/instances_val_sub.json'
print('generating training samples:')
gen_json(train_names, save_train_file)
print('generating validate samples:')
gen_json(val_names, save_val_file)
print('perfect!')

"annotations": [{"height": 720, "width": 1280, "length": 1, "category_id": 6, "segmentations": 
[null, null, null, null, null, null, null, null, {"counts": [830570, 11, 680, 
40, 677, 43, 671, 52, 668, 54, 666, 54, 666,
 1, 46, 668, 5, 1, 45, 675, 45, 675, 45, 676, 44, 679, 40, 682, 37, 685, 35, 689, 26, 702, 16, 76624], 
 "size": [720, 1280]}, 
 {"counts": [815416, ,0, 680, 40, 680, 40, 682, 38, 683, 37, 685, 35, 687, 32, 690, 28, 709, 9, 83103], 
 "size": [720, 1280]}]
   "bboxes": [null, null, null, null, null, null, null, null, [1153.0, 372.0, 21.0, 54.0], [1132.0, 360.0, 33.0, 67.0], [1114.0, 352.0, 38.0, 65.0], [1096.0, 353.0, 49.0, 66.0], [1088.0, 347.0, 55.0, 63.0], [1080.0, 350.0, 60.0, 59.0], [1074.0, 349.0, 65.0, 55.0], [1070.0, 343.0, 77.0, 54.0], [1093.0, 342.0, 73.0, 48.0], [1102.0, 328.0, 71.0, 50.0], [1113.0, 328.0, 67.0, 45.0], [1118.0, 333.0, 70.0, 48.0]],
    "video_id": 1, "iscrowd": 0, "id": 1, "areas": [null, null, null, null, null, null, null, null, 910, 1542, 1877, 2638, 2951, 2982, 3184, 3243, 2979, 2843, 2533, 2636]},

dict_keys(['height', 'width', 'length', 'category_id', 'segmentations', 'bboxes', 'video_id', 'iscrowd', 'id', 'areas'])
(Pdb) p js.keys()
dict_keys(['info', 'licenses', 'videos', 'categories', 'annotations'])
'''
