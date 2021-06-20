import datetime
import json
import os
import shutil

import cv2
import imgviz
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
import pandas as pd

def arrayImg(save_path):
    return np.array(Image.open(save_path), dtype=np.uint8)


def save_colored_mask(mask, save_path):
    pmask = Image.fromarray(mask.astype(np.uint8), mode='P')
    colormap = imgviz.label_colormap()
    pmask.putpalette(colormap.flatten())
    pmask.save(save_path)

def res2mask(res, ignore_exist=False, threshold=0.6):
    idx = 0
    firstID = float('inf')
    all_res = set()
    save_res = set()   # Not detect person!
    for vid_res in res:
        video = videos[vid_res['video_id'] - 1]
        all_res.add(video['id'])
        if vid_res['score'] < threshold or vid_res["category_id"] != 1:
            continue
        save_res.add(video['id'])
        print("processing video {}:".format(video['id']))

        if firstID != video['id']:
            idx = 0
            firstID = video['id']
        else:
            idx += 1
        if idx > 6:
            print('predicts too much')
            continue
        assert idx <= 6, 'predicts too much'
        save_dir_res = os.path.join(seg_save_dir, video['id'])
        if not os.path.exists(save_dir_res):
            os.mkdir(save_dir_res)
        else:
            if ignore_exist:
                continue
        for vid_frame_path, seg_res in zip(video["file_names"], vid_res['segmentations']):
            save_name = os.path.join(save_dir_res, os.path.basename(vid_frame_path.replace('jpg', 'png')))
            h, w = video['height'], video['width']
            if os.path.exists(save_name):
                assert idx != 0, "mask file {} exists!".format(save_name)
                mask = arrayImg(save_name)
                assert len(mask.shape) == 2
            else:
                mask = np.zeros((h, w), np.uint8)
                save_colored_mask(mask, save_name)

            if seg_res:
                seg_mask = mask_util.decode(seg_res)
                assert seg_mask.max() <= 1, 'decoding error!'
                assert mask.max() <= 8, 'mask overlapped'
                seg_mask[seg_mask == 1] += idx
                seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                seg_mask[mask!=0] = 0 # dealing with overlapped area.
                mask += seg_mask
                save_colored_mask(mask, save_name)
    return all_res, save_res

#TODO(yf): rank res
def preprocess_json(res):
    res_pd = pd.DataFrame(res)
    res_pd_sort = res_pd.groupby('video_id',group_keys=False).apply(lambda x: x.sort_values('score',ascending=False))
    res = list(res_pd_sort.to_dict('records'))
    # new_res = []
    # for vid_res in res:
    #     new_res.append(vid_res)
    return res
        # import pdb
        # pdb.set_trace()
seg_save_dir = '../user_data/result'
if not os.path.exists(seg_save_dir):
    os.makedirs(seg_save_dir)
else:
    shutil.rmtree(seg_save_dir)
    os.makedirs(seg_save_dir)

video_path = '/home/yunfeng/tianchi-code/user_data/annotations/instances_test_sub.json'
with open(video_path) as f:
    videos = json.load(f)["videos"]

res_path = '/home/yunfeng/tianchi-code/user_data/pred/result.pkl.json'
with open(res_path) as f:
    res = json.load(f)

res = preprocess_json(res)
threshold = 0.1
# threshold = 0.2
all_res, save_res = res2mask(res, ignore_exist=False, threshold=threshold)
# assert len(all_res) == 48
blank = list(all_res-save_res)
while blank:
    threshold -= 0.1
    all_res, save_res = res2mask(res, ignore_exist=True, threshold=threshold)
    blank = list(all_res - save_res)

# import pdb
# pdb.set_trace()


# zipping results
shutil.make_archive(os.path.join(seg_save_dir, '../../submit/sumbit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
                    'zip', seg_save_dir)
