import os
from glob import glob
import json
import random
from PIL import Image
import numpy as np


train_file = 'data/ori_data/PreRoundData/ImageSets/train.txt'
img_root_path = 'data/ori_data/PreRoundData/JPEGImages'
ann_root_path = 'data/ori_data/PreRoundData/Annotations'


with open(train_file, 'r') as f:
    train_names = f.read().split('\n')[:-1]

unbalanced = 0

for train_name in train_names:
    video_dir = os.path.join(img_root_path, train_name)
    ann_dir = os.path.join(ann_root_path, train_name)
    frames = list(sorted(glob(os.path.join(video_dir, '*.jpg'))))
    masks = list(sorted(glob(os.path.join(ann_dir, '*.png'))))
    # if len(frames) != len(masks):
    #     print('unbalanced annotations!')
    #     print(train_name)
    #     unbalanced += 1
    frames = set(map(lambda x:x[:-4], os.listdir(video_dir)))
    masks = set(map(lambda x:x[:-4], os.listdir(ann_dir)))
    if masks-frames:
        print('unbalanced annotations!')
        print(train_name)
        unbalanced += 1
    if len(frames) != len(masks):
        print(len(frames) - len(masks))

print(unbalanced)

