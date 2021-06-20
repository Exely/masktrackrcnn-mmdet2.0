import os
from glob import glob
import json
import pdb
from PIL import Image

root_dir = '/'

test_dirs_root = os.path.join(root_dir, 'tcdata')
save_test_file = os.path.join(root_dir, 'user_data/annotations/instances_test_sub.json')
test_dirs = sorted([x for x in os.listdir(test_dirs_root) if x.isdigit()])

js = {}
js['licenses']=['ly&yf']
js['info'] = ['ly&yf']
js['videos'] = []
js['categories'] = [{"supercategory": "object", "id": 1, "name": "person"},]


count = 0
img_id = 1
obj_id = 0
# vid_id = 1

'''
{"width": 1280, "length": 19, "date_captured": "2019-04-11 00:18:36.249120", "license": 1, "flickr_url": "",
 "file_names": ["0070461469/00000.jpg", "0070461469/00005.jpg", "0070461469/00010.jpg", "0070461469/00015.jpg", 
 "0070461469/00020.jpg", "0070461469/00025.jpg", "0070461469/00030.jpg", "0070461469/00035.jpg", 
 "0070461469/00040.jpg", "0070461469/00045.jpg", "0070461469/00050.jpg", "0070461469/00055.jpg", 
 "0070461469/00060.jpg", "0070461469/00065.jpg", "0070461469/00070.jpg", "0070461469/00075.jpg", 
 "0070461469/00080.jpg", "0070461469/00085.jpg", "0070461469/00090.jpg"],
'''


for test_name in test_dirs:
    test_dir = os.path.join(test_dirs_root, test_name)
    frames = list(sorted(glob(os.path.join(test_dir, '*.jpg'))))
    # frames_names = list(map(lambda x: os.path.join(), frames))
    length = len(frames)
    assert length > 0
    img0 = frames[0]
    img0 = Image.open(img0)
    w, h = img0.size
    # pdb.set_trace()
    img_list = []
    save_file_names = list(map(lambda x: os.path.relpath(x, test_dirs_root), frames))
    js['videos'].append({"width": w, "height":h, "length": length, "date_captured":'', "license": 1,
                         "coco_url": "", "flickr_url": "","id":test_name, "file_names": save_file_names})
    # vid_id += 1

# test = {k:v for (k,v) in js.items()}
json.dump(js, open(save_test_file, 'w'), indent=4, sort_keys=False)
