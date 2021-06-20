import os.path as osp
import random
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.ytvos import YTVOS
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS
from .pipelines import Compose
from .custom import CustomDataset


@DATASETS.register_module()
class YTVOSDataset(CustomDataset):

    CLASSES=('person',)

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
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
        self.vid_infos = self.load_annotations(ann_file)  # replace data_infos by vid_info, {'filenames':['']}
        img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
          for frame_id in range(len(vid_info['filenames'])):
            img_ids.append((idx, frame_id))
        self.img_ids = img_ids  # img_ids represent the tuple index of all frames, (video_id, frame_id)
        
        # # load annotations (and proposals)
        # self.data_infos = self.load_annotations(self.ann_file)

        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
                if len(self.get_ann_info(v, f)['bboxes'])]
            self.img_ids = [self.img_ids[i] for i in valid_inds]
            self._set_group_flag()

        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __str__(self):
        return 'YTVOS has vid_infos:\n'+str(self.vid_infos)

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        if self.test_mode:
             return self.prepare_test_img(self.img_ids[idx])
        data = self.prepare_train_img(self.img_ids[idx])
        return data

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i
            for i, cat_id in enumerate(self.cat_ids)
        }  # Here !
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            del info['file_names']
            vid_infos.append(info)
        return vid_infos

    def load_annotations_frame(self, vid, frame_id):
        vid_info = self.vid_infos[vid].copy()
        vid_info['filename'] = vid_info['filenames'][frame_id]
        return vid_info

    def get_ann_info(self, idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, frame_id)

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            vid_id, _ = self.img_ids[i]
            vid_info = self.vid_infos[vid_id]
            if vid_info['width'] / vid_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def sample_ref(self, idx):
        # sample another frame in the same sequence as reference
        vid, frame_id = idx
        vid_info = self.vid_infos[vid]
        sample_range = range(len(vid_info['filenames']))
        valid_samples = []
        for i in sample_range:
          # check if the frame id is valid
          ref_idx = (vid, i)
          if i != frame_id and ref_idx in self.img_ids:
              valid_samples.append(ref_idx)
        assert len(valid_samples) > 0
        return random.choice(valid_samples)

    def prepare_train_img(self, idx):
        vid, frame_id = idx
        img_info = self.load_annotations_frame(vid, frame_id)
        ann_info = self.get_ann_info(vid, frame_id)

        _, ref_frame_id = self.sample_ref(idx)
        ref_img_info = self.load_annotations_frame(vid, ref_frame_id)
        ref_ann_info = self.get_ann_info(vid, ref_frame_id)

        results = dict(img_info=img_info, ann_info=ann_info)
        results_ref = dict(img_info=ref_img_info, ann_info=ref_ann_info)  # TODO(yf): whether to use two pipelines?
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        self.pre_pipeline(results_ref)

        ref_ids = ref_ann_info['obj_ids']
        gt_ids = ann_info['obj_ids']
        # compute matching of reference frame with current frame
        # 0 denote there is no matching
        # instance ids
        gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]

        results['gt_pids'] = gt_pids
        results_ref['gt_pids'] = gt_pids

        # mmdetv2.11: by preprocessing results with pipeline = [
        #     dict(type='LoadImageFromFile'),
        #     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        #     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        #     dict(type='RandomFlip', flip_ratio=0.5),
        #     dict(type='Normalize', **img_norm_cfg),
        #     dict(type='Pad', size_divisor=32),
        #     dict(type='DefaultFormatBundle'),
        #     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
        # ]
        # then we obtained results as:
        # results = {dict:5}{
        # 'img_metas': DataContainer({'filename': '/home/yunfeng/tianchi-code/tcdata/PreRoundData/JPEGImages/606332/00102.jpg', 'ori_filename': '606332/00102.jpg', 'ori_shape': (720, 1280, 3), 'img_shape': (750, 1333, 3), 'pad_shape': (768, 1344, 3), 'scale_factor': array([1.0414063, 1.0416666, 1.0414063, 1.0416666], dtype=float32), 'flip': True, 'flip_direction': 'horizontal', 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}),
        # 'img': DataContainer(tensor([[[-1.1589, -1.1589, -1.1589,  ...,  0.0000,  0.0000,  0.0000],
        #          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]])),
        # 'gt_bboxes': DataContainer(tensor([[ 801.8828,   60.4167, 1033.0750,  750.0000]])),
        # 'gt_labels': DataContainer(tensor([1])),
        # 'gt_masks': DataContainer(BitmapMasks(num_masks=1, height=768, width=1344))}

        results = self.pipeline(results)
        results_ref = self.pipeline(results_ref)
        results['ref_img'] = results_ref['img']
        results['ref_bboxes'] = results_ref['gt_bboxes']

        return results

    def prepare_test_img(self, idx):
        vid, frame_id = idx
        img_info = self.load_annotations_frame(vid, frame_id)
        results = dict(img_info=img_info)
        results['video_id'] = vid
        results['frame_id'] = frame_id
        results['is_first'] = (frame_id == 0)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def format_results(self, results, **kwargs):
        out_file = kwargs['result_file']
        dataset = self
        json_results = []
        vid_objs = {}
        for idx in range(len(dataset)):
            # assume results is ordered

            vid_id, frame_id = dataset.img_ids[idx]
            if idx == len(dataset) - 1:
                is_last = True
            else:
                _, frame_id_next = dataset.img_ids[idx + 1]
                is_last = frame_id_next == 0
            det, seg = results[idx]
            for obj_id in det:
                bbox = det[obj_id]['bbox']
                segm = seg[obj_id]
                label = det[obj_id]['label']
                if obj_id not in vid_objs:
                    vid_objs[obj_id] = {'scores': [], 'cats': [], 'segms': {}}
                vid_objs[obj_id]['scores'].append(bbox[4])
                vid_objs[obj_id]['cats'].append(label)
                segm['counts'] = segm['counts'].decode()
                vid_objs[obj_id]['segms'][frame_id] = segm
            if is_last:
                # store results of  the current video
                for obj_id, obj in vid_objs.items():
                    data = dict()

                    data['video_id'] = vid_id + 1
                    data['score'] = np.array(obj['scores']).mean().item()
                    # majority voting for sequence category
                    data['category_id'] = np.bincount(np.array(obj['cats'])).argmax().item() + 1
                    vid_seg = []
                    for fid in range(frame_id + 1):
                        if fid in obj['segms']:
                            vid_seg.append(obj['segms'][fid])
                        else:
                            vid_seg.append(None)
                    data['segmentations'] = vid_seg
                    json_results.append(data)
                vid_objs = {}
            mmcv.dump(json_results, out_file)

        """Place holder to format result to dataset specific output."""

    # TODO(yf): set evaluation
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        raise NotImplementedError('evaluation not exists!')

    def _parse_ann_info(self, ann_info, frame_id):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            bbox = ann['bboxes'][frame_id]
            area = ann['areas'][frame_id]
            segm = ann['segmentations'][frame_id]
            if bbox is None: continue
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]  # ? -1 or not
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_ids.append(ann['id'])
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(segm)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            obj_ids=gt_ids,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann)

        return ann
