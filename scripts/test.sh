#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
python tools/test_video.py configs/masktrackrcnn_ytvos/masktrackrcnn_r50_fpn_2x.py \
work_dirs/latest.pth \
--out ../user_data/pred/result.pkl --format-only
python scripts/json2mask_v2.py
