#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
#cd ../
python tools/train.py \
configs/masktrackrcnn_ytvos/masktrackrcnn_r50_fpn_2x.py \
--work-dir work_dirs/masktrackrcnn_r50_fpn_2x_debug --no-validate
#source ./test.sh
