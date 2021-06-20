#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=${PORT:-29500}

#cd ../
python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT tools/train.py \
configs/masktrackrcnn_ytvos/masktrackrcnn_r50_fpn_2x.py \
--work-dir work_dirs/masktrackrcnn_r50_fpn_2x_4 --no-validate --launcher pytorch
#source ./test.sh
