# masktrackrcnn-mmdet2.0
This is the **UNOFFICIAL** implement of the MaskTrackRCNN [[paper](https://arxiv.org/abs/1905.04804)] for video instance segmentation. We adapt the method with additional post-processing for Tianchi competetion [[link](https://tianchi.aliyun.com/competition/entrance/531873/introduction)], which achieves a score of **J&F-Mean 0.660** and ranks **15 / 2904** in the competition. The code is mostly built based on [MaskTrackRCNN](https://github.com/youtubevos/MaskTrackRCNN) and re-implemented via [mmdetection 2.11.0](https://github.com/open-mmlab/mmdetection/tree/v2.11.0). **Some parameters (such as num_classes and dataset structure) are specifically set to adapt to the Tianchi dataset.** Although, this has no effect on the training of the model.


## Setup
We run the code successfully using pytoch>=1.7.0 and cuda 11.0.   We are not sure if the code works in other environments. 

## Installation
```sh
conda create -n MaskTrackRCNN -y
conda activate MaskTrackRCNN
conda install pytorch torchvision cudatoolkit=11 -c pytorch -y
conda install -c conda-forge opencv -y
conda install cython -y
# install the customized COCO API for YouTubeVIS dataset.
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
# install mmdetection, please refer to https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md
pip install mmcv-full
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```
## Training
1. modify the dataset path in `scripts/gen_train_val_json_mp.py`.
2. prepocessing the dataset using multiprocessing.
```sh
python ./scripts/gen_train_val_json_mp.py
```
3. run distributed training on 4 GPUs.
```
export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=${PORT:-29500}

python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT tools/train.py \
configs/masktrackrcnn_ytvos/masktrackrcnn_r50_fpn_2x.py \
--work-dir work_dirs/masktrackrcnn_r50_fpn_2x_4 --no-validate --launcher pytorch.  
```
We lost the pretrained models..
## Evaluation
1. prepare the datset.
```sh
python scripts/gen_test_json.py
```
2. modify the model path and evaluate the model.
```sh
python tools/test_video.py configs/masktrackrcnn_ytvos/masktrackrcnn_r50_fpn_2x.py \
work_dirs/latest.pth \
--out ../user_data/pred/result.pkl --format-only
```
The predicted results will be generated into a json file named `../user_data/pred/result.pkl.json`, you can convert the json results to segmentation masks by runing,
```sh
python scripts/json2mask_v2.py
```
## Acknowledgement
The authors are [LYxjtu](https://github.com/LYxjtu) and [Exely](https://github.com/Exely). If you have any questions about this repo, please contact us or create an issue.
