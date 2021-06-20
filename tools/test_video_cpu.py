import argparse

import mmcv
import torch
from mmcv import Config
from mmdet.apis import init_detector
from mmcv.parallel import MMDataParallel
from mmcv.parallel.scatter_gather import scatter
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import os


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        data = scatter(data, [-1])[0]  # dealing with parallel bug! yf&ly
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.extend(result)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model on CPU. Authors@yf&ly.')
    parser.add_argument('--config', default='../configs/tianchi/masktrackrcnn_r50_fpn_2x_debug.py', help='test config file path')
    parser.add_argument('--checkpoint', default='../../user_data/work_dirs/latest.pth', help='checkpoint file')
    parser.add_argument('--out', default=True, help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        default=True,
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    args = parser.parse_args()
    return args


def main():
    print('yf&ly is NO.1!')
    args = parse_args()
    print('great, we have all correct args!')
    cfg = Config.fromfile(args.config)
    print('great, we find the config file!')
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        # samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        # if samples_per_gpu > 1:
        #     # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        # cfg.data.test.pipeline = replace_ImageToTensor(
        #     cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
    # build the model and load checkpoint
    cfg.model.train_cfg = None

    print('great, we update the config file!')
    print("------------CONFIG---------------")
    print(cfg)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    print('great, we build the dataset!')
    print("------------DATASET---------------")
    print(dataset)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    print('great, we build the dataloader!')
    # print("------------DATALOADER---------------")
    # print(data_loader)

    print('model initialing...')
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    print('great, we build the model!')
    print("------------MODEL---------------")
    print(model.__str__().replace('\n', ' --- '))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print('great, we find the checkpoint!')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # model = init_detector(args.config, args.checkpoint, device='cpu')
    # model = MMDataParallel(model)
    print('finishing model initialization')
    # model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    print('now, we start to evaluate our model!')
    outputs = single_test(model, data_loader)
    print('congratulations, yf&ly! you finish the eval!')

    assert args.out, 'args.out must be set'
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(outputs, args.out)
        result_file = args.out + '.json'
        print('we have saved the result in json format')
        # results2json_videoseg(dataset, outputs, result_file)
    kwargs = {} if args.eval_options is None else args.eval_options
    kwargs['result_file'] = result_file
    if args.format_only:
        print('converting json to segmentation results')
        dataset.format_results(outputs, **kwargs)
        print('congratulations, yf&ly! you rank NO.1 on LB!')


if __name__ == '__main__':
    main()
