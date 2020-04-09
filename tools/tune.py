from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import numpy as np
import torch

from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.core.config import cfg
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp

torch.set_num_threads(1)


def parse_range(range_str):
    param = map(float, range_str.split(','))
    return np.arange(*param)


def parse_range_int(range_str):
    param = map(int, range_str.split(','))
    return np.arange(*param)


parser = argparse.ArgumentParser(description='Hyperparamter search')
parser.add_argument('--snapshot', type=str, help='snapshot of model')
parser.add_argument('--dataset', type=str, help='dataset name to eval')
parser.add_argument('--penalty-k', default='0.05,0.10,0.05', type=parse_range)
parser.add_argument('--lr', default='0.35,0.36,0.05', type=parse_range)
parser.add_argument('--window-influence', default='0.1,0.15,0.05', type=parse_range)
parser.add_argument('--search-region', default='255,256,8', type=parse_range_int)
parser.add_argument('--config', default='config.yaml', type=str)
args = parser.parse_args()


def run_tracker(tracker, img, gt, video_name, video, restart=True):
    frame_counter = 0
    lost_number = 0
    toc = 0
    pred_bboxes = []
    if restart:  # VOT2016 and VOT 2018
        for idx, (img, gt_bbox) in enumerate(video):
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1],
                           gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                           gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                           gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
            tic = cv2.getTickCount()
            if idx == frame_counter:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append([1])
            elif idx > frame_counter:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                overlap = vot_overlap(pred_bbox, gt_bbox,
                                      (img.shape[1], img.shape[0]))
                if overlap > 0:
                    # not lost
                    pred_bboxes.append(pred_bbox)
                else:
                    # lost object
                    pred_bboxes.append([2])
                    frame_counter = idx + 5  # skip 5 frames
                    lost_number += 1
            else:
                pred_bboxes.append([0])
            toc += cv2.getTickCount() - tic
        toc /= cv2.getTickFrequency()
        # print('Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(video_name, toc, idx / toc, lost_number))
        return pred_bboxes
    else:
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
        toc /= cv2.getTickFrequency()
        # print('Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(video_name, toc, idx / toc))
        return pred_bboxes, scores, track_times


def calculate_eao(tracker, tracker_name):
    all_pred_bboxes = {}
    for video in dataset:
        print('Running video: {}, tracker: {}'.format(video.name, tracker_name))
        video.load_img()
        pred_bboxes = [run_tracker(tracker, video.imgs, video.gt_traj, video.name, video, restart=True)]
        all_pred_bboxes[video.name] = pred_bboxes
        video.free_img()
    eao = benchmark._calculate_eao('tune', benchmark.tags, all_pred_bboxes)
    return eao['all']


# fitness function
def fitness(config, reporter):
    # Only support VOT Dataset temporarily
    # print('debug:fitness')
    tracker_name = 'tracker_penalty_k={0:.3f},window_influence={1:.3f},lr={2:.3f},search_region={3}'.format(config['penalty_k'], config['window_influence'], config['lr'], config['search_region'])

    if args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        tracker = build_tracker(model, config)
        eao = calculate_eao(tracker, tracker_name)
        print("penalty_k: {0:.3f}, lr: {1:.3f}, window_influence: {2:.3f}, search_region: {3}, eao: {4:.5f}".format(config['penalty_k'], config['window_influence'], config['lr'], config['search_region'], eao))
        reporter(EAO=eao)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)
    dataset = DatasetFactory.create_dataset(name=args.dataset, 
                                            dataset_root=dataset_root,
                                            load_img=False)
    benchmark = EAOBenchmark(dataset)

    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # the resources you computer have, object_store_memory is shm
    ray.init(num_gpus=1, num_cpus=8, object_store_memory=30000000000)
    tune.register_trainable('fitness', fitness)

    # define search space
    params = {
        'penalty_k': hp.quniform('penalty_k', 0.001, 0.6, 0.001),
        'lr': hp.quniform('scale_lr', 0.1, 0.8, 0.001),
        'window_influence': hp.quniform('window_influence', 0.05, 0.65, 0.001),
        'search_region': hp.choice('search_region', [255]),
    }

    # stop condition for VOT and OTB
    if args.dataset.startswith('VOT'):
        stop = {
            "EAO": 0.50,  # if EAO >= 0.50, this procedures will stop
            # "training_iteration": 100, # iteration times
        }

        scheduler = AsyncHyperBandScheduler(
            # time_attr="training_iteration",
            metric="EAO",
            mode='max',
            max_t=400,
            grace_period=20
        )
        algo = HyperOptSearch(params, max_concurrent=1, metric='EAO', mode='max') # max_concurrent: the max running task#
    else:
        raise ValueError("not support other dataset now")
    
    experiment_spec = tune.Experiment(
                            name='tune_tpe',
                            run='fitness',
                            stop=stop,
                            resources_per_trial={
                                'cpu': 1,
                                'gpu': 0.5
                            },
                            num_samples=10000,
                            local_dir='./TPE_results')
    tune.run_experiments(experiments=experiment_spec, search_alg=algo, scheduler=scheduler)

