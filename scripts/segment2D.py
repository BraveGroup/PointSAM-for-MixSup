import argparse
import os
from pathlib import Path
from mmcv import Config
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from InstanceSegmentor import InstanceSegmentor
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path')
    parser.add_argument('--sample_indices', default=None, help='the indices of samples to run')
    parser.add_argument('--out_dir', help='the dir to save point cloud segmentations')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.launcher == 'none':
        num_gpus = 1
        rank = 0
    else:
        num_gpus, rank = utils.init_dist(args.local_rank, backend='nccl')
    if rank == 0:
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        for camera in camera_list:
            os.makedirs(os.path.join(args.out_dir, camera), exist_ok=True)
        os.system('cp %s %s' % (args.config, (Path(args.out_dir)).resolve()))
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    segmentor_2d = InstanceSegmentor(nusc, cfg, device=rank)
    if args.sample_indices is None:
        sample_indices = np.arange(len(nusc.sample))
    else:
        sample_indices = np.load(args.sample_indices)
    start = (len(sample_indices) // num_gpus + 1) * rank
    end = (len(sample_indices) // num_gpus + 1) * (rank + 1)
    local_sample_indices = sample_indices[start : end]
    pbar = tqdm(local_sample_indices, desc=f"Rank {rank}")
    for sample_idx in local_sample_indices:
        segmentor_2d.save_masks(args.out_dir, sample_idx)
        pbar.update(1)

if __name__ == '__main__':
    main()