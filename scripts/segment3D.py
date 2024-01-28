import argparse
import os
from pathlib import Path

from mmcv import Config
import numpy as np
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from InstanceSegmentor import PointSAM
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--mask_root', help='the dir to save panoptic labels')
    parser.add_argument('--sample_indices', default=None, help='the indices of samples to run')
    parser.add_argument('--for_eval', action='store_true', help='save for evaluation')
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
    cfg.PointSAM.mask_root = args.mask_root
    if args.launcher == 'none':
        num_gpus = 1
        rank = 0
    else:
        num_gpus, rank = utils.init_dist(args.local_rank, backend='nccl')
    if rank == 0:
        if args.for_eval:
            os.makedirs(os.path.join(args.out_dir, 'panoptic/val'), exist_ok=True)
        else:
            os.makedirs(os.path.join(args.out_dir, 'samples'), exist_ok=True)
        os.system('cp %s %s' % (args.config, (Path(args.out_dir)).resolve()))
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    segmentor_3d = PointSAM(nusc, cfg, device=rank)
    if args.sample_indices is None:
        sample_indices = np.arange(len(nusc.sample))
    else:
        sample_indices = np.load(args.sample_indices)
    start = (len(sample_indices) // num_gpus + 1) * rank
    end = (len(sample_indices) // num_gpus + 1) * (rank + 1)
    local_sample_indices = sample_indices[start : end]
    pbar = tqdm(local_sample_indices, desc=f"Rank {rank}", position=rank)
    for sample_idx in local_sample_indices:
        point_semantic_id, point_instance_id = segmentor_3d.generate(sample_idx)
        segmentor_3d.save_panoptic_labels(point_semantic_id,
                                          point_instance_id,
                                          args.out_dir,
                                          sample_idx,
                                          for_eval=args.for_eval)
        pbar.update(1)

if __name__ == '__main__':
    main()