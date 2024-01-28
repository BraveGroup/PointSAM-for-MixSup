# convert npz to bin or bin to npz
# Usage: python scripts/convert_results.py --mode npz2bin --input /path/to/npz --output /path/to/bin
#        python scripts/convert_results.py --mode bin2npz --input /path/to/bin --output /path/to/npz
import argparse
import os
from os import path as osp
import numpy as np
from nuscenes import NuScenes
from tqdm import tqdm

BIN_CLASSES = {
    'barrier': 9,
    'bicycle': 5,
    'bus': 3,
    'car': 0,
    'construction_vehicle': 4,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 2,
    'truck': 1,
    'ignore': 10
}

NPZ_CLASSES = {
    'ignore': 0,
    'barrier': 1,
    'bicycle': 2,
    'bus': 3,
    'car': 4,
    'construction_vehicle': 5,
    'motorcycle': 6,
    'pedestrian': 7,
    'traffic_cone': 8,
    'trailer': 9,
    'truck': 10
}
mapping_from_bin_to_npz = np.zeros(len(BIN_CLASSES), dtype=np.uint16)
for class_name, class_id in BIN_CLASSES.items():
    mapping_from_bin_to_npz[class_id] = NPZ_CLASSES[class_name]
mapping_from_npz_to_bin = np.zeros(len(NPZ_CLASSES), dtype=np.uint16)
for class_name, class_id in NPZ_CLASSES.items():
    mapping_from_npz_to_bin[class_id] = BIN_CLASSES[class_name]


def convert_npz_to_bin(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
    # use tqdm to show progress bar
    for sample in tqdm(nusc.sample):
        pointsensor_token = sample['data']['LIDAR_TOP']
        input_filename = f"{pointsensor_token}_panoptic.npz"
        input_path = osp.join(input_root, input_filename)
        output_filename = osp.basename(nusc.get('sample_data', pointsensor_token)['filename'])
        output_path = osp.join(output_root, output_filename)
        if not osp.exists(input_path):
            continue
        input_label = np.load(input_path)['data']
        output_label = np.zeros((len(input_label), 2), dtype=np.uint16)
        output_label[:, 0] = 65535
        output_label[:, 1] = mapping_from_npz_to_bin[0]
        valid_mask = (input_label % 1000) != 0
        assert np.all(input_label[valid_mask] // 1000 != NPZ_CLASSES['ignore'])
        output_label[valid_mask, 0] = input_label[valid_mask] % 1000
        output_label[valid_mask, 1] = mapping_from_npz_to_bin[input_label[valid_mask] // 1000]
        output_label.tofile(output_path)


def convert_bin_to_npz(input_root, output_root):
    os.makedirs(output_root, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=True)
    for sample in tqdm(nusc.sample):
        pointsensor_token = sample['data']['LIDAR_TOP']
        input_filename = osp.basename(nusc.get('sample_data', pointsensor_token)['filename'])
        input_path = osp.join(input_root, input_filename)
        output_filename = f"{pointsensor_token}_panoptic.npz"
        output_path = osp.join(output_root, output_filename)
        if not osp.exists(input_path):
            continue
        input_label = np.fromfile(input_path, dtype=np.uint16).reshape((-1, 2))
        output_label = np.zeros((len(input_label),), dtype=np.uint16) + NPZ_CLASSES['ignore']
        valid_mask = input_label[:, 0] != 65535
        assert np.all(input_label[valid_mask, 1] != BIN_CLASSES['ignore'])
        output_label[valid_mask] += mapping_from_bin_to_npz[input_label[valid_mask, 1]] * 1000 + input_label[valid_mask, 0] + 1
        np.savez_compressed(output_path, data=output_label)

def args_parser():
    parser = argparse.ArgumentParser(description='Convert results between npz and bin')
    parser.add_argument('--mode', type=str, choices=['npz2bin', 'bin2npz'])
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args


def main(mode, input_path, output_path):
    if mode == 'npz2bin':
        convert_npz_to_bin(input_path, output_path)
    elif mode == 'bin2npz':
        convert_bin_to_npz(input_path, output_path)
    else:
        raise ValueError(f'Invalid mode: {mode}')

if __name__ == '__main__':
    args = args_parser()
    main(args.mode, args.input, args.output)