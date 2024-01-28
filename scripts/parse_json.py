import argparse
import json
import numpy as np


class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='json file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    with open(args.json, 'r') as f:
        data = json.load(f)
    PQ_mean = np.mean([data['segmentation'][class_name]["PQ"] for class_name in class_names])
    SQ_mean = np.mean([data['segmentation'][class_name]["SQ"] for class_name in class_names])
    RQ_mean = np.mean([data['segmentation'][class_name]["RQ"] for class_name in class_names])
    print(f'PQ: {PQ_mean*100:.1f}, SQ: {SQ_mean*100:.1f}, RQ: {RQ_mean*100:.1f}')