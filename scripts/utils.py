import os.path as osp
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from pyquaternion import Quaternion

COLOR_MAP = {
            'gray': np.array([140, 140, 136]) / 256,
            'light_gray': np.array([200, 200, 200]) / 256,
            'lighter_gray': np.array([220, 220, 220]) / 256,
            'light_blue': np.array([135, 206, 217]) / 256,
            'sky_blue': np.array([135, 206, 235]) / 256,
            'blue': np.array([0, 0, 255]) / 256,
            'wine_red': np.array([191, 4, 54]) / 256,
            'red': np.array([255, 0, 0]) / 256,
            'black': np.array([0, 0, 0]) / 256,
            'purple': np.array([224, 133, 250]) / 256, 
            'dark_green': np.array([32, 64, 40]) / 256,
            'green': np.array([77, 200, 67]) / 256,
            'yellow': np.array([200, 200, 100]) / 256
        }
COLOR_LIST = list(COLOR_MAP.keys())

def init_dist(local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
    )
    rank = dist.get_rank()
    return num_gpus, rank

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_segment(instance_mask):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((instance_mask.shape[0], instance_mask.shape[1], 4))
    img[:,:,3] = 0
    for i in range(instance_mask.max() + 1):
        mask = instance_mask == i
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[mask] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_pc(pc, ax, color='green', s=0.1, viewpoint=None):
    pc = pc[:, :3]
    if viewpoint is not None:
        pc = view_points(pc.T, viewpoint, normalize=False).T
    if isinstance(color, str):
        ax.scatter(pc[:, 0], pc[:, 1], s=s, color=COLOR_MAP[color])
    else:
        colors = np.array([COLOR_MAP[COLOR_LIST[c % (len(COLOR_LIST) - 1) + 1]] if c >= 0 else COLOR_MAP['gray'] for c in color])
        ax.scatter(pc[:, 0], pc[:, 1], s=s, color=colors)

def load_pointcloud(nusc, sample_data):
    return LidarPointCloud.from_file(osp.join(nusc.dataroot, sample_data['filename']))

def load_pointcloud_multisweep(nusc, sample_rec, sweeps=[0], min_distance=1.0):
    """
    Load a point cloud that aggregates multiple sweeps.
    :param nusc: A NuScenes instance.
    :param sample_rec: The current sample.
    :param sweeps: List of sweeps to aggregate, where 0 is the current sample, -1 is the previous sample, 1 is the next sample.
    :param min_distance: Distance below which points are discarded.
    :return: <np.float: 2, nbr_points>.
    """

    def get_sample_record_from_idx(ref_sample_data_rec, idx):
        current_sd_rec = ref_sample_data_rec
        for _ in range(abs(idx)):
            succeed_token = current_sd_rec['next'] if idx > 0 else current_sd_rec['prev']
            if succeed_token == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', succeed_token)
        return current_sd_rec
    
    def remove_close(points, dist_thresh):
        x_filt = np.abs(points[0, :]) < dist_thresh
        y_filt = np.abs(points[1, :]) < dist_thresh
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return not_close

    all_pc = LidarPointCloud(np.zeros((LidarPointCloud.nbr_dims(), 0), dtype=np.float32))
    all_times = np.zeros(0)
    all_sweeps = np.zeros(0, dtype=np.int32)
    all_masks = np.zeros(0, dtype=bool)

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transform from ego car frame to reference frame.
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate multiple sweeps.
    for idx in sweeps:
        # Load up the pointcloud and remove points close to the sensor.
        current_sd_rec = get_sample_record_from_idx(ref_sd_rec, idx)
        if current_sd_rec is None:
            continue
        current_pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        current_mask = np.ones(current_pc.nbr_points(), dtype=bool)
        if idx != 0:
            current_mask = remove_close(current_pc.points, min_distance)

        # Get pastfuture pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
        times = time_lag * np.ones(current_pc.nbr_points())
        all_times = np.concatenate((all_times, times))
        sweep_idx = idx * np.ones(current_pc.nbr_points(), dtype=np.int32)
        all_sweeps = np.concatenate((all_sweeps, sweep_idx))

        # Merge with key pc.
        all_pc.points = np.hstack((all_pc.points, current_pc.points))
        all_masks = np.concatenate((all_masks, current_mask))

    return all_pc, all_times, all_sweeps, all_masks

def map_proj_to_image(nusc, cam, pointsensor, pc, img):
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > 1.0)
    # x axis indicates right in image, y axis indicates downwards in image.
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < img.shape[1] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < img.shape[0] - 1)
    return points, mask

def farthest_point_sample(coords, npoint):
    """
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/eb64fe0b4c24055559cea26299cb485dcb43d8dd/models/pointnet2_utils.py#L63
    Input:
        coords: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = coords.device
    B, N, C = coords.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = coords[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((coords - centroid) ** 2, -1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids
