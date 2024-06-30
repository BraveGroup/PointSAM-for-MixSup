from functools import partial

import numpy as np
from skimage import transform
import torch
from pathlib import Path
import pickle
import torchvision
from ...utils import box_utils, common_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]
            if 'weak_clusters_label' in data_dict and 'instance_ID_of_point' in data_dict['weak_clusters_label']:
                instance_ID_of_point = data_dict['weak_clusters_label']['instance_ID_of_point']
                data_dict['weak_clusters_label']['instance_ID_of_point'] = instance_ID_of_point[mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]

        if 'weak_clusters_label' in data_dict:
            mask = common_utils.mask_points_by_range(data_dict['weak_clusters_label']['center_of_cluster'], self.point_cloud_range)
            for key in data_dict['weak_clusters_label']:
                if key.endswith('of_cluster'):
                    data_dict['weak_clusters_label'][key] = data_dict['weak_clusters_label'][key][mask]
            if 'instance_ID_of_point' in data_dict['weak_clusters_label']:
                instance_ID_of_point = data_dict['weak_clusters_label']['instance_ID_of_point']
                remapping = np.zeros(len(mask) + 1, dtype=np.int64)
                remapping[np.where(mask)[0] + 1] = np.arange(mask.sum()) + 1
                data_dict['weak_clusters_label']['instance_ID_of_point'] = remapping[instance_ID_of_point]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

            if 'weak_clusters_label' in data_dict and 'instance_ID_of_point' in data_dict['weak_clusters_label']:
                instance_ID_of_point = data_dict['weak_clusters_label']['instance_ID_of_point']
                data_dict['weak_clusters_label']['instance_ID_of_point'] = instance_ID_of_point[shuffle_idx]

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def extract_mixed_label_from_box(self, data_dict=None, config=None):
        """
        Extract point clusters from gt_boxes whose mask is False, and fuse them with gt_boxes whose mask is True
        """
        if data_dict is None:
            self.only_keep_gt_box_scenes = config.get('ONLY_KEEP_GT_BOX_SCENES', False)
            assert config.CLUSTER_CENTER in ('cluster', 'box')
            self.cluster_center = config.CLUSTER_CENTER
            gt_box_mask_path = Path(config.ROOT_PATH).resolve() / config.GT_BOX_MASK_FILE
            with open(gt_box_mask_path, 'rb') as f:
                self.gt_box_mask = pickle.load(f)
            return partial(self.extract_mixed_label_from_box, config=config)

        if not self.training:   # only extract mixed label in training mode
            return data_dict
        points = torch.from_numpy(data_dict['points'])
        gt_boxes = torch.from_numpy(data_dict['gt_boxes'][:, :-1]).float()
        gt_boxes_label = torch.from_numpy(data_dict['gt_boxes'][:, -1]).long()
        if len(gt_boxes):
            pts_in_bboxes = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], gt_boxes[:, 0:7])
        else:
            pts_in_bboxes = torch.zeros((1, len(points)), dtype=torch.int32)
        in_flag, box_idx = pts_in_bboxes.max(0)
        in_flag = (in_flag == 1)
        coord_of_point = points[in_flag, :3]
        cluster_idx_of_point = box_idx[in_flag]
        center_of_cluster_in_box = [points.new_zeros((0, 3))]
        for idx in range(len(gt_boxes)):
            mask = (cluster_idx_of_point == idx)
            # get center
            if mask.sum() > 0:
                center = (coord_of_point[mask].max(0)[0] + coord_of_point[mask].min(0)[0]) / 2
            else:
                center = gt_boxes[idx, :3]
            center_of_cluster_in_box.append(center.unsqueeze(0))
        center_of_cluster_in_box = torch.cat(center_of_cluster_in_box, 0)
        gt_mixed_label_boxes = dict(box=gt_boxes,
                                    label=gt_boxes_label,
                                    cluster_center=center_of_cluster_in_box)
        assert len(gt_mixed_label_boxes['box']) == len(gt_mixed_label_boxes['cluster_center'])
        gt_weak_label = data_dict['weak_clusters_label']
        gt_mixed_label_clusters = dict(instance_ID_of_point=torch.from_numpy(gt_weak_label['instance_ID_of_point']).long(),     # (point level) the index of the cluster that the point belongs to
                                       label_of_cluster=torch.from_numpy(gt_weak_label['label_of_cluster']).long(),             # (cluster level) the label of the cluster
                                       center_of_cluster=torch.from_numpy(gt_weak_label['center_of_cluster']),                  # (cluster level) the center of the cluster
                                       count_of_cluster=torch.from_numpy(gt_weak_label['count_of_cluster']))                    # (cluster level) the number of points in the cluster
        # gt_mixed_label_clusters = dict(indices_of_point=indices_of_point,               # (point level) the index of point in the whole point cloud
        #                                coord_of_point=coord_of_point,                   # (point level) the coordinate of point in the whole point cloud
        #                                label_of_point=label_of_point,                   # (point level) the label of point in the whole point cloud
        #                                cluster_idx_of_point=cluster_idx_of_point,       # (point level) the index of the cluster that the point belongs to
        #                                label_of_cluster=gt_boxes_label[cluster_mask],   # (cluster level) the label of the cluster
        #                                center_of_cluster=center_of_cluster,             # (cluster level) the center of the cluster
        #                                count_of_cluster=count_of_cluster)               # (cluster level) the number of points in the cluster
        gt_mixed_label = dict(Clusters=gt_mixed_label_clusters, Bboxes=gt_mixed_label_boxes)
        data_dict['gt_mixed_label'] = gt_mixed_label
        data_dict.pop('weak_clusters_label', None)
        return data_dict

    def extract_weak_label_from_box(self, points, gt_boxes, gt_names, filter_min_points=1, weak_label_safe_distance=0.25):
        points = torch.from_numpy(points)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        # GT boxes are randomly expanded 0% to 10% in each dimension
        expand_time = torch.rand(len(gt_boxes), 3) * 0.1 + 1
        gt_boxes[:, 3:6] *= expand_time

        if len(gt_boxes) > 0:
            pts_in_bboxes = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], gt_boxes[:, 0:7])
        else:
            pts_in_bboxes = torch.zeros((1, len(points)), dtype=torch.int32)
        in_flag, box_idx = pts_in_bboxes.max(0)
        in_flag = (in_flag == 1)
        in_flag_indices = in_flag.nonzero().reshape(-1)
        coord_of_point = points[in_flag, :3]
        cluster_idx_of_point = box_idx[in_flag]
        center_of_cluster = torch.zeros((len(gt_boxes), 3), dtype=torch.float32)
        count_of_cluster = torch.zeros(len(gt_boxes), dtype=torch.int32)
        radius_of_cluster = torch.zeros(len(gt_boxes), dtype=torch.float32)
        instance_ID_of_point = torch.zeros(len(points), dtype=torch.int64)
        for idx in range(len(gt_boxes)):
            mask = (cluster_idx_of_point == idx)
            # get center
            if mask.sum() > 0:
                instance_ID_of_point[in_flag_indices[mask]] = idx + 1    # instance_ID = 0 means background
                center_of_cluster[idx] = (coord_of_point[mask].max(0)[0] + coord_of_point[mask].min(0)[0]) / 2
                count_of_cluster[idx] = mask.sum()
                radius_of_cluster[idx] = torch.sqrt(((coord_of_point[mask, :2] - center_of_cluster[idx, :2]) ** 2).sum(1)).max()
        mask = (count_of_cluster >= filter_min_points)
        # remapping the instance ID
        remapping = torch.zeros(len(mask) + 1, dtype=torch.int64)
        remapping[mask.nonzero().reshape(-1) + 1] = torch.arange(mask.sum()) + 1
        instance_ID_of_point = remapping[instance_ID_of_point]

        radius_of_cluster += weak_label_safe_distance
        weak_clusters_label = dict(instance_ID_of_point=instance_ID_of_point.numpy(),   # (point level) the index of the cluster that the point belongs to
                                   name_of_cluster=gt_names[mask.numpy()],              # (cluster level) the name of the cluster
                                   center_of_cluster=center_of_cluster[mask].numpy(),   # (cluster level) the center of the cluster
                                   count_of_cluster=count_of_cluster[mask].numpy(),     # (cluster level) the number of points in the cluster
                                   radius_of_cluster=radius_of_cluster[mask].numpy())   # (cluster level) the radius of the cluster
        return weak_clusters_label

    def extract_mixed_label_from_panoptic(self, data_dict=None, config=None):
        """
        Extract point clusters from panoptic labels whose mask is False, and fuse them with gt_boxes whose mask is True
        """
        if data_dict is None:
            self.panoptic_path = Path(config.PANOPTIC_PATH).resolve()
            select_mask_path = Path(config.SELECT_MASK_FILE).resolve()
            with open(select_mask_path, 'rb') as f:
                self.select_mask = pickle.load(f)
            # map_target should plus 1, because the first class is background
            label_mapping = [map_target + 1 for map_target in config.LABEL_MAPPING]
            # previous label indicates the last class as background, which should be mapped to 0
            label_mapping.append(0)
            self.label_mapping = np.array(label_mapping)
            self.weak_label_filter_min_points = config.WEAK_LABEL_FILTER_MIN_POINTS
            self.weak_label_safe_distance = config.WEAK_LABEL_SAFE_DISTANCE
            self.only_keep_gt_box_scenes = config.get('ONLY_KEEP_GT_BOX_SCENES', False)
            return partial(self.extract_mixed_label_from_panoptic, config=config)
        if not self.training:   # only extract mixed label in training mode
            return data_dict

        points = torch.from_numpy(data_dict['points'])
        gt_boxes = torch.from_numpy(data_dict['gt_boxes'][:, :-1]).float()
        gt_boxes_label = torch.from_numpy(data_dict['gt_boxes'][:, -1]).long()
        if len(gt_boxes) > 0:
            pts_in_bboxes = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, :3], gt_boxes[:, 0:7])
        else:
            pts_in_bboxes = torch.zeros((1, len(points)), dtype=torch.int32)
        in_flag, box_idx = pts_in_bboxes.max(0)
        in_flag = (in_flag == 1)
        coord_of_point = points[in_flag, :3]
        cluster_idx_of_point = box_idx[in_flag]
        center_of_cluster_in_box = [points.new_zeros((0, 3))]
        for idx in range(len(gt_boxes)):
            mask = (cluster_idx_of_point == idx)
            # get center
            if mask.sum() > 0:
                center = (coord_of_point[mask].max(0)[0] + coord_of_point[mask].min(0)[0]) / 2
            else:
                center = gt_boxes[idx, :3]
            center_of_cluster_in_box.append(center.unsqueeze(0))
        center_of_cluster_in_box = torch.cat(center_of_cluster_in_box, 0)
        gt_mixed_label_boxes = dict(box=gt_boxes,
                                    label=gt_boxes_label,
                                    cluster_center=center_of_cluster_in_box)
        # in_flag[:] = False
        # weak_clusters_label = self.extract_weak_label_from_panoptic(points=data_dict['points'][~in_flag, :3],
        #                                                             panoptic_labels=data_dict['points'][~in_flag, -2:],
        #                                                             filter_min_points=self.weak_label_filter_min_points)
        weak_clusters_label = data_dict['weak_clusters_label']
        gt_mixed_label_clusters = dict(label_of_cluster=torch.from_numpy(weak_clusters_label['label_of_cluster']).long(),   # (cluster level) the label of the cluster
                                       center_of_cluster=torch.from_numpy(weak_clusters_label['center_of_cluster']),        # (cluster level) the center of the cluster
                                       count_of_cluster=torch.from_numpy(weak_clusters_label['count_of_cluster']))          # (cluster level) the number of points in the cluster
        gt_mixed_label = dict(Clusters=gt_mixed_label_clusters, Bboxes=gt_mixed_label_boxes)
        data_dict['gt_mixed_label'] = gt_mixed_label
        data_dict['points'] = data_dict['points'][:, :-2]
        data_dict.pop('temp_src_feature_list', None)
        data_dict.pop('temp_used_feature_list', None)
        data_dict.pop('weak_clusters_label', None)
        return data_dict

    def extract_weak_label_from_panoptic(self, points, panoptic_labels, filter_min_points=1):
        # each row in panoptic_labels is [instance_id, class_id], instance_id == 65535 means background
        # we need to extract the point clusters from panoptic_labels whose instance_id is not 65535
        panoptic_labels = panoptic_labels.astype(np.uint16)
        valid_mask = (panoptic_labels[:, 0] != 65535)
        points, panoptic_labels = points[valid_mask], panoptic_labels[valid_mask]
        if valid_mask.any():
            assert panoptic_labels[:, 1].min() > 0 # the first class is background
        instance_indices = np.unique(panoptic_labels[:, 0])
        label_of_cluster = np.zeros(len(instance_indices), dtype=np.int32)
        center_of_cluster = np.zeros((len(instance_indices), 3), dtype=np.float32)
        count_of_cluster = np.zeros(len(instance_indices), dtype=np.int32)
        radius_of_cluster = np.zeros(len(instance_indices), dtype=np.float32)
        for i in range(len(instance_indices)):
            instance_idx = instance_indices[i]
            mask = (panoptic_labels[:, 0] == instance_idx)
            if mask.sum() > 0:
                label_of_cluster[i] = np.bincount(panoptic_labels[mask, 1]).argmax()
                center_of_cluster[i] = (points[mask, :3].max(0) + points[mask, :3].min(0)) / 2
                count_of_cluster[i] = mask.sum()
                radius_of_cluster[i] = np.linalg.norm(points[mask, :2] - center_of_cluster[i, :2], axis=1).max()
        # filter out the clusters whose count is less than filter_min_points
        mask = (count_of_cluster >= filter_min_points)
        radius_of_cluster += self.weak_label_safe_distance
        weak_clusters_label = dict(label_of_cluster=label_of_cluster[mask],   # (cluster level) the label of the cluster
                                   center_of_cluster=center_of_cluster[mask], # (cluster level) the center of the cluster
                                   count_of_cluster=count_of_cluster[mask],   # (cluster level) the number of points in the cluster
                                   radius_of_cluster=radius_of_cluster[mask]) # (cluster level) the radius of the cluster
        return weak_clusters_label

    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict
    
    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
