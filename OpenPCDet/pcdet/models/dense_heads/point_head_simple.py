import torch

from ...utils import box_utils
from .point_head_template import PointHeadTemplate


class PointHeadSimple(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PV-RCNN keypoint segmentaion.
    Reference Paper: https://arxiv.org/abs/1912.13192
    PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )

        return targets_dict

    def assign_targets_with_mixed_label(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_mixed_label: (dict)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        instance_ID_of_keypoint = input_dict['instance_ID_of_keypoint']
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        gt_mixed_label = input_dict['gt_mixed_label']
        batch_size = len(gt_mixed_label)
        # generate gt_boxes from gt_mixed_label
        num_gt_boxes = [len(mixed['Bboxes']['box']) for mixed in gt_mixed_label]
        gt_boxes = point_coords.new_zeros((batch_size, max(num_gt_boxes), 7+1))
        for k in range(batch_size):
            gt_boxes[k, :num_gt_boxes[k], :7] = gt_mixed_label[k]['Bboxes']['box']
            gt_boxes[k, :num_gt_boxes[k], 7] = gt_mixed_label[k]['Bboxes']['label']

        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False
        )
        
        point_cls_labels_from_cluster = self.assign_foreground_targets(gt_mixed_label, point_coords, instance_ID_of_keypoint)
        mask = (point_cls_labels_from_cluster > 0)
        targets_dict['point_cls_labels'][mask] = point_cls_labels_from_cluster[mask]
        return targets_dict

    def assign_foreground_targets(self, gt_mixed_label, points, instance_ID_of_keypoint):
        batch_size = len(gt_mixed_label)
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            instance_ID_single = instance_ID_of_keypoint[bs_mask]
            fg_flag = instance_ID_single > 0
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 \
                else gt_mixed_label[k]['Clusters']['label_of_cluster'][instance_ID_single[fg_flag]-1].to(device=points.device, dtype=torch.long)
            point_cls_labels[bs_mask] = point_cls_labels_single
        return point_cls_labels

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()

        point_loss = point_loss_cls
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_cls_preds': point_cls_preds,
        }

        point_cls_scores = torch.sigmoid(point_cls_preds)
        batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)

        if self.training:
            if 'gt_mixed_label' in batch_dict:
                targets_dict = self.assign_targets_with_mixed_label(batch_dict)
            else:
                targets_dict = self.assign_targets(batch_dict)
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        self.forward_ret_dict = ret_dict

        return batch_dict
