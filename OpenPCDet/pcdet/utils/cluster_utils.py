import numpy as np
import scipy
import torch
import copy

import torch_scatter
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from . import common_utils


def c2b_iou(clusters, boxes, pts_xyz, mask_clusters=None, max_box_num=-1):
    """
    Compute the IoU between clusters and boxs.

    Args:
        clusters (dict): Clusters.
        boxes (Tensor): Bounding boxes, shape(n, 7) or (n, 4).
        pts_xyz (Tensor): Points xyz coordinates, shape(m, 3).
        max_box_num (int): If > 0, only consider the first max_box_num boxes for each point.
    """
    assert boxes.shape[1] == 7 or boxes.shape[1] == 4
    assert pts_xyz.shape[0] > 0 and pts_xyz.shape[1] == 3

    if mask_clusters is None:
        mask_clusters = torch.ones_like(clusters['label_of_cluster'], dtype=torch.bool)
    iou = pts_xyz.new_zeros((len(boxes), mask_clusters.sum()))
    if mask_clusters.sum() == 0 or boxes.shape[0] == 0:
        return iou
    instance_ID_of_point = clusters['instance_ID_of_point']
    remapping = torch.zeros(len(mask_clusters) + 1, dtype=torch.long)
    remapping[mask_clusters.nonzero().reshape(-1) + 1] = torch.arange(mask_clusters.sum()) + 1
    instance_ID_of_point = remapping[instance_ID_of_point].to(pts_xyz.device)

    num_pts_in_clusters = clusters['count_of_cluster'][mask_clusters].to(pts_xyz.device).float()
    if boxes.shape[1] == 7:
        pts_in_boxes = roiaware_pool3d_utils.points_in_multi_boxes_gpu(pts_xyz.unsqueeze(0), boxes.unsqueeze(0), max_box_num).squeeze(0)
    else:
        pts_in_boxes = roiaware_pool3d_utils.points_in_multi_aligned_bev_boxes_gpu(pts_xyz.unsqueeze(0), boxes.unsqueeze(0), max_box_num).squeeze(0)
    pts_idx_in_boxes, box_idx_of_pts = torch.where(pts_in_boxes != -1)
    box_ID_of_point = pts_in_boxes[pts_idx_in_boxes, box_idx_of_pts].long()
    cluster_ID_of_point_in_box = instance_ID_of_point[pts_idx_in_boxes] - 1 # convert 1-based to 0-based
    num_pts_in_boxes_and_clusters = torch.zeros_like(iou)
    mask_pts_in_boxes_and_clusters = (cluster_ID_of_point_in_box != -1)
    intersect_coords = torch.cat((box_ID_of_point[mask_pts_in_boxes_and_clusters].unsqueeze(1), cluster_ID_of_point_in_box[mask_pts_in_boxes_and_clusters].unsqueeze(1)), dim=1)
    if not mask_pts_in_boxes_and_clusters.any():
        # no points in boxes and clusters
        return iou

    assert (intersect_coords >= 0).all()
    # compute the number of points in each box
    num_pts_in_boxes = iou.new_zeros(len(boxes), dtype=torch.float32)
    num_pts_in_boxes.scatter_add_(0, box_ID_of_point, torch.ones_like(box_ID_of_point, dtype=torch.float32))
    # compute the number of points in each box and cluster
    new_coors, unq_cnt = torch.unique(intersect_coords, return_inverse=False, return_counts=True, dim=0)
    num_pts_in_boxes_and_clusters[new_coors[:, 0], new_coors[:, 1]] = unq_cnt.float()
    # compute the IoU
    iou = num_pts_in_boxes_and_clusters / (num_pts_in_boxes[:, None] + num_pts_in_clusters[None, :] - num_pts_in_boxes_and_clusters + 1e-6)
    return iou


def get_max_c2b_iou_with_same_class(clusters, boxes, labels, pts_xyz):
    max_overlaps = boxes.new_zeros(boxes.shape[0])
    gt_assignment = labels.new_zeros(labels.shape[0])
    gt_cluster_labels = clusters['label_of_cluster'].to(pts_xyz.device)

    if len(gt_cluster_labels) == 0 or len(labels) == 0:
        return max_overlaps, gt_assignment

    for k in range(gt_cluster_labels.min().item(), gt_cluster_labels.max().item() + 1):
        boxes_mask = (labels == k)
        gt_mask = (gt_cluster_labels == k)
        if boxes_mask.sum() > 0 and gt_mask.sum() > 0:
            cur_box = boxes[boxes_mask]
            original_gt_assignment = gt_mask.nonzero().view(-1)
            iou3d = c2b_iou(clusters, cur_box, pts_xyz, gt_mask)
            cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
            max_overlaps[boxes_mask] = cur_max_overlaps
            gt_assignment[boxes_mask] = original_gt_assignment[cur_gt_assignment]
    return max_overlaps, gt_assignment
