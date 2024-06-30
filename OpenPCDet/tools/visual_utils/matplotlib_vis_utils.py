import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

COLOR_MAP = {
    'gray': np.array([140, 140, 136]) / 256,
    'light_blue': np.array([4, 157, 217]) / 256,
    'blue': np.array([0, 0, 255]) / 256,
    'wine_red': np.array([191, 4, 54]) / 256,
    'red': np.array([255, 0, 0]) / 256,
    'black': np.array([0, 0, 0]) / 256,
    'purple': np.array([224, 133, 250]) / 256,
    'dark_green': np.array([32, 64, 40]) / 256,
    'green': np.array([77, 115, 67]) / 256,
    'yellow': np.array([255, 255, 0]) / 256
}


class Visualizer2D:
    def __init__(self, name='', figsize=(8, 8), x_range=None, y_range=None):
        self.figure = plt.figure(name, figsize=figsize)

        if x_range is not None:
            plt.xlim(x_range[0], x_range[1])
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])

        if x_range is None and y_range is None:
            plt.axis('equal')

        self.COLOR_MAP = {
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
        self.color_list = list(self.COLOR_MAP.keys())

    def show(self):
        plt.show()

    def close(self):
        plt.close()

    def save(self, path):
        plt.savefig(path)

    def handler_pc(self, pc, color='gray', s=0.25, marker='o'):
        vis_pc = np.asarray(pc)
        if isinstance(color, str):
            plt.scatter(vis_pc[:, 0], vis_pc[:, 1], s=s, marker=marker, color=self.COLOR_MAP[color])
        else:
            colors = np.array([self.COLOR_MAP[self.color_list[c % (len(self.color_list) - 1) + 1]] if c >= 0 else self.COLOR_MAP['gray'] for c in color])
            plt.scatter(vis_pc[:, 0], vis_pc[:, 1], s=s, marker=marker, color=colors)

    def handler_box(self, box, message: str='', color='red', linestyle='solid', text_color=None, fontsize='xx-small'):
        corners = np.array(boxes_to_corners_2d(box))[:, :2]
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        corner_index = np.random.randint(0, 4, 1)
        if text_color is None:
            text_color = color
        plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[text_color], fontsize=fontsize)

    def handler_box_4corners(self, corners, angle=False, message: str='', color='red', linestyle='solid', text_color=None, fontsize='xx-small'):
        assert corners.shape == (4, 2)
        corners = np.concatenate([corners, corners[0:1, :2]])
        plt.plot(corners[:, 0], corners[:, 1], color=self.COLOR_MAP[color], linestyle=linestyle)
        if angle:
            tail = corners[[1, 2], :].mean(0)
            head = corners[[0, 3], :2].mean(0)
            delta = head - tail
            delta = delta / np.linalg.norm(delta)
            plt.arrow(head[0], head[1], delta[0], delta[1], width=0.1, color=self.COLOR_MAP[color], length_includes_head=True)
        corner_index = np.random.randint(0, 4, 1)
        if text_color is None:
            text_color = color
        if len(message) > 0:
            plt.text(corners[corner_index, 0] - 1, corners[corner_index, 1] - 1, message, color=self.COLOR_MAP[text_color], fontsize=fontsize)

    def handler_tracklet(self, pc, id, color='gray', s=0.25, fontsize='xx-small'):
        vis_pc = np.asarray(pc)
        if isinstance(color, int):
            color = self.COLOR_MAP[self.color_list[color % len(self.color_list)]]
        else:
            color=self.COLOR_MAP[color]
        plt.scatter(vis_pc[:, 0], vis_pc[:, 1], s=s, marker='o', color=color)
        text = str(id)
        plt.text(vis_pc[0, 0] + 0.5, vis_pc[0, 1] + 0.5, text, color=self.COLOR_MAP['black'], fontsize=fontsize)
        plt.text(vis_pc[-1, 0] - 0.5, vis_pc[-1, 1] - 0.5, text, color=self.COLOR_MAP['red'], fontsize=fontsize)


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def boxes_to_corners_2d(boxes3d):
    """
        3 -------- 0
       /          /
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 4, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 4, 3), boxes3d[:, 6]).view(-1, 4, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def vis_bev_pc(pc, gts=None, pds=None, boxes=None, name='', save_root='./work_dirs/figs', figsize=(40, 40), color='gray', dir=None, messages=None, s=0.1, pc_range=None, angle=False):
    if isinstance(pc, torch.Tensor):
        pc = pc.cpu().detach().numpy()
    if isinstance(color, torch.Tensor):
        color = color.cpu().detach().numpy()
    if isinstance(gts, torch.Tensor):
        gts = gts.cpu().detach().numpy()
    if isinstance(pds, torch.Tensor):
        pds = pds.cpu().detach().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    assert '.png' in name or '.jpg' in name
    if dir:
        save_root = os.path.join(save_root, dir)
    os.makedirs(save_root, exist_ok=True)
    save_path = os.path.join(save_root, name)

    if pc_range is not None:
        x_range = y_range = (-pc_range, pc_range)
    else:
        x_range = y_range = None

    visualizer = Visualizer2D(name='', figsize=figsize, x_range=x_range, y_range=y_range)
    if pc is not None:
        visualizer.handler_pc(pc, s=s, color=color)

    if isinstance(gts, list):
        # for tracklet visualization
        colors = ['red', 'blue', 'green', 'black', 'light_blue']
        for frame_idx, gt in enumerate(gts):
            if gt is None:
                continue
            gt = boxes_to_corners_2d(gt)
            for i in range(len(gt)):
                visualizer.handler_box_4corners(gt[i, :, :2], angle=angle, color=colors[frame_idx % 5])

    elif gts is not None and len(gts) > 0:
        gts = boxes_to_corners_2d(gts)
        for i in range(len(gts)):
            visualizer.handler_box_4corners(gts[i, :, :2], angle=angle)

    if pds is not None and len(pds) > 0:
        pds = boxes_to_corners_2d(pds)
        for i in range(len(pds)):
            visualizer.handler_box_4corners(pds[i, :, :2], angle=angle, message='' if messages is None else messages[i], fontsize='xx-large', color='green')

    if boxes is not None and len(boxes) > 0:
        classes = boxes[:, -1].astype(np.int64)
        boxes = boxes_to_corners_2d(boxes[:, :7])
        color_list = visualizer.color_list
        for i in range(len(boxes)):
            visualizer.handler_box_4corners(boxes[i, :, :2], angle=angle, message='' if messages is None else messages[i], fontsize='xx-large', color=color_list[classes[i] % (len(color_list) - 1) + 1])

    visualizer.save(save_path)
    visualizer.close()
    print(f'Saved to {save_path}')