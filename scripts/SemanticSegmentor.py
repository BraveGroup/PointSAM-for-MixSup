import argparse
import torch
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', default='result.jpg', help='Output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


class NuImagesSegmentor:
    def __init__(self, config, checkpoint, device='cuda:0'):
        self.device = device
        self.model = init_detector(config, checkpoint, device=device)
        self.num_classes = len(self.model.CLASSES)

    def predict(self, img, score_thr=0.3, return_numpy=False):
        h, w = img.shape[:2]
        if return_numpy:
            semantic_mask = np.zeros((h, w), dtype=np.int) + self.num_classes
        else:
            semantic_mask = torch.zeros((h, w), dtype=torch.int, device=self.device) + self.num_classes
        result = inference_detector(self.model, img)
        bbox_result, segm_result = result
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        if len(labels) > 0:
            segms = mmcv.concat_list(segm_result)
            segms = np.stack(segms, axis=0)
            # filter out low score bboxes and masks
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            segms = segms[inds, :]
            for idx, segm in enumerate(segms):
                semantic_mask[segm] = labels[idx]
        return semantic_mask


if __name__ == '__main__':
    args = parse_args()
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    img = mmcv.imread(args.img)
    result = inference_detector(model, img)
    # show the results
    show_result_pyplot(model, args.img, result, out_file=args.out)
