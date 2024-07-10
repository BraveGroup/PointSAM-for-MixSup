<img src="docs/open_mmlab.png" align="right" width="30%">

# Instructions for OpenPCDet-based MixSup

## Installation

```bash
cd OpenPCDet
conda create -n MixSup python=3.8 -y
conda activate MixSup
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install spconv-cu113
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
python setup.py develop
```

## Prepare Dataset

Firstly, follow official OpenPCDet to prepare data: https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#dataset-preparation . Note that users do not need to install the official OpenPCDet, just following their instructions to prepare data.

### KITTI

Here we provide [the mask file](https://drive.google.com/file/d/1a8lBub-dmOA5X9scCEquSjjnckk5tb1M/view?usp=sharing) for selecting 10% of ground truth boxes to serve as box-level labels. These selected labels also function as the database for CopyPaste augmentation. Please download it and put it in the `data/kitti` folder.

Then, run the following command to generate the database for MixSup:

```bash
python -m pcdet.datasets.kitti.kitti_dataset --func create_kitti_gt_database_with_mask --cfg_file tools/cfgs/dataset_configs/kitti_dataset.yaml --gt_valid_mask_filename kitti_infos_train_valid_mask.pkl --tag 10_percent
```

### NuScenes

Please download [the mask file](https://drive.google.com/file/d/1B682EwSTVWcTNXy1nIbZnTzlhmY06SiF/view?usp=sharing) and put it in the `data/nuscenes/v1.0-trainval` folder.

Then, run the following command to generate the database for MixSup:

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_gt_database_with_mask --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --gt_valid_mask_filename nuscenes_infos_train_valid_mask.pkl --tag 10_percent
```

## Training

```bash
cd tools
bash scripts/dist_train.sh 8 --cfg_file ${cfg_path} --extra_tag mixsup
```

### KITTI

Performances on KITTI validation split with moderate difficulty. $^†$: Using coarse cluster labels and 10% accurate box labels.


|  Detector | Car | Pedestrian | Cyclist | Log |
| ----- | :-----: | :--------: | :-----: | :----: |
| [SECOND](tools/cfgs/kitti_models/second.yaml) (100% frames)     | 78.62   | 52.98       | 67.15   | - |
| [SECOND](tools/cfgs/kitti_models/second_mixed.yaml) (MixSup) $^†$     | 74.85   | 50.18       | 61.46   | [log](https://drive.google.com/file/d/1cqabH0-Pt1Y4yQKpqpSGRuTOYtsLuNsf/view?usp=sharing) |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) (100% frames) | 83.61   | 57.90      | 70.47   | - |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn_mixed.yaml) (MixSup) $^†$ | 76.09   | 54.33      | 65.67   | [log](https://drive.google.com/file/d/13FWtPKBl4fH5hR_VOfIruqi9AMsf3cNN/view?usp=sharing) |

### NuScenes

Performances on nuScenes validation split. $^†$: Using coarse cluster labels and 10% accurate box labels.

|  Detector | mAP | NDS | Log |
| ----- | :-----: | :-----: | :----: |
| [CenterPoint](tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint.yaml) (100% frames)     | 62.41   | 68.20  | - |
| CenterPoint (10% frames)     | 42.19   | 55.38  | - |
| [CenterPoint](tools/cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_mixed.yaml) (MixSup) $^†$     | 60.73   | 66.46  | [log](https://drive.google.com/file/d/1cqabH0-Pt1Y4yQKpqpSGRuTOYtsLuNsf/view?usp=sharing) |

## TODO

Now we have released the code of OpenPCDet-based MixSup on KITTI and NuScenes.
We will release the code on Waymo dataset in around one week.
