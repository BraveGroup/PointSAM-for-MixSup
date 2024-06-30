<img src="docs/open_mmlab.png" align="right" width="30%">

# Instructions for OpenPCDet-based MixSup

## Installation

```bash
cd OpenPCDet
conda create -n MixSup pyhton=3.8 -y
conda activate MixSup
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install spconv-cu113
pip install https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
python setup.py develop
```

## Prepare Dataset

Firstly, follow official OpenPCDet to prepare data: https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#dataset-preparation . Note that users do not need to install the official OpenPCDet, just following their instructions to prepare data.

### NuScenes

Here we provide [the mask file](https://drive.google.com/file/d/1B682EwSTVWcTNXy1nIbZnTzlhmY06SiF/view?usp=sharing) for selecting 10% of ground truth boxes to serve as box-level labels. These selected labels also function as the database for CopyPaste augmentation. Please download it and put it in the `data/nuscenes/v1.0-trainval` folder.

Then, run the following command to generate the database for MixSup:

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_gt_database_with_mask --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval --gt_valid_mask_filename nuscenes_infos_train_valid_mask.pkl --tag 10_percent
```

## Training

To train the CenterPoint with MixSup on NuScenes, run the following command:

```bash
cd tools
bash scripts/dist_train.sh 8 --cfg_file cfgs/nuscenes_models/cbgs_voxel0075_res3d_centerpoint_mixed.yaml --extra_tag mixsup
```

## TODO

Now we have released the code of OpenPCDet-based MixSup on NuScenes.
We will release the code on KITTI and Waymo datasets in around one week.