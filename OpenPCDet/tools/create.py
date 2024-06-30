import yaml
import argparse
from pathlib import Path
from easydict import EasyDict
from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset, create_nuscenes_info
from pcdet.datasets.waymo.waymo_dataset import WaymoDataset, create_waymo_infos, create_waymo_gt_database, create_waymo_gt_database_with_mask
from pcdet.datasets.kitti.kitti_dataset import KittiDataset, create_kitti_infos, create_kitti_gt_database_with_mask
from pcdet.utils import common_utils

# if __name__ == '__main__':
#     import argparse
#     import yaml
#     from easydict import EasyDict

#     parser = argparse.ArgumentParser(description='arg parser')
#     parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
#     parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
#     parser.add_argument('--processed_data_tag', type=str, default='waymo_processed_data_v0_5_0', help='')
#     parser.add_argument('--update_info_only', action='store_true', default=False, help='')
#     parser.add_argument('--use_parallel', action='store_true', default=False, help='')
#     parser.add_argument('--wo_crop_gt_with_tail', action='store_true', default=False, help='')
#     parser.add_argument('--train_filename', type=str, default=None, help='used in create_waymo_gt_database_with_mask')
#     parser.add_argument('--gt_valid_mask_filename', type=str, default=None, help='used in create_waymo_gt_database_with_mask')

#     args = parser.parse_args()

#     ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()

#     if args.func == 'create_waymo_infos':
#         try:
#             yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
#         except:
#             yaml_config = yaml.safe_load(open(args.cfg_file))
#         dataset_cfg = EasyDict(yaml_config)
#         dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
#         create_waymo_infos(
#             dataset_cfg=dataset_cfg,
#             class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
#             data_path=ROOT_DIR / 'data' / 'waymo',
#             save_path=ROOT_DIR / 'data' / 'waymo',
#             raw_data_tag='raw_data',
#             processed_data_tag=args.processed_data_tag,
#             update_info_only=args.update_info_only
#         )
#     elif args.func == 'create_waymo_gt_database':
#         try:
#             yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
#         except:
#             yaml_config = yaml.safe_load(open(args.cfg_file))
#         dataset_cfg = EasyDict(yaml_config)
#         dataset_cfg.PROCESSED_DATA_TAG = args.processed_data_tag
#         create_waymo_gt_database(
#             dataset_cfg=dataset_cfg,
#             class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
#             data_path=ROOT_DIR / 'data' / 'waymo',
#             save_path=ROOT_DIR / 'data' / 'waymo',
#             processed_data_tag=args.processed_data_tag,
#             use_parallel=args.use_parallel, 
#             crop_gt_with_tail=not args.wo_crop_gt_with_tail
#         )
#     elif args.func == 'create_waymo_gt_database_with_mask':
#         try:
#             yaml_config = yaml.safe_load(open(args.cfg_file), Loader=yaml.FullLoader)
#         except:
#             yaml_config = yaml.safe_load(open(args.cfg_file))
#         dataset_cfg = EasyDict(yaml_config)
#         create_waymo_gt_database_with_mask(
#             dataset_cfg=dataset_cfg,
#             class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
#             data_path=ROOT_DIR / 'data' / 'waymo',
#             save_path=ROOT_DIR / 'data' / 'waymo',
#             train_filename=args.train_filename,
#             gt_valid_mask_filename=args.gt_valid_mask_filename,
#             processed_data_tag=args.processed_data_tag
#         )
#     else:
#         raise NotImplementedError




# parser = argparse.ArgumentParser(description='arg parser')
# parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
# parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
# parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
# parser.add_argument('--gt_valid_mask_filename', type=str, default=None, help='used in create_nuscenes_gt_database_with_mask')
# parser.add_argument('--tag', type=str, default='', help='used in create_nuscenes_gt_database_with_mask')
# args = parser.parse_args()
# ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()

# # if args.func == 'create_nuscenes_infos':
# #     dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
# #     dataset_cfg.VERSION = args.version
# #     create_nuscenes_info(
# #         version=dataset_cfg.VERSION,
# #         data_path=ROOT_DIR / 'data' / 'nuscenes',
# #         save_path=ROOT_DIR / 'data' / 'nuscenes',
# #         max_sweeps=dataset_cfg.MAX_SWEEPS,
# #     )

# #     nuscenes_dataset = NuScenesDataset(
# #         dataset_cfg=dataset_cfg, class_names=None,
# #         root_path=ROOT_DIR / 'data' / 'nuscenes',
# #         logger=common_utils.create_logger(), training=True
# #     )
# #     nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
# # elif args.func == 'create_nuscenes_gt_database_with_mask':
# #     dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
# #     dataset_cfg.VERSION = args.version
# #     nuscenes_dataset = NuScenesDataset(
# #         dataset_cfg=dataset_cfg, class_names=None,
# #         root_path=ROOT_DIR / 'data' / 'nuscenes',
# #         logger=common_utils.create_logger(), training=True
# #     )
# #     nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS,
# #                                 gt_valid_mask_filename=args.gt_valid_mask_filename, tag=args.tag)

if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_kitti_infos', help='')
    parser.add_argument('--gt_valid_mask_filename', type=str, default=None, help='used in create_kitti_gt_database_with_mask')
    parser.add_argument('--tag', type=str, default='', help='used in create_kitti_gt_database_with_mask')

    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    args = parser.parse_args()
    if args.func == 'create_kitti_infos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
    elif args.func == 'create_kitti_gt_database_with_mask':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        create_kitti_gt_database_with_mask(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti',
            gt_valid_mask_filename=args.gt_valid_mask_filename,
            tag=args.tag
        )
