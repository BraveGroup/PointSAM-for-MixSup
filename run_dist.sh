CONFIG=configs/cfg_PointSAM.py
SAMPLE_INDICES=scripts/indices_val.npy
OUT_DIR=PointSAM_val
NUM_GPUS=8

# Firstly perform instance segmentation on 2D images
bash scripts/dist_segment2D.sh ${CONFIG} ${NUM_GPUS} --sample_indices ${SAMPLE_INDICES} --out_dir ${OUT_DIR}
# Then project to 3D point clouds and utilize Separability-Aware Refinement to generate instance segmentation masks
bash scripts/dist_segment3D.sh ${CONFIG} ${NUM_GPUS} --mask_root ${OUT_DIR} --sample_indices ${SAMPLE_INDICES} --out_dir ${OUT_DIR} --for_eval

# If not specify for_eval, the convert_results.py is needed
# bash scripts/dist_segment3D.sh ${CONFIG} ${NUM_GPUS} --mask_root ${OUT_DIR} --sample_indices ${SAMPLE_INDICES} --out_dir ${OUT_DIR}
# python scripts/convert_results.py --mode bin2npz --input ${OUT_DIR}/samples/ --output ${OUT_DIR}/panoptic/val/

python -m nuscenes.eval.panoptic.evaluate --result_path ${OUT_DIR} --eval_set val --dataroot data/nuscenes --version v1.0-trainval --out_dir ${OUT_DIR}
python scripts/parse_json.py ${OUT_DIR}/segmentation-result.json