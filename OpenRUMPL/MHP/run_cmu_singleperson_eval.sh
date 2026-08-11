#!/usr/bin/env bash
# 单人 CMU(171204_pose5/6) 评估数据全流程 (官方一致):
#   抽帧 -> preprocess+filter(建GT) -> HRNet -> mmpose匹配 -> 坐标swap
# 前置: 视频+hdPose3d已下到 /mnt/data/cjydata/cmu_singleperson/<seq>/
set -e
cd /home/lixiaob/cjy/OpenRUMPL
source env_rumpl.sh >/dev/null 2>&1
export TORCH_HOME=/mnt/data/dataset/c2i/torch CUDA_VISIBLE_DEVICES=0

CMU=/mnt/data/cjydata/cmu_singleperson
DSNAME=annot_pose56_5cams_coco
SEQS="171204_pose5 171204_pose6"
CAMS="3 6 12 13 23"
VPY=/home/lixiaob/cjy/rumpl_venv310/bin/python

echo "### 1. 抽帧 (opencv, 每64帧, GT对齐) ###"
$VPY MHP/extract_frames_cmu.py --root $CMU --seqs 171204_pose5 --cams 03 06 12 13 23 --skip 64 &
$VPY MHP/extract_frames_cmu.py --root $CMU --seqs 171204_pose6 --cams 03 06 12 13 23 --skip 64 &
wait

echo "### 2. preprocess + filter (建GT数据集) ###"
cd RUMPL/data
$VPY preprocess_cmu_panoptic.py $CMU $DSNAME \
  --running-modes preprocess filter --run-for-sets validation \
  --val-cats $SEQS \
  --skip-step-train 1 --skip-step-val 1 --keypoints-standard coco

echo "### 3. HRNet 跑单人 2D (双卡: pose5->GPU0, pose6->GPU1) ###"
cd /home/lixiaob/cjy/OpenRUMPL
CUDA_VISIBLE_DEVICES=0 $VPY MHP/run_hrnet_cmu.py --root $CMU --out $CMU/MPL_data/mmpose_outputs --seqs 171204_pose5 --cams 03 06 12 13 23 &
CUDA_VISIBLE_DEVICES=1 $VPY MHP/run_hrnet_cmu.py --root $CMU --out $CMU/MPL_data/mmpose_outputs --seqs 171204_pose6 --cams 03 06 12 13 23 &
wait
echo "  双卡 HRNet 完成"

echo "### 4. preprocess mmpose 模式 (匹配HRNet到GT, 单人平凡) ###"
cd RUMPL/data
$VPY preprocess_cmu_panoptic.py $CMU $DSNAME \
  --running-modes mmpose --run-for-sets validation \
  --val-cats $SEQS \
  --skip-step-train 1 --skip-step-val 1 --keypoints-standard coco \
  --mmpose-dataset-name mmpose_hrnet_coco_matched \
  --mmpose-output-path $CMU/MPL_data/mmpose_outputs

echo "### 5. 坐标轴 swap (CMU y-down -> z-up) ###"
cd /home/lixiaob/cjy/OpenRUMPL
BASE=$CMU/MPL_data/datasets_mmpose/${DSNAME}_filtered_1_1_mmpose_hrnet_coco_matched
$VPY MHP/06_swap_cmu_axes.py \
  --in-pkl  $BASE/cmu_panoptic_validation.pkl \
  --out-pkl ${BASE}_swapv3/cmu_panoptic_validation.pkl

echo "### 6. 接线 ROOT 符号链接 ###"
mkdir -p /mnt/data/cjydata/cmu_singleperson_eval
ln -sfn $CMU/MPL_data /mnt/data/cjydata/cmu_singleperson_eval/data
echo "=== 完成! 训练后评估config需设: ==="
echo "  ROOT: /mnt/data/cjydata/cmu_singleperson_eval/"
echo "  TEST_CMU_DATASET_NAME: ${DSNAME}_filtered_1_1"
echo "  TEST_MMPOSE_TYPE: mmpose_hrnet_coco_matched_swapv3"
echo "  TEST_CMU_CALIB: 171204_pose5, 171204_pose6"
echo "  TEST_VIEWS: 3 6 12 13 23 (V=5) / 3 6 (V=2)"
