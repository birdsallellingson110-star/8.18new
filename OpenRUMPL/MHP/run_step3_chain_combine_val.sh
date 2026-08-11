#!/usr/bin/env bash
# 链式: 等 val 渲染完 (stage_V/validation 99 pkl) → 合并 val → 软链进 official_combined
set -uo pipefail
VAL_SV=/mnt/data/cjydata/mhp_workspace/paper_single_cmu/stage_V/validation
DS=/mnt/data/cjydata/mhp_workspace/paper_single_cmu/datasets
COMBINED="$DS/official_combined"
LOG=/mnt/data/cjydata/step3_chain_combine_val.log

echo "[chain-combine-val] 等待 val 渲染完成 (99 pkl)... $(date '+%F %T')" >> "$LOG"
while true; do
  n=$(ls "$VAL_SV"/*.pkl 2>/dev/null | grep -v temp | wc -l)
  if [ "$n" -ge 99 ]; then break; fi
  sleep 120
done
echo "[chain-combine-val] val 渲染完成 ($n pkl)。开始合并 $(date '+%F %T')" >> "$LOG"

bash /home/lixiaob/cjy/OpenRUMPL/MHP/run_step3_combine.sh validation >> "$LOG" 2>&1

VAL_DIR="$DS/_random_20_small_room_person_dist_2_amass_mmpose_joints_validation_dome_coco_calibs_171204_pose5_171204_pose6_no_fit_rotated_triangulated__random_cameras_20_"
ln -sf "$VAL_DIR/amass_mmpose_joints_validation.pkl" "$COMBINED/amass_mmpose_joints_validation.pkl"
echo "[chain-combine-val] 完成。official_combined 内容:" >> "$LOG"
ls -la "$COMBINED/" >> "$LOG" 2>&1
echo "[chain-combine-val] 训练集就绪, 可开训: python run/train_rumpl.py --cfg configs/cmu_panoptic/rumpl_amass/clip_full.yaml  $(date '+%F %T')" >> "$LOG"
