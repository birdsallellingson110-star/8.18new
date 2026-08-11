#!/usr/bin/env bash
# Step3: 合并 stage_V 的 99 个 split pkl 成训练/验证集 (官方 run_mmpose_04_combine.py)
# 用法: bash run_step3_combine.sh [subset=train]   (subset = train | validation)
set -euo pipefail
SUBSET=${1:-train}
source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh >/dev/null 2>&1
cd /home/lixiaob/cjy/OpenRUMPL/MHP

WORK=/mnt/data/cjydata/mhp_workspace
SV=$WORK/paper_single_cmu/stage_V/$SUBSET

n=$(ls $SV/*.pkl 2>/dev/null | grep -v temp | wc -l)
echo "stage_V/$SUBSET final pkl 数量: $n (需要 99)"
if [ "$n" -ne 99 ]; then
  echo "!! $SUBSET 还没齐 99 个, 先等生成完。"
  exit 1
fi

python -u run_mmpose_04_combine.py \
  --exp paper_single_cmu \
  --extra-name random_20_small_room_person_dist_2 \
  --n-splits 100 \
  --operation-on "$SUBSET" \
  --work-dir "$WORK"

echo
echo "=== $SUBSET 合并完成。输出: ==="
find $WORK/paper_single_cmu/datasets -name "amass_mmpose_joints_${SUBSET}.pkl"
