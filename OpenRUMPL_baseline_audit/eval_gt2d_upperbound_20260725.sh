#!/usr/bin/env bash
# GT-2D / org upper bound vs mmpose on R5 (no training).
# Usage: $0 GPU TAG CHECKPOINT N_VIEWS [mmpose|gt2d]
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG CHECKPOINT N_VIEWS [mmpose|gt2d]}
tag=${2:?}
checkpoint=${3:?}
n_views=${4:?}
mode=${5:-gt2d}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/gt2d_upperbound

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=${RUMPL_FIX_PFT_LAST_BLOCK:-0}
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_SYMMETRY_LOSS_WEIGHT=0.0
export RUMPL_KPA=0 RUMPL_MULTI_HYP=0 RUMPL_POSE_CODEBOOK=0
export RUMPL_2D_REFINE=0
export RUMPL_TRAIN_STRUCT_OCC=0 RUMPL_OCC_JOINT_LOSS=0

if [[ "$mode" == gt2d ]]; then
  mmpose_flag=(--not-use-mmpose-val)
elif [[ "$mode" == mmpose ]]; then
  mmpose_flag=(--use-mmpose-val)
else
  echo "unknown mode: $mode (want mmpose|gt2d)" >&2
  exit 2
fi

output="$root/$tag"
mkdir -p "$output" "$root/log_$tag"
echo "[gt2d-ub] tag=$tag mode=$mode v=$n_views gpu=$gpu"
(
  cd "$repo/RUMPL"
  "$python" run/valid_rumpl.py \
    --cfg "$cfg" --gpus 0 --workers 16 --n-views "$n_views" \
    --model-file "$checkpoint" \
    --modelDir "$output" --logDir "$root/log_$tag" \
    --state "best_$tag" --eval-comments "ub_${mode}_$tag" \
    "${mmpose_flag[@]}"
)
prediction=$(find "$output" -type f -name '*_dict.pkl' -print | sort | tail -n 1)
"$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" \
  "$prediction" --n-views "$n_views" --output-json "$root/${tag}_summary.json"
echo "[gt2d-ub] wrote $root/${tag}_summary.json"
