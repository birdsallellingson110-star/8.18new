#!/usr/bin/env bash
# Evaluate a depth/anchor model checkpoint on CMU for one view count.
# usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE
#   MODE: d1_anchor | d2_depthaux | d3_combo
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE}
tag=${2:?}
checkpoint=${3:?}
n_views=${4:?}
mode=${5:?}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/depth_anchor_eval

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0 RUMPL_FIX_SCHEDULER_ORDER=1
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=0
export RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0

case "$mode" in
  d1_anchor)   export RUMPL_TRI_ANCHOR=1 ;;
  d2_depthaux) export RUMPL_RAY_DEPTH_AUX=1 ;;
  d3_combo)    export RUMPL_TRI_ANCHOR=1 RUMPL_RAY_DEPTH_AUX=1 ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

output="$root/$tag"
mkdir -p "$output" "$root/log_$tag"
(
  cd "$repo/RUMPL"
  "$python" run/valid_rumpl.py \
    --cfg "$cfg" --gpus 0 --workers 16 --n-views "$n_views" \
    --model-file "$checkpoint" \
    --modelDir "$output" --logDir "$root/log_$tag" \
    --state "best_$tag" --eval-comments "exact_best_$tag" --use-mmpose-val
)
prediction=$(find "$output" -type f -name '*_dict.pkl' -print | sort | tail -n 1)
"$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" \
  "$prediction" --n-views "$n_views" --output-json "$root/${tag}_summary.json"
