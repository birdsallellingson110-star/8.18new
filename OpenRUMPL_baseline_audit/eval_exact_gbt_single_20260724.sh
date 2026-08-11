#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE}
tag=${2:?usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE}
checkpoint=${3:?usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE}
n_views=${4:?usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE}
mode=${5:?usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/gbt_multiview_eval

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0 RUMPL_FIX_SCHEDULER_ORDER=1
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_CONF_INIT=0.1 GBT_GEOM_INIT=0.1

case "$mode" in
  g0_formula)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
    export GBT_FUSION_GEOM=0 GBT_GEOM_INIT=1.0
    ;;
  g1_fusion)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
    export GBT_FUSION_GEOM=1 GBT_GEOM_INIT=0.1
    ;;
  g4_fusion)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
    export GBT_FUSION_GEOM=1 GBT_GEOM_INIT=0.05
    ;;
  g2_conf_only)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=0
    ;;
  g3_geom_fusion)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=1
    export GBT_FUSION_GEOM=1 GBT_GEOM_INIT=0.1
    ;;
  *)
    echo "unknown gbt mode: $mode" >&2
    exit 2
    ;;
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
