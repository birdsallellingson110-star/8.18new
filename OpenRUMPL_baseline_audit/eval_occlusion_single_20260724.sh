#!/usr/bin/env bash
# Evaluate a checkpoint on CMU with test-time structured 2D occlusion (limb dropout per view).
# usage: $0 GPU TAG CHECKPOINT N_VIEWS MODE OCC_LEVEL
#   MODE: baseline | d2_depthaux | d1_anchor
set -euo pipefail

gpu=${1:?}
tag=${2:?}
checkpoint=${3:?}
n_views=${4:?}
mode=${5:?}
occ=${6:-0}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=${RUMPL_FIX_PFT_LAST_BLOCK:-0}
export RUMPL_FIX_SCHEDULER_ORDER=1
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0

case "$mode" in
  baseline) ;;
  d1_anchor)   export RUMPL_TRI_ANCHOR=1 ;;
  d2_depthaux) export RUMPL_RAY_DEPTH_AUX=1 ;;
  adafuse_vw)  export RUMPL_ADAFUSE_VW=1 RUMPL_ADAFUSE_VW_MIX=0.0 ;;
  pose_codebook) export RUMPL_POSE_CODEBOOK=1 ;;
  kpa)         export RUMPL_KPA=1 ;;
  mh3)         export RUMPL_MULTI_HYP=3 ;;
  conf_film)   export RUMPL_CONF_FILM=1 ;;
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
    --state "best_$tag" --eval-comments "occ_$tag" --use-mmpose-val \
    --test-occlusion-level "$occ" --test-occlusion-mode limb
)
prediction=$(find "$output" -type f -name '*_dict.pkl' -print | sort | tail -n 1)
"$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" \
  "$prediction" --n-views "$n_views" --output-json "$root/${tag}_summary.json"
