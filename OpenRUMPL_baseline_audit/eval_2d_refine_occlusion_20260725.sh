#!/usr/bin/env bash
# Eval R5 (or any ckpt) with test-time RUMPL_2D_REFINE on CMU occlusion axis.
# usage: $0 GPU TAG CHECKPOINT N_VIEWS OCC_LEVEL [MODE] [STRENGTH] [FILL_CONF]
set -euo pipefail

gpu=${1:?}
tag=${2:?}
checkpoint=${3:?}
n_views=${4:?}
occ=${5:-0.0}
mode=${6:-soft_fill}
strength=${7:-0.5}
fill_conf=${8:-0.35}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0 RUMPL_FIX_SCHEDULER_ORDER=1
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0

export RUMPL_2D_REFINE=1
export RUMPL_2D_REFINE_MODE="$mode"
export RUMPL_2D_REFINE_STRENGTH="$strength"
export RUMPL_2D_REFINE_FILL_CONF="$fill_conf"
export RUMPL_2D_REFINE_CONF_THR=0.1
export RUMPL_2D_REFINE_MIN_VIEWS=2

output="$root/$tag"
mkdir -p "$output" "$root/log_$tag"
echo "[2d-refine] tag=$tag mode=$mode strength=$strength fill_conf=$fill_conf occ=$occ v=$n_views"
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
echo "[done] $root/${tag}_summary.json"
