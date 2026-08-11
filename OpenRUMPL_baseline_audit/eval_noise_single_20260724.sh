#!/usr/bin/env bash
# Robustness eval: evaluate a checkpoint on CMU with optional camera-calibration noise.
# usage: eval_noise_single_20260724.sh GPU TAG CHECKPOINT N_VIEWS MODE ROT_DEG TRANS_STD
#   MODE: baseline | g2_conf_only | g3_geom_fusion | g4_fusion | g1_fusion
#   ROT_DEG/TRANS_STD: if ROT_DEG>0, test-time camera-calib noise is applied.
set -euo pipefail

gpu=${1:?}
tag=${2:?}
checkpoint=${3:?}
n_views=${4:?}
mode=${5:?}
rot_deg=${6:-0}
trans_std=${7:-0}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/noise_robustness_eval

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0 RUMPL_FIX_SCHEDULER_ORDER=1
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=0
export GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=0 GBT_FUSION_GEOM=0
export GBT_CONF_INIT=0.1 GBT_GEOM_INIT=0.1
export RUMPL_TRI_ANCHOR=0 RUMPL_RAY_DEPTH_AUX=0

case "$mode" in
  baseline) ;;
  d1_anchor)   export RUMPL_TRI_ANCHOR=1 ;;
  d2_depthaux) export RUMPL_RAY_DEPTH_AUX=1 ;;  # build head so ckpt loads; skipped at inference (is_training=False)
  g1_fusion)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
    export GBT_FUSION_GEOM=1 GBT_GEOM_INIT=0.1 ;;
  g4_fusion)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=1
    export GBT_FUSION_GEOM=1 GBT_GEOM_INIT=0.05 ;;
  g2_conf_only)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=1 GBT_USE_GEOM_BIAS=0 ;;
  g3_geom_fusion)
    export GBT_LEARNABLE_BIAS=1 GBT_USE_CONF_BIAS=0 GBT_USE_GEOM_BIAS=1
    export GBT_FUSION_GEOM=1 GBT_GEOM_INIT=0.1 ;;
  *) echo "unknown mode: $mode" >&2; exit 2 ;;
esac

noise_args=()
if [ "$(echo "$rot_deg > 0" | bc -l)" = "1" ]; then
  noise_args=(--test-add-noise-to-camera-calib --noise-rot-deg "$rot_deg" --noise-trans-std "$trans_std")
fi

output="$root/$tag"
mkdir -p "$output" "$root/log_$tag"
(
  cd "$repo/RUMPL"
  "$python" run/valid_rumpl.py \
    --cfg "$cfg" --gpus 0 --workers 16 --n-views "$n_views" \
    --model-file "$checkpoint" \
    --modelDir "$output" --logDir "$root/log_$tag" \
    --state "best_$tag" --eval-comments "noise_$tag" --use-mmpose-val \
    "${noise_args[@]}"
)
prediction=$(find "$output" -type f -name '*_dict.pkl' -print | sort | tail -n 1)
"$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" \
  "$prediction" --n-views "$n_views" --output-json "$root/${tag}_summary.json"
