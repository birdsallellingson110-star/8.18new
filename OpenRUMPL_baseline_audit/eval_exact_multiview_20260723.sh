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
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/multiview_model_best_eval

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=${RUMPL_FIX_PFT_LAST_BLOCK:-0}
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0
export GBT_LEARNED_RELIABILITY=0
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_GLOBAL_JOINT_VIEW_DEPTH=2
export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=0
export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=0
export RUMPL_GLOBAL_JOINT_VIEW_GEOM_NORM=0
export RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT=0.1
export RUMPL_GLOBAL_JOINT_VIEW_COUNT_GATE=0
export RUMPL_GLOBAL_JOINT_VIEW_GATE_MAX_INIT=0.12
export RUMPL_SINGLEFRAME_GBT=0
export RUMPL_SF_GBT_ENCODER_DEPTH=3
export RUMPL_SF_GBT_DECODER_DEPTH=2
export RUMPL_SF_GBT_PFT_DEPTH=4
export RUMPL_SF_GBT_CONF_BIAS=0
export RUMPL_SF_GBT_GEOM_BIAS=0
export RUMPL_SF_GBT_GEOM_NORM=0
export RUMPL_SYMMETRY_LOSS_WEIGHT=0.0

if [[ "$mode" == learned ]]; then
  export GBT_LEARNED_RELIABILITY=1
  export GBT_RELIABILITY_GATE_INIT=0.02
  export GBT_RELIABILITY_TEMPERATURE=0.2
  export GBT_RELIABILITY_PAIR_GATE=0
elif [[ "$mode" == global_biased || "$mode" == global_biased_norm || "$mode" == global_conf || "$mode" == global_plain ]]; then
  export RUMPL_GLOBAL_JOINT_VIEW_FUSION=1
  export RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT=0.1
  if [[ "$mode" == global_biased || "$mode" == global_biased_norm ]]; then
    export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=1
    export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=1
    if [[ "$mode" == global_biased_norm ]]; then
      export RUMPL_GLOBAL_JOINT_VIEW_GEOM_NORM=1
    fi
  elif [[ "$mode" == global_conf ]]; then
    export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=1
    export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=0
  else
    export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=0
    export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=0
  fi
elif [[ "$mode" == j2_adaptive ]]; then
  export RUMPL_GLOBAL_JOINT_VIEW_FUSION=1
  export RUMPL_GLOBAL_JOINT_VIEW_DEPTH=2
  export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=0
  export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=0
  export RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT=0.05
  export RUMPL_GLOBAL_JOINT_VIEW_COUNT_GATE=1
  export RUMPL_GLOBAL_JOINT_VIEW_GATE_MAX_INIT=0.12
elif [[ "$mode" == sf_plain || "$mode" == sf_biased_norm ]]; then
  export RUMPL_SINGLEFRAME_GBT=1
  export RUMPL_SF_GBT_ENCODER_DEPTH=3
  export RUMPL_SF_GBT_DECODER_DEPTH=2
  export RUMPL_SF_GBT_PFT_DEPTH=4
  if [[ "$mode" == sf_biased_norm ]]; then
    export RUMPL_SF_GBT_CONF_BIAS=1
    export RUMPL_SF_GBT_GEOM_BIAS=1
    export RUMPL_SF_GBT_GEOM_NORM=1
  fi
elif [[ "$mode" == baseline ]]; then
  :
elif [[ "$mode" != baseline ]]; then
  echo "unknown mode: $mode" >&2
  exit 2
fi

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
