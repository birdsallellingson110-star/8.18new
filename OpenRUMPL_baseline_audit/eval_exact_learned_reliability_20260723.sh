#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG CHECKPOINT GATE_INIT TEMPERATURE [PAIR_GATE]}
tag=${2:?usage: $0 GPU TAG CHECKPOINT GATE_INIT TEMPERATURE [PAIR_GATE]}
checkpoint=${3:?usage: $0 GPU TAG CHECKPOINT GATE_INIT TEMPERATURE [PAIR_GATE]}
gate_init=${4:?usage: $0 GPU TAG CHECKPOINT GATE_INIT TEMPERATURE [PAIR_GATE]}
temperature=${5:?usage: $0 GPU TAG CHECKPOINT GATE_INIT TEMPERATURE [PAIR_GATE]}
pair_gate=${6:-0}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
root=/mnt/data/cjyoutput/baseline_reaudit_20260722/learned_reliability_model_best_eval

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0 GBT_LEARNED_RELIABILITY=1
export GBT_RELIABILITY_GATE_INIT="$gate_init"
export GBT_RELIABILITY_TEMPERATURE="$temperature"
export GBT_RELIABILITY_PAIR_GATE="$pair_gate"

output="$root/$tag"
mkdir -p "$output" "$root/log_$tag"
(
  cd "$repo/RUMPL"
  "$python" run/valid_rumpl.py \
    --cfg "$cfg" --gpus 0 --workers 16 \
    --model-file "$checkpoint" \
    --modelDir "$output" --logDir "$root/log_$tag" \
    --state "best_$tag" --eval-comments "exact_best_$tag" --use-mmpose-val
)
prediction=$(find "$output" -type f -name '*_dict.pkl' -print | sort | tail -n 1)
"$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" \
  "$prediction" --output-json "$root/${tag}_summary.json"
