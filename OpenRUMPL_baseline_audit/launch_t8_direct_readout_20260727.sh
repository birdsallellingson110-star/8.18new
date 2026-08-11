#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG}
tag=${2:?}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
result_dir="$root/transformer_separate_v2_20260727"

export RUMPL_GATED_JOINT_ADAPTER=1
export RUMPL_JOINT_ADAPTER_INDICES=11
export RUMPL_JOINT_ADAPTER_SCALE_INIT=1.2
export RUMPL_JOINT_ADAPTER_VIEW_POWER=0
export RUMPL_JOINT_ADAPTER_COUNT_LOOKUP=0
export RUMPL_JOINT_ADAPTER_DIRECT_READOUT=1
export RUMPL_JOINT_ADAPTER_DIRECT_SCALE_INIT=0.1
export RUMPL_ADAPTER_ONLY=1
export RUMPL_GLOBAL_ONLY=0
export RUMPL_INIT_CHECKPOINT="$root/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar"
export RUMPL_ALT_JOINT_VIEW=0 RUMPL_VFT_DEPTH=12 RUMPL_PFT_DEPTH=12
export RUMPL_MULTI_HYP=1
export RUMPL_GLOBAL_JOINT_VIEW_FUSION=0
export RUMPL_SINGLEFRAME_GBT=0
export GBT_LEARNABLE_BIAS=0 GBT_ORACLE_RELIABILITY=0
export GBT_LEARNED_RELIABILITY=0
export RUMPL_TRAIN_LR=0.00005
export RUMPL_SAVE_EPOCH_STATES=1

mkdir -p "$result_dir"
echo "T8_START tag=$tag gpu=$gpu time=$(date --iso-8601=seconds)"
bash "$repo/run_official_like_cmu_seed0_20260722.sh" \
  "$gpu" "$tag" 0 1 8

run_dir=$(find \
  "$root/output/multiview_amass_rumpl/multiview_rumpl_999" \
  -maxdepth 1 -type d -name "${tag}_*" -print | sort | tail -n 1)
if [[ -z "$run_dir" ]]; then
  echo "missing run directory for $tag" >&2
  exit 3
fi
for state in best final; do
  if [[ "$state" == best ]]; then
    checkpoint="$run_dir/model_best.pth.tar"
  else
    checkpoint="$run_dir/final_state.pth.tar"
  fi
  printf '%s\n' "$checkpoint" > "$result_dir/${tag}_${state}_checkpoint.txt"
  bash "$repo/eval_joint_adapter_variant_20260727.sh" \
    "$gpu" "${tag}_${state}" "$checkpoint" 11 0 1.2
done
echo "T8_END tag=$tag time=$(date --iso-8601=seconds)"
