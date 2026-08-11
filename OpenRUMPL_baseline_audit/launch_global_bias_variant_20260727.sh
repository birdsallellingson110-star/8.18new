#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG MODE}
tag=${2:?}
mode=${3:?global_conf or global_biased_norm}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
result_dir="$root/transformer_clean_r5_20260726"

export RUMPL_GLOBAL_JOINT_VIEW_FUSION=1
export RUMPL_GLOBAL_JOINT_VIEW_DEPTH=2
export RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT=0.1
export RUMPL_GLOBAL_JOINT_VIEW_COUNT_GATE=0
export RUMPL_GLOBAL_JOINT_VIEW_GATE_MAX_INIT=0.12
export GBT_LEARNABLE_BIAS=0 GBT_LEARNED_RELIABILITY=0

case "$mode" in
  global_conf)
    export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=1
    export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=0
    export RUMPL_GLOBAL_JOINT_VIEW_GEOM_NORM=0
    ;;
  global_biased_norm)
    export RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS=1
    export RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS=1
    export RUMPL_GLOBAL_JOINT_VIEW_GEOM_NORM=1
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

echo "GLOBAL_BIAS_START tag=$tag mode=$mode gpu=$gpu time=$(date --iso-8601=seconds)"
bash "$repo/run_official_like_cmu_seed0_20260722.sh" \
  "$gpu" "$tag" 0 1 16
checkpoint=$(find \
  "$root/output/multiview_amass_rumpl/multiview_rumpl_999" \
  -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
  -print | sort | tail -n 1)
if [[ -z "$checkpoint" ]]; then
  echo "missing checkpoint for $tag" >&2
  exit 3
fi
printf '%s\n' "$checkpoint" > "$result_dir/${tag}_checkpoint.txt"
for n_views in 2 3 4 5; do
  bash "$repo/eval_exact_multiview_20260723.sh" \
    "$gpu" "${tag}_mmpose_v${n_views}" "$checkpoint" "$n_views" "$mode"
done
echo "GLOBAL_BIAS_END tag=$tag time=$(date --iso-8601=seconds)"
