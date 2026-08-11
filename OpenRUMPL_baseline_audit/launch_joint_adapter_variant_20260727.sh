#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TAG INDICES VIEW_POWER [SCALE_INIT]}
tag=${2:?}
indices=${3:?}
view_power=${4:?}
scale_init=${5:-1.2}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
result_dir="$root/transformer_clean_r5_20260726"

echo "VARIANT_START tag=$tag gpu=$gpu indices=$indices power=$view_power scale=$scale_init time=$(date --iso-8601=seconds)"
bash "$repo/run_joint_adapter_variant_20260727.sh" \
  "$gpu" "$tag" "$indices" "$view_power" "$scale_init"
checkpoint=$(find \
  "$root/output/multiview_amass_rumpl/multiview_rumpl_999" \
  -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
  -print | sort | tail -n 1)
if [[ -z "$checkpoint" ]]; then
  echo "missing checkpoint for $tag" >&2
  exit 3
fi
printf '%s\n' "$checkpoint" > "$result_dir/${tag}_checkpoint.txt"
bash "$repo/eval_joint_adapter_variant_20260727.sh" \
  "$gpu" "$tag" "$checkpoint" "$indices" "$view_power" "$scale_init"
echo "VARIANT_END tag=$tag time=$(date --iso-8601=seconds)"
