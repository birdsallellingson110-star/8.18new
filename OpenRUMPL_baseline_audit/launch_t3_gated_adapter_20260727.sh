#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
tag=T3_gated_joint_adapter_r5init_seed0_20260727
result_dir="$root/transformer_clean_r5_20260726"

echo "T3_START $(date --iso-8601=seconds)"
bash "$repo/run_t3_gated_joint_adapter_20260727.sh" "$gpu" "$tag"
checkpoint=$(find \
  "$root/output/multiview_amass_rumpl/multiview_rumpl_999" \
  -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
  -print | sort | tail -n 1)
if [[ -z "$checkpoint" ]]; then
  echo "missing T3 checkpoint" >&2
  exit 3
fi
printf '%s\n' "$checkpoint" > "$result_dir/${tag}_checkpoint.txt"
bash "$repo/eval_t3_gated_joint_adapter_20260727.sh" \
  "$gpu" "$tag" "$checkpoint"
/home/lixiaob/cjy/rumpl_venv310/bin/python \
  "$repo/summarize_transformer_suite_20260726.py" \
  --root "$root" --output "$result_dir/RESULTS_with_T3.json" \
  --tags \
    T0_fixpft_r5proto_seed0_20260726 \
    T1_vft4_pft8_r5proto_seed0_20260726 \
    T2_altjv4_pft4_r5proto_seed0_20260726 \
    "$tag"
echo "T3_END $(date --iso-8601=seconds)"
