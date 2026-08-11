#!/usr/bin/env bash
# Sequential train+evaluation suite on one GPU.
# All persistent artifacts live under /mnt/data/cjyoutput.
set -euo pipefail

gpu=${1:?usage: $0 GPU}
repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
suite="$root/transformer_clean_r5_20260726"
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
mkdir -p "$suite"

modes=(t0_fix t1_shallow t2_alt)
tags=(
  T0_fixpft_r5proto_seed0_20260726
  T1_vft4_pft8_r5proto_seed0_20260726
  T2_altjv4_pft4_r5proto_seed0_20260726
)

echo "SUITE_START $(date --iso-8601=seconds) gpu=$gpu"
for index in "${!modes[@]}"; do
  mode=${modes[$index]}
  tag=${tags[$index]}
  echo "TRAIN_START $(date --iso-8601=seconds) mode=$mode tag=$tag"
  bash "$repo/run_transformer_ablation_20260726.sh" "$gpu" "$mode" "$tag"

  checkpoint=$(find \
    "$root/output/multiview_amass_rumpl/multiview_rumpl_999" \
    -maxdepth 2 -type f -path "*${tag}*/model_best.pth.tar" \
    -print | sort | tail -n 1)
  if [[ -z "$checkpoint" ]]; then
    echo "missing model_best checkpoint for $tag" >&2
    exit 3
  fi
  printf '%s\n' "$checkpoint" > "$suite/${tag}_checkpoint.txt"

  echo "EVAL_START $(date --iso-8601=seconds) checkpoint=$checkpoint"
  bash "$repo/eval_transformer_ablation_20260726.sh" \
    "$gpu" "$mode" "$tag" "$checkpoint"

  "$python" "$repo/summarize_transformer_suite_20260726.py" \
    --root "$root" --output "$suite/RESULTS_partial.json" \
    --tags "${tags[@]}"
  echo "CANDIDATE_END $(date --iso-8601=seconds) tag=$tag"
done

"$python" "$repo/summarize_transformer_suite_20260726.py" \
  --root "$root" --output "$suite/RESULTS_final.json" \
  --tags "${tags[@]}"
echo "SUITE_END $(date --iso-8601=seconds)"
