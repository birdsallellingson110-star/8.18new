#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU TEMPERATURE_MM [TEMPERATURE_MM ...]}
shift

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
python=/home/lixiaob/cjy/rumpl_venv310/bin/python
cfg=../RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml
checkpoint=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar
root=/mnt/data/cjyoutput/oracle_reliability_20260723
if [[ "${GBT_COARSE_RELIABILITY:-0}" == "1" ]]; then
  root=/mnt/data/cjyoutput/coarse_reliability_20260723
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0
export GBT_LEARNABLE_BIAS=0
export GBT_ORACLE_RELIABILITY=1
export GBT_COARSE_RELIABILITY="${GBT_COARSE_RELIABILITY:-0}"

mkdir -p "$root"
for temperature_mm in "$@"; do
  tag="oracle_t${temperature_mm}mm"
  export GBT_ORACLE_TEMPERATURE
  GBT_ORACLE_TEMPERATURE=$("$python" -c "print(float('${temperature_mm}') / 1000.0)")
  output="$root/$tag"
  log_root="$root/log_$tag"
  mkdir -p "$output" "$log_root"
  {
    echo "START $(date --iso-8601=seconds) gpu=$gpu temperature_mm=$temperature_mm"
    cd "$repo/RUMPL"
    "$python" run/valid_rumpl.py \
      --cfg "$cfg" \
      --gpus 0 \
      --workers 16 \
      --model-file "$checkpoint" \
      --modelDir "$output" \
      --logDir "$log_root" \
      --state "best_$tag" \
      --eval-comments "$tag" \
      --use-mmpose-val
    prediction=$(find "$output" -type f -name '*_dict.pkl' -print | sort | tail -n 1)
    "$python" "$repo/RUMPL/run/summarize_cmu_predictions.py" \
      "$prediction" \
      --output-json "$root/${tag}_summary.json"
    echo "END $(date --iso-8601=seconds) gpu=$gpu temperature_mm=$temperature_mm"
  } > "$root/${tag}.log" 2>&1
done
