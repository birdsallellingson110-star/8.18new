#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: $0 GPU VARIANT MIN_CENTER_OKS}
variant=${2:?usage: $0 GPU VARIANT MIN_CENTER_OKS}
min_center_oks=${3:?usage: $0 GPU VARIANT MIN_CENTER_OKS}

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
output_root=/mnt/data/cjyoutput/temporal_exact_20260723
config="$repo/RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
checkpoint=/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar
data='/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl'
python=/home/lixiaob/cjy/rumpl_venv310/bin/python

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo/RUMPL/lib"
export TORCH_HOME=/mnt/data/dataset/c2i/torch
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache
export WANDB_MODE=disabled
export RUMPL_FIX_PFT_LAST_BLOCK=0
export GBT_LEARNABLE_BIAS=0

mkdir -p "$output_root/$variant"
exec > >(tee -a "$output_root/${variant}.log") 2>&1
echo "START $(date --iso-8601=seconds)"
echo "GPU=$gpu VARIANT=$variant MIN_CENTER_OKS=$min_center_oks"
git -C "$repo" status --short
git -C "$repo" diff -- RUMPL/lib/models/exact_temporal_rumpl.py \
  RUMPL/lib/dataset/exact_temporal_clip.py RUMPL/run/train_exact_temporal_rumpl.py
cp "$repo/RUMPL/lib/models/exact_temporal_rumpl.py" "$output_root/$variant/"
cp "$repo/RUMPL/lib/dataset/exact_temporal_clip.py" "$output_root/$variant/"
cp "$repo/RUMPL/run/train_exact_temporal_rumpl.py" "$output_root/$variant/"
sha256sum "$config" "$checkpoint"

"$python" "$repo/RUMPL/run/train_exact_temporal_rumpl.py" \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --data "$data" \
  --output "$output_root/$variant" \
  --frames 9 --depth 2 --epochs 20 --batch-size 16 --workers 4 \
  --validation-clips 200 --learning-rate 4e-5 --weight-decay 0.1 \
  --huber-delta 0.1 --min-center-oks "$min_center_oks" --seed 0

echo "END $(date --iso-8601=seconds)"
