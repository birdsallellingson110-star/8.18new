#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {v2|v34} PHYSICAL_GPU" >&2
  exit 2
fi

split=$1
physical_gpu=$2
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
DENSE=${ROOT}/A0_h36m_val_heatmap_topk8

for shard in 0 1 2 3; do
  while [[ ! -s "${DENSE}/shard${shard}.npz" \
        || ! -s "${DENSE}/shard${shard}.heatmaps.npy" ]]; do
    sleep 30
  done
done
while tmux has-session -t cjy_h20_random24 2>/dev/null \
   || tmux has-session -t cjy_h20_fixed2 2>/dev/null; do
  sleep 30
done

# GPU 1 first runs the independent official-PPT reference.  Once its 2D
# forward pass is complete, PPT's CPU triangulation and this dense diagnostic
# can overlap safely.
if [[ "${split}" == v34 ]]; then
  ppt_marker=/mnt/data/cjyoutput/open_source_fusion_audit_20260731/PPT_official_h36m_protocol_matched/official_pose2d_complete.marker
  while tmux has-session -t cjy_ppt_official 2>/dev/null \
     && [[ ! -e "${ppt_marker}" ]]; do
    sleep 30
  done
fi

if [[ "${split}" == v2 ]]; then
  views=(2)
  output="${ROOT}/A0D_dense_epipolar_full_V2.json"
elif [[ "${split}" == v34 ]]; then
  views=(3 4)
  output="${ROOT}/A0D_dense_epipolar_full_V3V4.json"
else
  echo "Unsupported split: ${split}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
exec /home/lixiaob/cjy/rumpl_venv310/bin/python -u \
  /home/lixiaob/cjy/OpenRUMPL_baseline_audit/eval_h36m_dense_epipolar_heatmaps.py \
  --input-pkl \
    /mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/datasets/annot_filtered_5_64/h36m_validation.pkl \
  --dense-shards \
    "${DENSE}/shard0.npz" "${DENSE}/shard1.npz" \
    "${DENSE}/shard2.npz" "${DENSE}/shard3.npz" \
  --views "${views[@]}" \
  --alphas 0.25 0.5 1.0 \
  --depth-samples 64 \
  --device cuda:0 \
  --output "${output}"
