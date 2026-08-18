#!/usr/bin/env bash
# Export HRNet coordinates after the official LT H36M crop/undistort protocol.
# This line intentionally uses annotation/GT boxes first: it is an LT frontend
# upper bound, not an off-the-shelf detector comparison.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: $0 {train|validation} OUT_ROOT [GPU0] [GPU1] [NUM_SHARDS] [MAX_RECORDS_PER_SHARD]" >&2
  exit 2
fi

split=$1
out_root=$2
gpu0=${3:-0}
gpu1=${4:-1}
num_shards=${5:-4}
max_records=${6:-}

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
POSE_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth

case "${split}" in
  train) input_pkl=${DATA}/data/datasets/annot_filtered_5_64/h36m_train.pkl; final_name=h36m_train.pkl ;;
  validation|val) input_pkl=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl; final_name=h36m_validation.pkl ;;
  *) echo "split must be train or validation" >&2; exit 2 ;;
esac

run_dir=${out_root}/${split}
shard_dir=${run_dir}/shards
mkdir -p "${shard_dir}" "${run_dir}/logs" "${run_dir}/merged"

echo "split=${split} input=${input_pkl} output=${run_dir} LT-bbox=annotation crop=384x384 shards=${num_shards}"

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  if (( shard % 2 == 0 )); then gpu=${gpu0}; else gpu=${gpu1}; fi
  shard_output=${shard_dir}/shard${shard}.pkl
  shard_manifest=${shard_dir}/shard${shard}.manifest.json
  shard_log=${run_dir}/logs/shard${shard}.log
  extra=()
  if [[ -n "${max_records}" ]]; then extra+=(--max-records "${max_records}"); fi
  CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "${PY}" -u "${AUDIT}/export_h36m_lt_style_hrnet_20260814.py" \
      --input-pkl "${input_pkl}" --images-root "${DATA}/images" \
      --pose-config "${POSE_CONFIG}" --pose-checkpoint "${POSE_CHECKPOINT}" \
      --bbox-padding 1.0 --device cuda:0 --shard-id "${shard}" --num-shards "${num_shards}" \
      "${extra[@]}" --output "${shard_output}" --manifest "${shard_manifest}" \
      >"${shard_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed )); then
  echo "LT-style exporter failed; inspect ${run_dir}/logs" >&2
  exit 1
fi

shards=("${shard_dir}"/shard*.pkl)
merged=${run_dir}/merged/${final_name}
manifest=${run_dir}/merged/${final_name%.pkl}.manifest.json
"${PY}" -u "${AUDIT}/merge_h36m_lt_style_hrnet_20260814.py" \
  --input-pkl "${input_pkl}" --shards "${shards[@]}" \
  --output "${merged}" --manifest "${manifest}" \
  >"${run_dir}/logs/merge.log" 2>&1

echo "LT-style HRNet cache ready: ${merged}"
