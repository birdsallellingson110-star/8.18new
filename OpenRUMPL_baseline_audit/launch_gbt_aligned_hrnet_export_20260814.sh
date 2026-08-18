#!/usr/bin/env bash
# Launch the auditable, coordinate-only H36M HRNet exporter on configurable shards.
#
# This script does not train a 3D model. It creates one complete, reproducible
# front-end cache and merges it only after every shard has passed strict
# coverage checks. Two processes are placed on each 4090 by default; override
# GPU0/GPU1 when another job is using a card.
set -euo pipefail

if [[ $# -lt 4 || $# -gt 11 ]]; then
  echo "Usage: $0 {train|validation} DET_CONFIG DET_CHECKPOINT OUT_ROOT [GPU0] [GPU1] [SCORE_THR] [NUM_SHARDS] [BBOX_PADDING] [COORDINATE_SYSTEM] [DETECTOR_TEST_SCORE_THR]" >&2
  exit 2
fi

split=$1
det_config=$2
det_checkpoint=$3
out_root=$4
gpu0=${5:-0}
gpu1=${6:-1}
score_thr=${7:-0.20}
num_shards=${8:-4}
bbox_padding=${9:-}
coordinate_system=${10:-undistorted_K_equals_K}
detector_test_score_thr=${11:-}
if ! [[ "${num_shards}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SHARDS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${bbox_padding}" ]] && ! [[ "${bbox_padding}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "BBOX_PADDING must be a positive decimal or empty" >&2
  exit 2
fi
if [[ "${coordinate_system}" != "undistorted_K_equals_K" && "${coordinate_system}" != "original_distorted" ]]; then
  echo "COORDINATE_SYSTEM must be undistorted_K_equals_K or original_distorted" >&2
  exit 2
fi

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m
POSE_CONFIG=/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-384x288.py
POSE_CHECKPOINT=/mnt/data/dataset/c2i/torch/hub/checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-384x288-ca5956af_20220909.pth
case "${split}" in
  train)
    input_pkl=${DATA}/data/datasets/annot_filtered_5_64/h36m_train.pkl
    final_name=h36m_train.pkl
    ;;
  validation|val)
    input_pkl=${DATA}/data/datasets/annot_filtered_5_64/h36m_validation.pkl
    final_name=h36m_validation.pkl
    ;;
  *)
    echo "split must be train or validation" >&2
    exit 2
    ;;
esac

run_dir=${out_root}/${split}
shard_dir=${run_dir}/shards
mkdir -p "${shard_dir}" "${run_dir}/logs" "${run_dir}/merged"

echo "split=${split} input=${input_pkl} detector=${det_checkpoint} score_thr=${score_thr}"
echo "output=${run_dir} shards=${num_shards} gpus=${gpu0},${gpu1} bbox_padding=${bbox_padding:-config-default} coordinate_system=${coordinate_system} detector_test_score_thr=${detector_test_score_thr:-config-default}"

extra_export_args=()
if [[ -n "${bbox_padding}" ]]; then
  extra_export_args+=(--bbox-padding "${bbox_padding}")
fi
if [[ "${coordinate_system}" == "original_distorted" ]]; then
  extra_export_args+=(--no-undistort)
fi
if [[ -n "${detector_test_score_thr}" ]]; then
  extra_export_args+=(--detector-test-score-thr "${detector_test_score_thr}")
fi

pids=()
for shard in $(seq 0 $((num_shards - 1))); do
  if (( shard % 2 == 0 )); then gpu=${gpu0}; else gpu=${gpu1}; fi
  shard_output=${shard_dir}/shard${shard}.pkl
  shard_manifest=${shard_dir}/shard${shard}.manifest.json
  shard_log=${run_dir}/logs/shard${shard}.log
  CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "${PY}" -u "${AUDIT}/export_h36m_gbt_aligned_hrnet_20260814.py" \
      --input-pkl "${input_pkl}" \
      --images-root "${DATA}/images" \
      --pose-config "${POSE_CONFIG}" \
      --pose-checkpoint "${POSE_CHECKPOINT}" \
      --det-config "${det_config}" \
      --det-checkpoint "${det_checkpoint}" \
      --score-thr "${score_thr}" \
      "${extra_export_args[@]}" \
      --device cuda:0 --shard-id "${shard}" --num-shards "${num_shards}" \
      --output "${shard_output}" --manifest "${shard_manifest}" \
      >"${shard_log}" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if (( failed )); then
  echo "one or more exporter shards failed; inspect ${run_dir}/logs" >&2
  exit 1
fi

shards=("${shard_dir}"/shard*.pkl)
merged=${run_dir}/merged/${final_name}
manifest=${run_dir}/merged/${final_name%.pkl}.manifest.json
merge_extra_args=()
if [[ "${coordinate_system}" == "original_distorted" ]]; then
  merge_extra_args+=(--keep-camera-distortion)
fi
"${PY}" -u "${AUDIT}/merge_h36m_gbt_aligned_hrnet_20260814.py" \
  --input-pkl "${input_pkl}" --shards "${shards[@]}" \
  "${merge_extra_args[@]}" \
  --output "${merged}" --manifest "${manifest}" \
  >"${run_dir}/logs/merge.log" 2>&1

printf '%s\n' "${merged}" >"${run_dir}/merged/path.txt"
echo "complete: ${merged}"
if [[ -n "${detector_test_score_thr}" ]] && ! [[ "${detector_test_score_thr}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "DETECTOR_TEST_SCORE_THR must be a non-negative decimal or empty" >&2
  exit 2
fi
