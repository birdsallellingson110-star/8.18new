#!/usr/bin/env bash
set -euo pipefail

# Reproduce the official MTF-Transformer Human3.6M T=7 checkpoint without
# modifying the upstream repository.  The official script resolves `data/`
# relative to the working directory, so a mounted-disk runtime directory is
# used here.

repo_dir="/home/lixiaob/cjy/reference/MTF-Transformer"
python_bin="/home/lixiaob/cjy/rumpl_venv310/bin/python"
data_dir="/mnt/data/cjydata/datasets/mtf_transformer_official/data"
output_root="/mnt/data/cjyoutput/mtf_transformer_official"
checkpoint_path="${output_root}/checkpoint/submit/t_7_dim_4_2022-03-24-17-56/model.bin"
runtime_dir="${output_root}/runtime"
log_path="${output_root}/logs/official_t7_v2_v4_eval.log"

declare -A expected_sizes=(
  [h36m_sub1.npz]=253384559
  [h36m_sub5.npz]=404283365
  [h36m_sub6.npz]=254902321
  [h36m_sub7.npz]=414654721
  [h36m_sub8.npz]=263927279
  [h36m_sub9.npz]=324062399
  [h36m_sub11.npz]=230229214
  [score.pkl]=145390738
)

for file_name in "${!expected_sizes[@]}"; do
  file_path="${data_dir}/${file_name}"
  if [[ ! -f "${file_path}" ]]; then
    printf 'missing official MTF data: %s\n' "${file_path}" >&2
    exit 2
  fi
  actual_size="$(stat -c '%s' "${file_path}")"
  if [[ "${actual_size}" != "${expected_sizes[${file_name}]}" ]]; then
    printf 'incomplete official MTF data: %s (%s/%s bytes)\n' \
      "${file_path}" "${actual_size}" "${expected_sizes[${file_name}]}" >&2
    exit 2
  fi
done

if [[ ! -f "${checkpoint_path}" ]] || \
   [[ "$(stat -c '%s' "${checkpoint_path}")" != "218287245" ]]; then
  printf 'missing or incomplete official MTF checkpoint: %s\n' "${checkpoint_path}" >&2
  exit 2
fi

mkdir -p "${runtime_dir}" "${output_root}/logs"
ln -sfn "${data_dir}" "${runtime_dir}/data"

cd "${runtime_dir}"
exec "${python_bin}" -u "${repo_dir}/run_h36m.py" \
  --cfg "${repo_dir}/cfg/submit/t_7_dim_4.yaml" \
  --eval \
  --checkpoint "${checkpoint_path}" \
  --gpu 1 \
  --eval_n_frames 1 7 \
  --eval_n_views 2 4 \
  --eval_batch_size 500 \
  --n_frames 7 \
  2>&1 | tee "${log_path}"
