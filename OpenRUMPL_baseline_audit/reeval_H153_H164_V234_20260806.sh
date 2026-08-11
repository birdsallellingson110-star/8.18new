#!/usr/bin/env bash
# Re-run V3/V4 eval for H153–H164 (V2 already in eval trees). Uses RUMPL_EVAL_STRICT=0 + view adapt load.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml
BASE=${ROOT}/H153_H164_radical_sprint
physical_gpu=${1:-0}
workers=${RUMPL_WORKERS:-8}

export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTHONPATH="${AUDIT}:${REPO}/lib"
export RUMPL_EVAL_STRICT=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

tags=(
  H153_H81_skipVft_meanFuse_w322_ftH81_workers12_seed0_20260805
  H154_H81_skipVftPft_minimal_w322_ftH81_workers12_seed0_20260805
  H155_H81_globalJV2_skipVft_w322_ftH81_workers12_seed0_20260805
  H156_H81_vftDepth1_w322_ftH81_workers12_seed0_20260805
  H157_H81_noTri_skipVft_w322_ftH81_workers12_seed0_20260805
  H158_H81_triOnly_skipVftPft_w322_ftH81_workers12_seed0_20260805
  H159_H81_jv2_skipVftPft_w322_ftH81_workers12_seed0_20260805
  H160_H76_skipVft_meanFuse_w322_ftH76_workers12_seed0_20260805
  H161_H81_vft1_skipPft_w322_ftH81_workers12_seed0_20260805
  H162_H81_skipVft_graphRes_w322_ftH81_workers12_seed0_20260805
  H163_H76_setDecoder_w322_ftH76_workers12_seed0_20260805
  H164_H81_skipVft_relView_w322_ftH81_workers12_seed0_20260805
)

cd "${REPO}"
for tag in "${tags[@]}"; do
  ckpt_file="${BASE}/checkpoints/${tag}.txt"
  if [[ ! -s "${ckpt_file}" ]]; then
    echo "[skip] no checkpoint pointer for ${tag}" >&2
    continue
  fi
  checkpoint=$(cat "${ckpt_file}")
  if [[ ! -s "${checkpoint}" ]]; then
    echo "[skip] missing ckpt ${checkpoint}" >&2
    continue
  fi
  type_line=$(grep -m1 'train-mmpose-type' "${BASE}/logs/${tag}_train.log" 2>/dev/null || true)
  TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap
  eval_root="${BASE}/eval/${tag}"
  for n_views in 3 4; do
    eval_dir=${eval_root}/V${n_views}
    mkdir -p "${eval_dir}"
    echo "[reeval] ${tag} V${n_views} $(date --iso-8601=seconds)"
    "${PY}" -u run/eval_rumpl_checkpoint.py \
      --cfg "${CFG}" --checkpoint "${checkpoint}" --output-dir "${eval_dir}" \
      --workers "${workers}" --gpu 0 --use-mmpose-val true --flip-lower-body-kp-test true \
      --test-on-all-cameras true --n-views-combinations "${n_views}" \
      --test-mmpose-type "${TYPE}" >"${eval_dir}/eval.log" 2>&1 || {
        echo "[warn] eval failed ${tag} V${n_views}" >&2
        continue
      }
    prediction=$(find "${eval_dir}" -maxdepth 1 -name 'preds_gt_*_dict.pkl' -print -quit)
    if [[ -n "${prediction}" ]]; then
      "${PY}" run/eval_h36m_table2.py --dict-pkl "${prediction}" \
        --output-json "${eval_dir}/table2.json" >"${eval_dir}/table2.log" 2>&1 || true
    fi
  done
done
echo "[reeval] done $(date --iso-8601=seconds)"
