#!/usr/bin/env bash
set -euo pipefail
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python; REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL; ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml; BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar; TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap; OUT=${ROOT}/H93_H95_temporal_eval
H91=${ROOT}/H91_H92_temporal_long/H91_unbiased/checkpoint_step_0005000.pth; H92=${ROOT}/H91_H92_temporal_long/H92_biased/checkpoint_step_0005000.pth
mkdir -p "${OUT}/logs"; exec 9>"${OUT}/pipeline.lock"; flock 9; export CUDA_VISIBLE_DEVICES=1 PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
eval_one(){ local name=$1 ckpt=$2 bias=$3 v=$4; local d=${OUT}/${name}/V${v}; mkdir -p "${d}"; extra=(); [[ -n "${ckpt}" ]] && extra+=(--temporal-checkpoint "${ckpt}"); [[ "${bias}" == 1 ]] && extra+=(--biased); "${PY}" -u "${REPO}/run/eval_temporal_h36m_table2.py" --cfg "${CFG}" --base-checkpoint "${BASE}" "${extra[@]}" --output-dir "${d}" --mmpose-type "${TYPE}" --dataset-name annot_temporal_5_5 --num-views "${v}" --window-length 9 --frame-stride 5 --depth 3 --heads 8 --batch-size 32 --workers 6 --device cuda:0 >"${d}/eval.log" 2>&1; }
for n in H76_gate0 H91_unbiased H92_biased; do mkdir -p "${OUT}/${n}"; done
eval_one H76_gate0 '' 0 2 & p0=$!; eval_one H91_unbiased "${H91}" 0 2 & p1=$!; eval_one H92_biased "${H92}" 1 2 & p2=$!; wait "${p0}"; wait "${p1}"; wait "${p2}"
eval_one H76_gate0 '' 0 4 & q0=$!
eval_one H91_unbiased "${H91}" 0 4 & q1=$!
eval_one H92_biased "${H92}" 1 4 & q2=$!
wait "${q0}"; wait "${q1}"; wait "${q2}"
wait; date --iso-8601=seconds >"${OUT}/completed.done"
