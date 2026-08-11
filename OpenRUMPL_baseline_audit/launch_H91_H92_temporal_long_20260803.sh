#!/usr/bin/env bash
set -euo pipefail
physical_gpu=${1:-1}; PY=/home/lixiaob/cjy/rumpl_venv310/bin/python; REPO=/home/lixiaob/cjy/OpenRUMPL/RUMPL; ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CFG=${ROOT}/H35_a1d_h21_refined_rumpl_tri_anchor.yaml; BASE=/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999/H76_H50_anchorCenteredPlucker_workers12_seed0_20260803_2026-08-03_03-00-38/model_best.pth.tar; TYPE=mmpose_hrnet_coco_a1d_h21_a1dmatched_legswap; OUT=${ROOT}/H91_H92_temporal_long
mkdir -p "${OUT}/logs" "${OUT}/H91_unbiased" "${OUT}/H92_biased"; exec 9>"${OUT}/pipeline.lock"; flock 9
export CUDA_VISIBLE_DEVICES="${physical_gpu}" PYTHONPATH="${REPO}/lib" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
common=(--cfg "${CFG}" --base-checkpoint "${BASE}" --train-mmpose-type "${TYPE}" --train-dataset-name annot_filtered_5_64 --window-length 9 --frame-stride 5 --num-views 2 --depth 3 --heads 8 --token-dropout 0.2 --optimizer-steps 5000 --warmup-steps 500 --micro-batch-size 1 --effective-batch-size 8 --workers 4 --lr 0.0001 --weight-decay 0.0001 --seed 0 --device cuda:0 --amp-dtype bf16 --log-every 50 --save-every 1000)
"${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" "${common[@]}" --output-dir "${OUT}/H91_unbiased" >"${OUT}/logs/H91.log" 2>&1 & p0=$!
"${PY}" -u "${REPO}/run/train_temporal_gbt_rumpl.py" "${common[@]}" --biased --output-dir "${OUT}/H92_biased" >"${OUT}/logs/H92.log" 2>&1 & p1=$!
wait "${p0}"; wait "${p1}"; date --iso-8601=seconds >"${OUT}/completed.done"
