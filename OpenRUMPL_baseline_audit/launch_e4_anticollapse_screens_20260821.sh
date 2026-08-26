#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
CACHE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/train_c2_22c.npz
OUTROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e4_anticollapse
mkdir -p "${OUTROOT}"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

common=(
  --train-cache "${CACHE}" --holdout-subject 8 --holdout-stride 2
  --epochs 12 --batch-size 160 --learning-rate 1e-3
  --hidden-dim 256 --token-num 34 --encoder-depth 4 --decoder-depth 1
  --mask-rate 0.2 --max-train-samples 48000 --workers 0 --seed 0 --gpu 0
)

run_simvq() {
  local name=$1 size=$2
  local out="${OUTROOT}/${name}"; mkdir -p "${out}"
  "${PY}" -u "${AUDIT}/train_e4_pct_3d_tokenizer_stage1_20260821.py" \
    "${common[@]}" --output-dir "${out}" --quantizer simvq \
    --token-dim 128 --codebook-size "${size}" --simvq-beta 0.25 \
    --commitment-weight 1.0 2>&1 | tee "${out}/train.log"
}

run_fsq() {
  local name=$1; shift
  local out="${OUTROOT}/${name}"; mkdir -p "${out}"
  local dims=$#
  "${PY}" -u "${AUDIT}/train_e4_pct_3d_tokenizer_stage1_20260821.py" \
    "${common[@]}" --output-dir "${out}" --quantizer fsq \
    --token-dim "${dims}" --fsq-levels "$@" --commitment-weight 0 \
    2>&1 | tee "${out}/train.log"
}

run_simvq simvq34_cb512_seed0 512 & p0=$!
run_simvq simvq34_cb1024_seed0 1024 & p1=$!
run_fsq fsq34_1000_seed0 8 5 5 5 & p2=$!
run_fsq fsq34_4375_seed0 7 5 5 5 5 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"
