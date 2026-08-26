#!/usr/bin/env bash
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
CACHE=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/e2_c2_input_protocol_v2/train_c2_22c.npz
OUTROOT=/mnt/data/cjyoutput/gbt_aligned_hrnet_20260821/e4_pct
mkdir -p "${OUTROOT}"

# Do not compete with the two differentiable-RANSAC screens.  This queue is
# persistent in tmux and starts immediately after their GPU-0 session exits.
while tmux has-session -t cjy_e3_dransac_20260821 2>/dev/null; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

run_one() {
  local name=$1 codebook=$2 token_dim=$3 seed=$4
  local out="${OUTROOT}/${name}"
  mkdir -p "${out}"
  "${PY}" -u "${AUDIT}/train_e4_pct_3d_tokenizer_stage1_20260821.py" \
    --train-cache "${CACHE}" --output-dir "${out}" \
    --holdout-subject 8 --holdout-stride 2 --epochs 12 --batch-size 192 \
    --learning-rate 1e-3 --hidden-dim 256 --token-dim "${token_dim}" \
    --token-num 34 --codebook-size "${codebook}" \
    --encoder-depth 4 --decoder-depth 1 --mask-rate 0.2 \
    --ema-decay 0.9 --commitment-weight 15.0 \
    --max-train-samples 48000 --workers 0 --seed "${seed}" --gpu 0 \
    2>&1 | tee "${out}/train.log"
}

# Both preserve the official PCT training mechanics.  The second checks
# whether codebook capacity, rather than the module itself, is limiting.
run_one pct34_cb512_seed0 512 128 0 &
p0=$!
run_one pct34_cb1024_seed0 1024 128 0 &
p1=$!
wait "${p0}"
wait "${p1}"
