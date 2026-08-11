#!/usr/bin/env bash
# H0 chain: export A1D-refined 2D PKLs (val then train) -> retrain H22 curriculum.
set -euo pipefail

GPU=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
A1D=${ROOT}/A1D_dense_residual_balanced/final.pth
TYPE_DIR=${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_legswap
VAL_OUT=${TYPE_DIR}/h36m_validation.pkl
TRAIN_OUT=${TYPE_DIR}/h36m_train.pkl
SUMMARY=${ROOT}/H0_SUMMARY.txt
CHAIN_LOG=${ROOT}/H0_chain.log

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${AUDIT}"
mkdir -p "${TYPE_DIR}" "${ROOT}"

export_split() {
  local split=$1
  local out=$2
  local base_mmpose=$3
  shift 3
  local shards=("$@")
  if [[ -s "${out}" ]]; then
    echo "[H0] skip export ${split}: ${out} exists"
    return 0
  fi
  if [[ ${#shards[@]} -lt 1 ]]; then
    echo "[H0] no dense shards for ${split}" >&2
    exit 2
  fi
  echo "[H0] export ${split} start $(date --iso-8601=seconds) shards=${#shards[@]}"
  "${PY}" -u "${AUDIT}/export_a1d_refined_mmpose_pkl.py" \
    --input-pkl "${DATA}/datasets/annot_filtered_5_64/h36m_${split}.pkl" \
    --base-mmpose-pkl "${base_mmpose}" \
    --dense-shards "${shards[@]}" \
    --a1d-checkpoint "${A1D}" \
    --device cuda:0 \
    --output "${out}" \
    >"${TYPE_DIR}/export_${split}.log" 2>&1
  echo "[H0] export ${split} done $(date --iso-8601=seconds)"
}

# Val first (fast sanity), then train. Expand globs before the call.
val_shards=("${ROOT}/A0_h36m_val_heatmap_topk8"/shard{0..3}.npz)
train_shards=("${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz)

export_split validation "${VAL_OUT}" \
  "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_inferencer_legswap/h36m_validation.pkl" \
  "${val_shards[@]}"

export_split train "${TRAIN_OUT}" \
  "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_legswap/h36m_train.pkl" \
  "${train_shards[@]}"

echo "[H0] launch curriculum train $(date --iso-8601=seconds)"
bash "${AUDIT}/launch_H0_a1d_refined_rumpl_tri_anchor_20260731.sh" "${GPU}"

{
  echo "=== H0 A1D-refined RUMPL summary ($(date --iso-8601=seconds)) ==="
  echo "TARGETS: V2 All < 40 | V4 All < 30"
  TAG=H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731
  EVAL=${ROOT}/H0_a1d_refined_rumpl_tri_anchor/eval/${TAG}
  "${PY}" - "${EVAL}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"{'view':>4s} {'All':>8s} {'KP*':>8s}  vs target")
for n in (2, 3, 4):
    p = root / f"V{n}" / "table2.json"
    if not p.is_file():
        print(f"V{n} missing {p}")
        continue
    t = json.load(open(p))["table2_action_equal"]
    allv, kpv = t["all17_mm"], t["kp_star_mm"]
    tgt = 40.0 if n == 2 else (30.0 if n == 4 else None)
    mark = ""
    if tgt is not None:
        mark = "HIT" if allv < tgt else f"MISS(need {allv-tgt:+.2f})"
    print(f"V{n} {allv:8.3f} {kpv:8.3f}  {mark}")
print("H22 baseline: V2=62.268 V3=41.999 V4=38.333")
print("H31b gate:    V2=61.294 V3=41.350 V4=37.692")
print("GBT paper 9f: V2=36.8   V3=30.4   V4=26.0")
PY
} >"${SUMMARY}" 2>&1

echo "[H0] complete $(date --iso-8601=seconds)" | tee -a "${CHAIN_LOG}"
cat "${SUMMARY}"
