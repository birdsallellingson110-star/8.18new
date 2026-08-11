#!/usr/bin/env bash
# H31: learn a per-joint gate for the A1D (dense cross-view fusion) anchor
# delta and evaluate it on the full validation set.  Two variants are chained
# serially on one GPU:
#   H31a - positive sigmoid gate (context features, warm-started from H24)
#   H31b - signed residual gate (tanh, warm-started from H31a) targeting the
#          clipMinus1to1 oracle ceiling observed in H30.
set -euo pipefail

GPU=${1:-0}
WAIT_FOR=${2:-}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
CACHE=${ROOT}/H24_h22_train_predictions
H21=${ROOT}/H21_pose_query_v2focus_reg005/final.pth
A1D=${ROOT}/A1D_dense_residual_balanced/final.pth
H24_GATE=${ROOT}/H24_learned_anchor_gate/final.pth
H22_EVAL=/mnt/data/cjyoutput/h36m_original_rumpl_tri_anchor_20260731/eval/H20_H22_CUR_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_clean_realH36M_seed0_20260731

OUT_A=${ROOT}/H31a_a1d_context_gate
OUT_B=${ROOT}/H31b_a1d_signed_gate
SUMMARY=${ROOT}/H31_a1d_gate_summary.txt

if [[ -n "${WAIT_FOR}" ]]; then
  while [[ ! -s "${WAIT_FOR}" ]]; do
    sleep 15
  done
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${AUDIT}"

train_gate() {
  local out="$1"; shift
  mkdir -p "${out}"
  if [[ ! -s "${out}/final.pth" ]]; then
    "${PY}" -u "${AUDIT}/train_anchor_delta_gate.py" \
      --input-pkl \
        "${DATA}/datasets/annot_filtered_5_64/h36m_train.pkl" \
      --rumpl-input-pkl \
        "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_legswap/h36m_train.pkl" \
      --dense-shards "${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz \
      --h21-checkpoint "${H21}" \
      --anchor-delta-source a1d \
      --a1d-checkpoint "${A1D}" \
      --context-features \
      --rumpl-prediction-dicts \
        "${CACHE}/V2/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl" \
        "${CACHE}/V3/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl" \
        "${CACHE}/V4/preds_gt_multiview_h36m_rumpl_mmpose__dict.pkl" \
      --group-manifests \
        "${CACHE}/V2/groups.json" \
        "${CACHE}/V3/groups.json" \
        "${CACHE}/V4/groups.json" \
      --steps 10000 \
      --view-probabilities 3 1 1 \
      --learning-rate 0.0003 \
      --seed 0 \
      --device cuda:0 \
      --output-dir "${out}" \
      "$@" \
      >"${out}/train.log" 2>&1
  fi
}

eval_gate() {
  local out="$1"
  if [[ ! -s "${out}/full_eval.json" ]]; then
    "${PY}" -u "${AUDIT}/eval_h23_rumpl_pose_query_anchor.py" \
      --input-pkl \
        "${DATA}/datasets/annot_filtered_5_64/h36m_validation.pkl" \
      --rumpl-input-pkl \
        "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_inferencer_legswap/h36m_validation.pkl" \
      --dense-shards "${ROOT}/A0_h36m_val_heatmap_topk8"/shard{0..3}.npz \
      --checkpoint "${H21}" \
      --a1d-checkpoint "${A1D}" \
      --gate-checkpoint "${out}/final.pth" \
      --prediction-root "${H22_EVAL}" \
      --views 2 3 4 \
      --query-sources old_anchor \
      --anchor-delta-scales 0.25 \
      --a1d-delta-scales 0.25 \
      --device cuda:0 \
      --output "${out}/full_eval.json" \
      >"${out}/full_eval.log" 2>&1
  fi
}

# H31a: positive gate warm-started from the H24 reliability gate.
train_gate "${OUT_A}" --initial-gate-checkpoint "${H24_GATE}"
eval_gate "${OUT_A}"

# H31b: signed residual gate warm-started from H31a.
train_gate "${OUT_B}" \
  --signed-residual-gate \
  --initial-gate-checkpoint "${OUT_A}/final.pth"
eval_gate "${OUT_B}"

{
  echo "=== H31 A1D gate summary ($(date --iso-8601=seconds)) ==="
  "${PY}" - "${OUT_A}/full_eval.json" "${OUT_B}/full_eval.json" <<'PYEOF'
import json, sys
labels = [
    "frozen_h22",
    "h22_plus_0.25x_a1d_delta",
    "h22_plus_a1d_learned_gate",
    "a1d_oracle_per_joint_scale_clip0to1",
    "a1d_oracle_per_joint_scale_clipMinus1to1",
]
for path in sys.argv[1:]:
    name = path.split("/")[-2]
    try:
        data = json.load(open(path))
    except Exception as exc:  # noqa: BLE001
        print(f"{name}: {exc}")
        continue
    for tag in ("V2", "V3", "V4"):
        methods = data.get("results", {}).get(tag, {}).get("methods", {})
        for label in labels:
            entry = methods.get(label)
            if entry is None:
                continue
            allv = entry.get("action_equal_all17_mm")
            kpv = entry.get("frame_weighted_kp_star_mm")
            print(
                f"{name:24s} {tag} {label:42s} "
                f"All={allv:.3f} KP*={kpv:.3f}"
            )
PYEOF
} >"${SUMMARY}" 2>&1

echo "[H31] complete $(date --iso-8601=seconds)"
