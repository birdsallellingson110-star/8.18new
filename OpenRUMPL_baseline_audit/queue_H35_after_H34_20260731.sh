#!/usr/bin/env bash
# After H34: export A1D→H21 refined 2D PKLs, then retrain H35.
set -euo pipefail

GPU=${1:-1}
PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
AUDIT=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
DATA=/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data
ROOT=/mnt/data/cjyoutput/open_source_fusion_audit_20260731
H34_DONE=${ROOT}/H34_a1d_nobias_geom_losses/completed/H34_a1d_nobias_triAnchor_bone01_reproj01_ray01_curriculum_seed0_20260731.done
# Also accept starting after H33 if H34 is skipped/failed for long; prefer H34.
H33_DONE=${ROOT}/H33_ultra_v2_a1d_nobias/completed/H33_a1d_nobias_triAnchor_fixedK2First15_thenW6to1to1_seed0_20260731.done
TYPE_DIR=${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_h21_legswap
VAL_OUT=${TYPE_DIR}/h36m_validation.pkl
TRAIN_OUT=${TYPE_DIR}/h36m_train.pkl
A1D=${ROOT}/A1D_dense_residual_balanced/final.pth
H21=${ROOT}/H21_pose_query_v2focus_reg005/final.pth
LOG=${ROOT}/H35_chain.log
SCOREBOARD=${ROOT}/ACCURACY_ASSAULT_SCOREBOARD.txt

exec >>"${LOG}" 2>&1
echo "[H35] start $(date --iso-8601=seconds) wait H34 (or H33+30min fallback)"

# Wait for H34; if H33 done and H34 not started within long time, still wait for H34 done.
while [[ ! -s "${H34_DONE}" ]]; do
  sleep 60
done
echo "[H35] H34 done $(date --iso-8601=seconds)"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${AUDIT}"
mkdir -p "${TYPE_DIR}"

export_split() {
  local split=$1
  local out=$2
  local base_mmpose=$3
  shift 3
  local shards=("$@")
  if [[ -s "${out}" ]]; then
    echo "[H35] skip export ${split}"
    return 0
  fi
  echo "[H35] export ${split} shards=${#shards[@]} $(date --iso-8601=seconds)"
  "${PY}" -u "${AUDIT}/export_h21_refined_mmpose_pkl.py" \
    --input-pkl "${DATA}/datasets/annot_filtered_5_64/h36m_${split}.pkl" \
    --base-mmpose-pkl "${base_mmpose}" \
    --dense-shards "${shards[@]}" \
    --h21-checkpoint "${H21}" \
    --mode a1d_h21 \
    --a1d-checkpoint "${A1D}" \
    --device cuda:0 \
    --output "${out}" \
    >"${TYPE_DIR}/export_${split}.log" 2>&1
  echo "[H35] export ${split} done $(date --iso-8601=seconds)"
}

val_shards=("${ROOT}/A0_h36m_val_heatmap_topk8"/shard{0..3}.npz)
train_shards=("${ROOT}/A0_h36m_train_heatmap_topk8"/shard{0..15}.npz)

export_split validation "${VAL_OUT}" \
  "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_legswap/h36m_validation.pkl" \
  "${val_shards[@]}"

export_split train "${TRAIN_OUT}" \
  "${DATA}/datasets_mmpose/annot_filtered_5_64_mmpose_hrnet_coco_a1d_legswap/h36m_train.pkl" \
  "${train_shards[@]}"

echo "[H35] launch train $(date --iso-8601=seconds)"
bash "${AUDIT}/launch_H35_a1d_h21_tri_anchor_20260731.sh" "${GPU}"

{
  echo "=== scoreboard after H35 ($(date --iso-8601=seconds)) ==="
  "${PY}" - <<'PY'
import json
from pathlib import Path
root=Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
jobs=[]
for base, label in [
 (root/"H0_a1d_refined_rumpl_tri_anchor/eval/H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_fixedK2First8_thenWeighted3to1to1_seed0_20260731","H0"),
]:
    jobs.append((label, base))
for base in [root/"H33_ultra_v2_a1d_nobias/eval", root/"H34_a1d_nobias_geom_losses/eval", root/"H35_a1d_h21_tri_anchor/eval"]:
    if base.is_dir():
        for p in sorted(base.iterdir()):
            jobs.append((p.name, p))
print(f"{'tag':72s} {'V2':>8s} {'V4':>8s}")
for name, path in jobs:
    def read(n):
        p=path/f"V{n}"/"table2.json"
        if not p.is_file(): return None
        return json.load(open(p))["table2_action_equal"]["all17_mm"]
    v2,v4=read(2),read(4)
    if v2 is None and v4 is None: continue
    print(f"{name:72s} {v2 if v2 else float('nan'):8.3f} {v4 if v4 else float('nan'):8.3f}")
PY
} | tee -a "${SCOREBOARD}"

echo "[H35] complete $(date --iso-8601=seconds)"
