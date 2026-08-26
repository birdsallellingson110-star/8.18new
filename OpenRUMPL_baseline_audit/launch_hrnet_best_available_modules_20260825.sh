#!/usr/bin/env bash
# After all predeclared generator repairs finish, freeze the best available
# canonical HRNet checkpoint and continue with fresh E2/H18 artifacts.  The
# epoch-13 internal-validation checkpoint is the fallback requested by the
# user; old C2 is a reference target, not a downstream blocker.
set -euo pipefail

PY=/home/lixiaob/cjy/rumpl_venv310/bin/python
BASE=/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_canonical_repair
BRANCHES=${BASE}/branches_20260825
DIAG=${BASE}/diagnostics_20260825/internal_best_epoch13/eval
PHASE1=${BASE}/phase1_cont20_lr1e5
MODULE_ROOT=${MODULE_ROOT:-${BASE}/best_available_modules_20260825_fixed_geometry}
SHARED=/home/lixiaob/cjy/OpenRUMPL_baseline_audit/launch_stage1_canonical_dual_frontend_downstream_20260824.sh

names=(
  cont10_lr3e6_fixed
  cont10_lr1e6_fixed
  pelvisprior20_lr1e5_reg1e3
  pelvisprior20_lr1e5_reg1e2
)
for name in "${names[@]}"; do
  while [[ ! -s "${BRANCHES}/${name}/COMPLETED" ]]; do sleep 30; done
done

mkdir -p "${MODULE_ROOT}/hrnet/generator"
"${PY}" - "${BASE}" "${MODULE_ROOT}/selection.json" <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
phase1 = base / "phase1_cont20_lr1e5"
model_dir = pathlib.Path((phase1 / "model_dir.txt").read_text().strip())
candidates = {
    "internal_best_epoch13": {
        "eval": base / "diagnostics_20260825/internal_best_epoch13/eval",
        "checkpoint": model_dir / "model_best.pth.tar",
        "kind": "fallback selected by internal four-view validation",
    },
}
for name in (
    "cont10_lr3e6_fixed",
    "cont10_lr1e6_fixed",
    "pelvisprior20_lr1e5_reg1e3",
    "pelvisprior20_lr1e5_reg1e2",
):
    root = base / "branches_20260825" / name
    candidates[name] = {
        "eval": root / "eval",
        "checkpoint": pathlib.Path((root / "final_checkpoint.txt").read_text().strip()),
        "kind": "predeclared final epoch",
    }

records = []
for name, candidate in candidates.items():
    metrics = {}
    for views in (2, 3, 4):
        with (candidate["eval"] / f"V{views}/table2.json").open() as stream:
            payload = json.load(stream)
        metrics[f"V{views}"] = float(
            payload["table2_action_equal"]["all17_mm"]
        )
    checkpoint = candidate["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    records.append({
        "name": name,
        "kind": candidate["kind"],
        "checkpoint": str(checkpoint),
        "metrics_mm": metrics,
        "mean_v234_mm": sum(metrics.values()) / 3.0,
    })

# The aggregate criterion is declared before branch results are available.
# It prevents choosing three separate checkpoints for three view counts.
selected = min(records, key=lambda item: item["mean_v234_mm"])
report = {
    "selection_policy": "lowest mean of formal V2/V3/V4 MPJPE; one checkpoint for all view counts",
    "fallback": "internal_best_epoch13",
    "old_c2_reference_mm": {"V2": 38.686, "V3": 30.943, "V4": 28.629},
    "candidates": records,
    "selected": selected,
}
with output.open("w") as stream:
    json.dump(report, stream, indent=2)
    stream.write("\n")
print(json.dumps(report, indent=2))
PY

checkpoint=$("${PY}" - "${MODULE_ROOT}/selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])
PY
)
test -s "${checkpoint}"
printf '%s\n' "${checkpoint}" >"${MODULE_ROOT}/hrnet/generator/checkpoint.txt"
{
  echo "checkpoint=${checkpoint}"
  echo "selection=${MODULE_ROOT}/selection.json"
  echo "cache_policy=fresh; no epoch-7 or previous repair cache reuse"
} >"${MODULE_ROOT}/hrnet/generator/manifest.txt"

source "${SHARED}"
ROOT="${MODULE_ROOT}"
export RUMPL_BODY_CANONICAL_FRAME=1
export RUMPL_BODY_CANONICAL_REG=1e-2
export RUMPL_BODY_CANONICAL_PELVIS_PRIOR=1
# The raw geometric intersection can be numerically valid yet far outside the
# H76 pose on difficult H36M camera pairs.  Keep the added candidate as a
# small, bounded IRLS residual around the matching frozen H76 hypothesis.
export CANDIDATE_SOLVER_MODE="${CANDIDATE_SOLVER_MODE:-irls}"
export CANDIDATE_BLEND_ALPHA="${CANDIDATE_BLEND_ALPHA:-0.1}"
export CANDIDATE_MAX_DELTA_M="${CANDIDATE_MAX_DELTA_M:-0.1}"
run_frontend hrnet gbt_yolox_x_score001_fallback_legswap 0 \
  /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_train_v2/train/merged/h36m_train.pkl \
  /mnt/data/cjyoutput/gbt_aligned_hrnet_20260814/yolox_x_score001_validation_v2/validation/merged/h36m_validation.pkl \
  annot_temporal_5_5 \
  /mnt/data/cjyoutput/gbt_aligned_hrnet_20260817/h8_temporal_frontend/validation/merged/h36m_validation.pkl

date --iso-8601=seconds >"${MODULE_ROOT}/STAGE1_COMPLETED"
echo "[best available canonical HRNet E2/H18] complete ${MODULE_ROOT}"
