#!/usr/bin/env bash
set -euo pipefail

repo=/home/lixiaob/cjy/OpenRUMPL_baseline_audit
root=/mnt/data/cjyoutput/baseline_reaudit_20260722
ckpt="$root/output/multiview_amass_rumpl/multiview_rumpl_999/G4_gbt_fusion_geom005_exact_seed0_20260724_2026-07-24_12-04-16/model_best.pth.tar"
log="$root/phase1_g4_multiview_eval_20260724.log"
report="$root/phase1_g4_multiview_eval_summary_20260724.txt"
python=/home/lixiaob/cjy/rumpl_venv310/bin/python

exec > >(tee -a "$log") 2>&1
echo "START $(date --iso-8601=seconds)"
"$repo/eval_exact_gbt_multiview_20260724.sh" 0 G4_gbt_fusion_geom005_exact_seed0_20260724 "$ckpt" g4_fusion
"$python" - <<'PY' | tee "$report"
import json
from pathlib import Path

ROOT = Path("/mnt/data/cjyoutput/baseline_reaudit_20260722")
EVAL = ROOT / "gbt_multiview_eval"
R5_V2 = ROOT / "R5_workers16_fix_scheduler_exact_seed0_20260722_model_best_summary.json"
NAME = "G4_gbt_fusion_geom005_exact_seed0_20260724"

def combo_map(summary):
    if "per_combination" in summary:
        return {k: float(v["all17_mm"]) for k, v in summary["per_combination"].items()}
    return {k: float(v["all17_mm"]) for k, v in summary["per_pair"].items()}

r5 = combo_map(json.load(open(R5_V2)))
for n in (3, 4, 5):
    r5.update(combo_map(json.load(open(ROOT / f"multiview_model_best_eval/R5_v{n}_summary.json"))))

print("=== G4 conf+geom fusion geom_init=0.05 vs R5 ===")
print("K\tavg\tvs_R5\timproved/total\tworst\tbest_combo\tbest_delta")
for n in (2, 3, 4, 5):
    cand = combo_map(json.load(open(EVAL / f"{NAME}_v{n}_summary.json")))
    keys = sorted(set(cand) & set(r5))
    deltas = [cand[k] - r5[k] for k in keys]
    avg = sum(cand[k] for k in keys) / len(keys)
    print(
        f"V{n}\t{avg:.3f}\t{sum(deltas)/len(deltas):+.3f}\t"
        f"{sum(d < 0 for d in deltas)}/{len(keys)}\t{max(deltas):+.3f}\t"
        f"{min(keys, key=lambda k: cand[k])}\t{cand[min(keys, key=lambda k: cand[k])] - r5[min(keys, key=lambda k: cand[k])]:+.3f}"
    )
PY
echo "REPORT $report"
echo "END $(date --iso-8601=seconds)"
