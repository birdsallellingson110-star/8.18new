#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/data/cjyoutput/baseline_reaudit_20260722")
EVAL = ROOT / "gbt_multiview_eval"
R5_V2 = ROOT / "R5_workers16_fix_scheduler_exact_seed0_20260722_model_best_summary.json"

MODELS = [
    ("G0_gbt_formula_exact_seed0_20260723", "G0 formula"),
    ("G1_gbt_fusion_exact_seed0_20260723", "G1 conf+geom fusion"),
    ("G2_gbt_conf_only_exact_seed0_20260723", "G2 conf-only"),
    ("G3_gbt_geom_fusion_only_exact_seed0_20260723", "G3 geom fusion-only"),
]


def load_summary(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def combo_map(summary: dict) -> dict[str, float]:
    if "per_combination" in summary:
        return {
            key: float(value["all17_mm"])
            for key, value in summary["per_combination"].items()
        }
    return {
        key: float(value["all17_mm"]) for key, value in summary["per_pair"].items()
    }


def build_r5() -> dict[str, float]:
    r5 = combo_map(load_summary(R5_V2))
    for n_views in (3, 4, 5):
        r5.update(
            combo_map(load_summary(ROOT / f"multiview_model_best_eval/R5_v{n_views}_summary.json"))
        )
    return r5


def summarize(name: str, label: str, r5: dict[str, float]) -> None:
    print(f"=== {label} vs R5 ===")
    print("K\tavg\tvs_R5\timproved/total\tworst\tbest_combo\tbest_delta")
    for n_views in (2, 3, 4, 5):
        summary = load_summary(EVAL / f"{name}_v{n_views}_summary.json")
        candidate = combo_map(summary)
        keys = sorted(set(candidate) & set(r5))
        deltas = [candidate[k] - r5[k] for k in keys]
        improved = sum(delta < 0 for delta in deltas)
        worst = max(deltas)
        avg = sum(candidate[k] for k in keys) / len(keys)
        avg_delta = sum(deltas) / len(deltas)
        best_key = min(keys, key=lambda key: candidate[key])
        best_delta = candidate[best_key] - r5[best_key]
        print(
            f"V{n_views}\t{avg:.3f}\t{avg_delta:+.3f}\t"
            f"{improved}/{len(keys)}\t{worst:+.3f}\t"
            f"{best_key}\t{best_delta:+.3f}"
        )
    print()


def main() -> None:
    r5 = build_r5()
    print("R5 reference averages:")
    for n_views in (2, 3, 4, 5):
        keys = [key for key in r5 if len(key.split("_")) == n_views]
        print(f"  V{n_views}: {sum(r5[k] for k in keys) / len(keys):.3f}")
    print()
    for name, label in MODELS:
        summarize(name, label, r5)


if __name__ == "__main__":
    main()
