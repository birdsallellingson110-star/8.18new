#!/usr/bin/env python3
"""Summarize Phase-0 J2/S1 eval vs frozen R5 baseline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/data/cjyoutput/baseline_reaudit_20260722/multiview_model_best_eval")
R5_V2 = Path(
    "/mnt/data/cjyoutput/baseline_reaudit_20260722/"
    "R5_workers16_fix_scheduler_exact_seed0_20260722_model_best_summary.json"
)


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


def summarize_family(name: str, r5: dict[str, float]) -> None:
    print(f"=== {name} vs R5 (All17 mm, lower is better) ===")
    print("K\tavg\tvs_R5\timproved/total\tworst\tbest_combo\tbest_mpjpe\tbest_delta")
    for n_views in (2, 3, 4, 5):
        summary = load_summary(ROOT / f"{name}_v{n_views}_summary.json")
        candidate = combo_map(summary)
        keys = sorted(set(candidate) & set(r5))
        deltas = [candidate[k] - r5[k] for k in keys]
        improved = sum(delta < 0 for delta in deltas)
        worst = max(deltas)
        best_key = min(keys, key=lambda key: candidate[key])
        best_delta = candidate[best_key] - r5[best_key]
        avg = sum(candidate[k] for k in keys) / len(keys)
        avg_delta = sum(deltas) / len(deltas)
        print(
            f"V{n_views}\t{avg:.3f}\t{avg_delta:+.3f}\t"
            f"{improved}/{len(keys)}\t{worst:+.3f}\t"
            f"{best_key}\t{candidate[best_key]:.3f}\t{best_delta:+.3f}"
        )
    print()


def main() -> None:
    r5_v2 = combo_map(load_summary(R5_V2))
    r5_rest = {}
    for n_views in (3, 4, 5):
        r5_rest.update(combo_map(load_summary(ROOT / f"R5_v{n_views}_summary.json")))
    r5 = {**r5_v2, **r5_rest}

    print("Phase 0 reference R5 averages (All17 mm):")
    for n_views in (2, 3, 4, 5):
        keys = [
            key
            for key in r5
            if len(key.split("_")) == n_views
        ]
        avg = sum(r5[key] for key in keys) / len(keys)
        print(f"  V{n_views}: {avg:.3f} ({len(keys)} combos)")
    print()

    summarize_family("J2", r5)
    summarize_family("S1", r5)


if __name__ == "__main__":
    main()
