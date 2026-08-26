#!/usr/bin/env python3
"""Freeze one H18 checkpoint per frontend using S8 holdout only."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/data/cjyoutput/camera_generalization_20260824")
OUTPUT = ROOT / "final_temporal_selection_20260825.json"


def candidates() -> dict[str, dict[str, Path]]:
    hr = ROOT / "hrnet_token10_generalization_20260825/canonical_h18"
    rn = ROOT / "stage1_h36m_dual_frontend/resnet152/canonical_h18"
    return {
        "hrnet": {
            "continuous_nowarp": hr / "model_continuous_nowarp/result.json",
            "uncertainty_stagebalanced": hr / "model_uncertainty_stagebalanced/result.json",
            "uncertainty_seq025": hr / "model_uncertainty_seq025/result.json",
        },
        "resnet152": {
            "canonical_h18": rn / "model/result.json",
            "uncertainty_stagebalanced": rn / "model_uncertainty_stagebalanced/result.json",
            "uncertainty_seq025": rn / "model_uncertainty_seq025/result.json",
        },
    }


def compact_test(result: dict) -> dict:
    final = result["S9_S11_final_once"]
    return {
        stage: {
            "baseline_mm": final[stage]["baseline_action_equal_all17_mm"],
            "temporal_mm": final[stage]["temporal_action_equal_all17_mm"],
            "delta_mm": final[stage]["delta_mm"],
        }
        for stage in ("V2", "V3", "V4")
    }


def main() -> None:
    report = {
        "selection_protocol": (
            "minimum best_holdout_metric_mm on clean H36M S8; S9/S11 is "
            "reported after selection and is not used to choose the checkpoint"
        ),
        "frontends": {},
    }
    for frontend, paths in candidates().items():
        rows = {}
        for name, result_path in paths.items():
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            result = json.loads(result_path.read_text())
            checkpoint = result_path.parent / "model_best.pth.tar"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            rows[name] = {
                "result": str(result_path.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "holdout_metric_mm": float(result["best_holdout_metric_mm"]),
                "best_epoch": int(result["best_epoch"]),
                "S9_S11_final_once": compact_test(result),
            }
        selected_name = min(rows, key=lambda name: rows[name]["holdout_metric_mm"])
        report["frontends"][frontend] = {
            "selected": selected_name,
            "selected_checkpoint": rows[selected_name]["checkpoint"],
            "candidates": rows,
        }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
