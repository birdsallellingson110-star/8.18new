#!/usr/bin/env python3
"""Freeze the reconstructed H36M-Occl square size from external controls only."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


GBT_CONTROL = {"V2": 163.3, "V3": 39.5, "V4": 27.9}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    rows = []
    for directory in sorted(root.glob("f*")):
        result_path = directory / "result.json"
        cache_path = directory / "h36m_validation_res152_occl.pkl"
        if not result_path.is_file() or not cache_path.is_file():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        observed = {
            view: float(payload["results"][view]["absolute"]["action_equal_mm"])
            for view in GBT_CONTROL
        }
        relative_errors = {
            view: (observed[view] - target) / target
            for view, target in GBT_CONTROL.items()
        }
        # Symmetric multiplicative distance gives the three cardinalities equal
        # weight and is fixed before evaluating any learned model.
        log_rmse = math.sqrt(
            sum(math.log(observed[v] / GBT_CONTROL[v]) ** 2 for v in GBT_CONTROL)
            / len(GBT_CONTROL)
        )
        protocol = payload["protocol"].get("occlusion") or {}
        fraction = float(
            protocol["paper_unspecified_reconstructed"]["square_fraction"]
        )
        rows.append(
            {
                "tag": directory.name,
                "square_fraction": fraction,
                "observed_action_equal_mm": observed,
                "relative_error": relative_errors,
                "log_rmse": log_rmse,
                "result": str(result_path),
                "coordinate_cache": str(cache_path),
            }
        )
    if not rows:
        raise RuntimeError(f"no completed calibration arms in {root}")
    rows.sort(key=lambda row: row["log_rmse"])
    output = {
        "selection_rule": (
            "minimum equal-cardinality log-RMSE to GBT-reported Algebraic "
            "Triangulation H36M-Occl row; no learned Ours result used"
        ),
        "gbt_reported_control_mm": GBT_CONTROL,
        "selected": rows[0],
        "arms_ranked": rows,
        "reporting_boundary": (
            "GBT publishes probability 0.1 and white squares but omits size, "
            "seed, and code; selected protocol is a disclosed reconstruction, "
            "not an official dataset download or exact reproduction"
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
