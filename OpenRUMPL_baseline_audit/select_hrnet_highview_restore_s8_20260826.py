#!/usr/bin/env python3
"""Select one canonical HRNet continuation using S8 only."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "/mnt/data/cjyoutput/camera_generalization_20260824/"
    "hrnet_highview_restore_20260826"
)


def main() -> None:
    candidates = {
        "token10_control": {
            "checkpoint": Path(
                "/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/"
                "multiview_h36m_rumpl/multiview_rumpl_999/"
                "CAMGEN_HRNET_CANON_REPAIR_camera_ab4_token10_synth0_seed0_"
                "20260825_2026-08-25_16-44-00/final_state.pth.tar"
            ),
            "result": ROOT / "s8/token10_control/direct.json",
        },
        "restore_433": {
            "checkpoint_file": ROOT / "restore_433/final_checkpoint.txt",
            "result": ROOT / "s8/restore_433/direct.json",
        },
        "restore_244": {
            "checkpoint_file": ROOT / "restore_244/final_checkpoint.txt",
            "result": ROOT / "s8/restore_244/direct.json",
        },
    }
    rows = {}
    for name, spec in candidates.items():
        checkpoint = spec.get("checkpoint")
        if checkpoint is None:
            checkpoint = Path(spec["checkpoint_file"].read_text().strip())
        result_path = spec["result"]
        result = json.loads(result_path.read_text())
        metrics = {
            stage: float(result["results"][stage]["action_equal_all17_mm"])
            for stage in ("V2", "V3", "V4")
        }
        rows[name] = {
            "checkpoint": str(checkpoint.resolve()),
            "s8_result": str(result_path.resolve()),
            "s8_metrics_mm": metrics,
            "s8_mean_v234_mm": sum(metrics.values()) / 3.0,
        }
    selected = min(rows, key=lambda name: rows[name]["s8_mean_v234_mm"])
    report = {
        "selection_protocol": (
            "one checkpoint selected by the lowest clean H36M S8 mean over "
            "all V2/V3/V4 camera combinations; S9/S11 is not used"
        ),
        "selected": selected,
        "selected_checkpoint": rows[selected]["checkpoint"],
        "candidates": rows,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "selection_s8.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "selected_checkpoint.txt").write_text(
        rows[selected]["checkpoint"] + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
