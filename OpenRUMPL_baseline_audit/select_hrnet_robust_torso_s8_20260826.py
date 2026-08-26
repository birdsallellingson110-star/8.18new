#!/usr/bin/env python3
"""Select the robust-torso continuation using clean S8 only."""

import json
from pathlib import Path


ROOT = Path(
    "/mnt/data/cjyoutput/camera_generalization_20260824/"
    "hrnet_robust_torso_20260826"
)


def main():
    candidates = {
        "token10_control": {
            "checkpoint": Path(
                "/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/"
                "multiview_h36m_rumpl/multiview_rumpl_999/"
                "CAMGEN_HRNET_CANON_REPAIR_camera_ab4_token10_synth0_seed0_"
                "20260825_2026-08-25_16-44-00/final_state.pth.tar"
            ),
            "result": Path(
                "/mnt/data/cjyoutput/camera_generalization_20260824/"
                "hrnet_highview_restore_20260826/s8/token10_control/direct.json"
            ),
            "robust_torso": False,
        },
        "robust_drop0": {
            "checkpoint": ROOT / "robust_drop0/final_checkpoint.txt",
            "result": ROOT / "s8/robust_drop0/direct.json",
            "robust_torso": True,
        },
        "robust_drop10": {
            "checkpoint": ROOT / "robust_drop10/final_checkpoint.txt",
            "result": ROOT / "s8/robust_drop10/direct.json",
            "robust_torso": True,
        },
    }
    rows = {}
    for name, spec in candidates.items():
        checkpoint_path = spec["checkpoint"]
        checkpoint = (
            checkpoint_path.read_text().strip()
            if checkpoint_path.suffix == ".txt"
            else str(checkpoint_path)
        )
        payload = json.loads(spec["result"].read_text())
        metrics = {
            f"V{views}": payload["results"][f"V{views}"]["action_equal_all17_mm"]
            for views in (2, 3, 4)
        }
        rows[name] = {
            "checkpoint": checkpoint,
            "s8_result": str(spec["result"]),
            "robust_torso": spec["robust_torso"],
            "s8_metrics_mm": metrics,
            "s8_mean_v234_mm": sum(metrics.values()) / 3,
        }
    selected = min(rows, key=lambda name: rows[name]["s8_mean_v234_mm"])
    report = {
        "selection_protocol": (
            "lowest clean H36M S8 action-equal All-17 mean over V2/V3/V4; "
            "S9/S11 untouched until selection"
        ),
        "candidates": rows,
        "selected": selected,
        "selected_record": rows[selected],
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "selection_s8.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "selected_name.txt").write_text(selected + "\n")
    (ROOT / "selected_checkpoint.txt").write_text(rows[selected]["checkpoint"] + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
