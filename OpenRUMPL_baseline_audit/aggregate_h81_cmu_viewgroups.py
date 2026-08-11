#!/usr/bin/env python3
"""Merge V2..V5 view-group JSON summaries and print GBT comparison notes."""

import argparse
import json
from pathlib import Path

# RUMPL-conf baseline (absolute KP*, mm) from internal CMU pose5/6 audits.
RUMPL_BASELINE_KPSTAR_ABS = {
    2: 46.91,
    3: 33.93,
    4: 29.0,
    5: 21.4,
}

# GBT (Moliner et al.) reference numbers — different protocol; see notes in output.
GBT_REFERENCES = {
    "h36m_train_cmu_test_all17_abs_mm": {
        "value": 38.9,
        "cite": "GBT Table IV: train CMU → test H36M is 38.9; train H36M → test CMU is 17.2 (inverse row).",
        "h36m_to_cmu_all17_mm": 17.2,
        "note": (
            "Paper CMU test uses 4 fixed HD cameras (2,13,10,19), T=9 frames, "
            "absolute All-17 MPJPE (mm), Iskakov split. Not the RUMPL 5-cam combo mean."
        ),
    },
    "h36m_few_cam_table1_hrnet_all17_abs_mm": {
        2: 36.8,
        3: 30.4,
        4: 26.0,
        "note": "GBT Table I: H36M dataset, fewer cameras, HRNet, absolute All-17 — H36M not CMU.",
    },
    "algebraic_triangulation_hrnet_cmu_all17_abs_mm": 26.2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.eval_root)
    rows = {}
    for n in (2, 3, 4, 5):
        path = root / f"V{n}_all_combinations" / "viewgroups.json"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        mean = data["mean_over_combinations"]
        rows[f"V{n}"] = {
            "n_combinations": data["n_combinations"],
            "mean_over_combinations": mean,
            "pooled_over_all_samples": data["overall_pooled"],
        }

    out = {
        "model": "H81",
        "protocol": (
            "CMU pose5/6; cameras 3,6,12,13,23; all C(5,k) combos; "
            "KP* = matched body joints; metrics absolute and pelvis-relative (mm)."
        ),
        "input_2d": "mmpose_hrnet_coco_matched_swapv3",
        "training": "H36M only (zero-shot CMU)",
        "view_groups": rows,
        "rumpl_baseline_absolute_kpstar_mm": RUMPL_BASELINE_KPSTAR_ABS,
        "gbt_paper_references": GBT_REFERENCES,
    }

    print(json.dumps(out, indent=2))
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n=== H81 mean over camera combinations (KP*, mm) ===")
    print("| group | combos | abs KP* | rel KP* | RUMPL baseline abs KP* |")
    print("|---:|---:|---:|---:|---:|")
    for n in (2, 3, 4, 5):
        key = f"V{n}"
        m = rows[key]["mean_over_combinations"]
        base = RUMPL_BASELINE_KPSTAR_ABS.get(n)
        print(
            f"| V{n} | {rows[key]['n_combinations']} | "
            f"{m['absolute']['kp_star_mm']:.2f} | "
            f"{m['pelvis_relative']['kp_star_mm']:.2f} | "
            f"{base:.2f} |"
        )


if __name__ == "__main__":
    main()
