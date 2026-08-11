#!/usr/bin/env python3
"""Recompute the Human3.6M metrics using the RUMPL Table-2 protocol.

The regular validation summary averages all evaluated frame/camera-pair
records.  Table 2 in the paper instead reports one value per action and then
gives the arithmetic mean of the 15 action values.  This script reports both
quantities for All-17 and for the paper's KP* subset.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


ACTION_NAMES = {
    2: "Direction",
    3: "Discuss",
    4: "Eating",
    5: "Greet",
    6: "Phone",
    7: "Photo",
    8: "Pose",
    9: "Purchase",
    10: "Sitting",
    11: "SittingDown",
    12: "Smoke",
    13: "Wait",
    14: "WalkDog",
    15: "Walk",
    16: "WalkTwo",
}

# Shoulders, elbows, wrists, knees and ankles in the repository's H36M order.
KP_STAR_INDICES = [11, 14, 12, 15, 13, 16, 5, 2, 6, 3]

# COCO-17 indices for the same named joints.  These are intentionally kept as
# a diagnostic: applying them directly to an H36M-ordered tensor does *not*
# select the joints described by the paper, but reproduces the published 56.8
# number surprisingly closely and is therefore useful for auditing a likely
# evaluation-index mismatch.
COCO_KP_STAR_INDICES_ON_H36M = [5, 6, 7, 8, 9, 10, 13, 14, 15, 16]

PAPER_ALL_MM = 52.5
PAPER_KP_STAR_MM = 56.8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dict-pkl",
        required=True,
        help="preds_gt_*_dict.pkl produced by RUMPL validation",
    )
    parser.add_argument(
        "--coordinate-unit",
        choices=("meter", "millimeter"),
        default="meter",
        help="unit used by pred and gt arrays",
    )
    parser.add_argument("--output-json", help="optional machine-readable output")
    return parser.parse_args()


def action_id(fname):
    fields = fname.split("_")
    try:
        act_pos = fields.index("act")
        return int(fields[act_pos + 1])
    except (ValueError, IndexError):
        # Matches the official evaluator's historical filename convention.
        return int(fields[3])


def main():
    args = parse_args()
    with open(args.dict_pkl, "rb") as stream:
        data = pickle.load(stream)

    pred = np.asarray(data["pred"])
    gt = np.asarray(data["gt"])
    fnames = list(data["fnames"])
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[1:] != (17, 3):
        raise ValueError(
            "expected pred/gt with identical (N,17,3) shapes, got "
            f"{pred.shape} and {gt.shape}"
        )
    if len(fnames) != len(pred):
        raise ValueError(
            f"filename count {len(fnames)} != prediction count {len(pred)}"
        )

    scale = 1000.0 if args.coordinate_unit == "meter" else 1.0
    joint_error_mm = np.linalg.norm(pred - gt, axis=-1) * scale
    actions = np.asarray([action_id(fname) for fname in fnames])

    per_action = {}
    for action, name in ACTION_NAMES.items():
        selected = actions == action
        if not selected.any():
            raise ValueError(f"no records found for H36M action {action} ({name})")
        errors = joint_error_mm[selected]
        per_action[name] = {
            "records": int(selected.sum()),
            "all17_mm": float(errors.mean()),
            "kp_star_mm": float(errors[:, KP_STAR_INDICES].mean()),
            "coco_indices_on_h36m_mm": float(
                errors[:, COCO_KP_STAR_INDICES_ON_H36M].mean()
            ),
        }

    action_equal_all = float(
        np.mean([entry["all17_mm"] for entry in per_action.values()])
    )
    action_equal_kp = float(
        np.mean([entry["kp_star_mm"] for entry in per_action.values()])
    )
    action_equal_coco_indices = float(
        np.mean(
            [
                entry["coco_indices_on_h36m_mm"]
                for entry in per_action.values()
            ]
        )
    )
    result = {
        "source": str(Path(args.dict_pkl).resolve()),
        "records": int(len(pred)),
        "kp_star_indices": KP_STAR_INDICES,
        "coco_kp_star_indices_applied_to_h36m_diagnostic": (
            COCO_KP_STAR_INDICES_ON_H36M
        ),
        "frame_weighted": {
            "all17_mm": float(joint_error_mm.mean()),
            "kp_star_mm": float(joint_error_mm[:, KP_STAR_INDICES].mean()),
            "coco_indices_on_h36m_mm": float(
                joint_error_mm[:, COCO_KP_STAR_INDICES_ON_H36M].mean()
            ),
        },
        "table2_action_equal": {
            "all17_mm": action_equal_all,
            "kp_star_mm": action_equal_kp,
            "coco_indices_on_h36m_mm": action_equal_coco_indices,
            "delta_to_paper_all17_mm": action_equal_all - PAPER_ALL_MM,
            "delta_to_paper_kp_star_mm": action_equal_kp - PAPER_KP_STAR_MM,
        },
        "per_action": per_action,
    }

    print("Per-action MPJPE (mm)")
    print(f"{'Action':12s} {'records':>7s} {'All-17':>9s} {'KP*':>9s}")
    for name, entry in per_action.items():
        print(
            f"{name:12s} {entry['records']:7d} "
            f"{entry['all17_mm']:9.3f} {entry['kp_star_mm']:9.3f}"
        )
    print()
    print(
        "Frame-weighted: "
        f"All-17={result['frame_weighted']['all17_mm']:.3f} mm, "
        f"KP*={result['frame_weighted']['kp_star_mm']:.3f} mm"
    )
    print(
        "COCO-index-on-H36M diagnostic (not the textual KP* joints): "
        f"frame={result['frame_weighted']['coco_indices_on_h36m_mm']:.3f} mm, "
        f"action-equal={action_equal_coco_indices:.3f} mm"
    )
    print(
        "Table-2 action-equal: "
        f"All-17={action_equal_all:.3f} mm, KP*={action_equal_kp:.3f} mm"
    )
    print(
        "Delta to paper (52.5/56.8): "
        f"All-17={action_equal_all - PAPER_ALL_MM:+.3f} mm, "
        f"KP*={action_equal_kp - PAPER_KP_STAR_MM:+.3f} mm"
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
