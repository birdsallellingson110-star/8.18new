#!/usr/bin/env python3
"""Compare aligned H36M MMPose-format PKLs against the real 2D labels."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "root", "rhip", "rkne", "rank", "lhip", "lkne", "lank", "belly",
    "neck", "nose", "head", "lsho", "lelb", "lwri", "rsho", "relb", "rwri",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="LABEL=PKL; repeat for every method")
    parser.add_argument("--reference-label", default=None,
                        help="Also report movement from this input method")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def load(path):
    with open(path, "rb") as stream:
        records = pickle.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty list: {path}")
    return records


def aligned_xy(gt, records, path):
    if len(gt) != len(records):
        raise ValueError(f"Length mismatch for {path}: {len(records)} != {len(gt)}")
    for index, (left, right) in enumerate(zip(gt, records)):
        if left["image"] != right["image"]:
            raise ValueError(
                f"Image mismatch at {index}: {left['image']} != {right['image']}"
            )
    return np.stack([np.asarray(row["joints_2d"], np.float64) for row in records])


def distribution(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def main():
    args = parse_args()
    gt = load(args.ground_truth)
    gt_xy = aligned_xy(gt, gt, args.ground_truth)
    methods = {}
    method_xy = {}
    for specification in args.input:
        if "=" not in specification:
            raise ValueError(f"Expected LABEL=PKL, got: {specification}")
        label, path = specification.split("=", 1)
        records = load(path)
        xy = aligned_xy(gt, records, path)
        error = np.linalg.norm(xy - gt_xy, axis=-1)
        action_values = {}
        for action in sorted({int(row["action"]) for row in gt}):
            keep = np.asarray([int(row["action"]) == action for row in gt])
            action_values[str(action)] = distribution(error[keep])
        methods[label] = {
            "path": path,
            "all_joints_pixels": distribution(error),
            "per_joint_mean_pixels": {
                name: float(error[:, joint].mean())
                for joint, name in enumerate(JOINT_NAMES)
            },
            "per_action_pixels": action_values,
        }
        method_xy[label] = xy
    if args.reference_label is not None:
        reference = method_xy[args.reference_label]
        for label, xy in method_xy.items():
            movement = np.linalg.norm(xy - reference, axis=-1)
            methods[label][f"movement_from_{args.reference_label}_pixels"] = distribution(
                movement
            )
    output = {
        "ground_truth": args.ground_truth,
        "num_records": len(gt),
        "joint_names": JOINT_NAMES,
        "methods": methods,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
