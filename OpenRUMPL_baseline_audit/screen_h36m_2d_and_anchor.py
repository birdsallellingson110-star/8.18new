#!/usr/bin/env python3
"""Screen aligned H36M 2D inputs by pixel error and RUMPL anchor MPJPE."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_h23_rumpl_pose_query_anchor import (
    ACTION_NAMES,
    rumpl_anchor_h36m,
    swap_lower_body,
    target_world_metres,
)
from eval_h36m_sparse_epipolar_topk import build_four_view_groups


JOINT_NAMES = [
    "root", "rhip", "rkne", "rank", "lhip", "lkne", "lank", "belly",
    "neck", "nose", "head", "lsho", "lelb", "lwri", "rsho", "relb", "rwri",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--input", action="append", required=True,
                        help="LABEL=PKL; repeat for each candidate")
    parser.add_argument("--confidence-epsilon", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load(path):
    with open(path, "rb") as stream:
        return pickle.load(stream)


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def main():
    args = parse_args()
    gt = load(args.ground_truth)
    gt_xy = np.stack([np.asarray(row["joints_2d"], np.float64) for row in gt])
    groups = build_four_view_groups(gt)
    output = {
        "ground_truth": args.ground_truth,
        "records": len(gt),
        "complete_four_view_groups": len(groups),
        "confidence_epsilon": args.confidence_epsilon,
        "regularization": args.regularization,
        "methods": {},
    }
    for specification in args.input:
        label, path = specification.split("=", 1)
        records = load(path)
        if len(records) != len(gt):
            raise ValueError(f"{label}: length mismatch")
        for index, (left, right) in enumerate(zip(gt, records)):
            if left["image"] != right["image"]:
                raise ValueError(f"{label}: image mismatch at {index}")
        xy_all = np.stack(
            [np.asarray(row["joints_2d"], np.float64) for row in records]
        )
        pixel_error = np.linalg.norm(xy_all - gt_xy, axis=-1)
        method = {
            "path": path,
            "pixels": distribution(pixel_error),
            "per_joint_mean_pixels": {
                name: float(pixel_error[:, joint].mean())
                for joint, name in enumerate(JOINT_NAMES)
            },
            "anchors": {},
        }
        for views in (2, 3, 4):
            errors = defaultdict(list)
            for group in groups:
                for combination in itertools.combinations(range(4), views):
                    indices = [group[position] for position in combination]
                    group_gt = [gt[index] for index in indices]
                    group_input = [records[index] for index in indices]
                    xy = np.stack(
                        [np.asarray(row["joints_2d"], np.float64) for row in group_input]
                    )
                    confidence = np.stack(
                        [
                            np.asarray(row["joints_2d_conf"], np.float64).reshape(17)
                            for row in group_input
                        ]
                    )
                    prediction = rumpl_anchor_h36m(
                        group_gt,
                        xy,
                        confidence,
                        args.confidence_epsilon,
                        args.regularization,
                    )
                    prediction = swap_lower_body(prediction[None])[0]
                    target = swap_lower_body(
                        target_world_metres(group_gt[0])[None]
                    )[0]
                    error = float(
                        np.linalg.norm(prediction - target, axis=-1).mean() * 1000.0
                    )
                    errors[ACTION_NAMES[int(group_gt[0]["action"])]].append(error)
            per_action = {
                action: float(np.mean(values))
                for action, values in sorted(errors.items())
            }
            method["anchors"][f"V{views}"] = {
                "action_equal_all17_mm": float(np.mean(list(per_action.values()))),
                "per_action_all17_mm": per_action,
                "samples": int(sum(len(values) for values in errors.values())),
            }
        output["methods"][label] = method
        print(
            label,
            f"2D={method['pixels']['mean']:.3f}",
            "/".join(
                f"{method['anchors'][f'V{views}']['action_equal_all17_mm']:.3f}"
                for views in (2, 3, 4)
            ),
            flush=True,
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
