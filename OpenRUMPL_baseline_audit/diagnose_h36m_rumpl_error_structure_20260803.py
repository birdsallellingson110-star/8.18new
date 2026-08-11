#!/usr/bin/env python3
"""Compare protocol-matched RUMPL errors by joint and camera combination."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_h23_rumpl_pose_query_anchor import ACTION_NAMES
from eval_h36m_sparse_epipolar_topk import build_four_view_groups


JOINT_NAMES = [
    "root", "rhip", "rknee", "rankle", "lhip", "lknee", "lankle",
    "belly", "neck", "nose", "head", "lshoulder", "lelbow", "lwrist",
    "rshoulder", "relbow", "rwrist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-pkl", required=True)
    parser.add_argument("--experiment", action="append", nargs=2, metavar=("NAME", "EVAL_ROOT"), required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def action_equal(values: dict[str, list[np.ndarray]]) -> np.ndarray:
    return np.stack([np.stack(items).mean(axis=0) for items in values.values()]).mean(axis=0)


def load_prediction(root: Path, views: int) -> dict:
    matches = list((root / f"V{views}").glob("preds_gt_*_dict.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{root}/V{views}: expected one prediction, got {matches}")
    with matches[0].open("rb") as stream:
        return pickle.load(stream)


def main() -> None:
    args = parse_args()
    with open(args.dataset_pkl, "rb") as stream:
        records = pickle.load(stream)
    groups = build_four_view_groups(records)
    result = {
        "metric": "Human3.6M S9/S11 action-equal All-17 MPJPE (mm)",
        "dataset_pkl": args.dataset_pkl,
        "joint_order": JOINT_NAMES,
        "experiments": {},
    }
    for name, root_text in args.experiment:
        root = Path(root_text)
        experiment = {}
        for views in (2, 3, 4):
            frozen = load_prediction(root, views)
            prediction = np.asarray(frozen["pred"], dtype=np.float64)
            target = np.asarray(frozen["gt"], dtype=np.float64)
            work = [
                (group, combination)
                for group in groups
                for combination in itertools.combinations(range(4), views)
            ]
            if len(work) != len(prediction):
                raise ValueError(f"{name} V{views}: {len(work)} groups != {len(prediction)} predictions")
            per_joint_action: dict[str, list[np.ndarray]] = defaultdict(list)
            per_combo_action: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
            for index, (group, combination) in enumerate(work):
                action = ACTION_NAMES[int(records[group[0]]["action"])]
                joint_mm = np.linalg.norm(prediction[index] - target[index], axis=-1) * 1000.0
                per_joint_action[action].append(joint_mm)
                combo = "-".join(str(camera + 1) for camera in combination)
                per_combo_action[combo][action].append(joint_mm)
            joint_ae = action_equal(per_joint_action)
            combo_ae = {
                combo: float(action_equal(action_values).mean())
                for combo, action_values in sorted(per_combo_action.items())
            }
            experiment[f"V{views}"] = {
                "all17_mm": float(joint_ae.mean()),
                "per_joint_mm": {joint: float(value) for joint, value in zip(JOINT_NAMES, joint_ae)},
                "per_camera_combination_all17_mm": combo_ae,
            }
        result["experiments"][name] = experiment
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
