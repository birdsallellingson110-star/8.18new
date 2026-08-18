#!/usr/bin/env python3
"""Fit sample-independent H76/Volumetric joint weights on train subjects.

The 17 convex weights are fitted only on H36M train subjects and then frozen
for S9/S11.  This makes the reported validation metric a legitimate held-out
evaluation, unlike the complementarity oracle diagnostic.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


COMBOS = ("1-2", "1-3", "1-4", "2-3", "2-4", "3-4")
ACTION_IDS = tuple(range(2, 17))
VOL_PREFIX = "backbone_prediction_lt_to_rumpl_control_V2_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h76-pkl", required=True)
    parser.add_argument("--train-vol-npz", required=True)
    parser.add_argument("--test-h76-pkl", required=True)
    parser.add_argument("--test-vol-npz", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_pair(h76_path: str, vol_path: str) -> tuple[np.ndarray, ...]:
    with open(h76_path, "rb") as stream:
        h76_data = pickle.load(stream)
    vol_data = np.load(vol_path)
    h76 = np.asarray(h76_data["pred"], dtype=np.float64) * 1000.0
    h76_gt = np.asarray(h76_data["gt"], dtype=np.float64) * 1000.0
    if len(h76) % len(COMBOS):
        raise ValueError("H76 records are not six-combination frame chunks")
    groups = len(h76) // len(COMBOS)
    h76 = h76.reshape(groups, len(COMBOS), 17, 3)
    h76_gt = h76_gt.reshape(groups, len(COMBOS), 17, 3)
    vol = np.stack([
        np.asarray(vol_data[VOL_PREFIX + combo.replace("-", "_")])
        for combo in COMBOS
    ], axis=1).astype(np.float64)
    target = np.asarray(vol_data["targets"], dtype=np.float64)
    actions = np.asarray(vol_data["actions"], dtype=np.int64)
    if h76.shape != vol.shape:
        raise ValueError(f"prediction mismatch {h76.shape} != {vol.shape}")
    if target.shape != h76_gt[:, 0].shape:
        raise ValueError(f"target mismatch {target.shape} != {h76_gt[:, 0].shape}")
    repeat_delta = float(np.abs(h76_gt - h76_gt[:, :1]).max())
    cross_delta = np.linalg.norm(h76_gt[:, 0] - target, axis=-1)
    if repeat_delta > 1e-3 or cross_delta.max() > 1e-2:
        raise ValueError(
            f"GT alignment failed repeat={repeat_delta} cross={cross_delta.max()}"
        )
    return h76, vol, target, actions, np.asarray([
        repeat_delta, cross_delta.mean(), cross_delta.max()
    ])


def summarize(prediction: np.ndarray, target: np.ndarray, actions: np.ndarray) -> dict:
    target = np.broadcast_to(target[:, None], prediction.shape)
    frame_error = np.linalg.norm(prediction - target, axis=-1).mean(-1)
    return {
        "frame_weighted_mm": float(frame_error.mean()),
        "action_equal_mm": float(np.mean([
            frame_error[actions == action].mean()
            for action in ACTION_IDS
        ])),
        "per_pair_action_equal_mm": {
            combo: float(np.mean([
                frame_error[actions == action, combo_index].mean()
                for action in ACTION_IDS
            ]))
            for combo_index, combo in enumerate(COMBOS)
        },
    }


def main() -> None:
    args = parse_args()
    train_h76, train_vol, train_target, train_actions, train_gt_delta = load_pair(
        args.train_h76_pkl, args.train_vol_npz
    )
    test_h76, test_vol, test_target, test_actions, test_gt_delta = load_pair(
        args.test_h76_pkl, args.test_vol_npz
    )
    alpha_grid = np.linspace(0.0, 1.0, 101)
    train_target_expanded = np.broadcast_to(
        train_target[:, None], train_h76.shape
    )
    alphas = []
    train_losses = []
    for joint in range(17):
        losses = np.asarray([
            np.linalg.norm(
                alpha * train_h76[..., joint, :]
                + (1.0 - alpha) * train_vol[..., joint, :]
                - train_target_expanded[..., joint, :],
                axis=-1,
            ).mean()
            for alpha in alpha_grid
        ])
        best_index = int(np.argmin(losses))
        alphas.append(float(alpha_grid[best_index]))
        train_losses.append(float(losses[best_index]))
    alpha_array = np.asarray(alphas)[None, None, :, None]
    train_blend = alpha_array * train_h76 + (1.0 - alpha_array) * train_vol
    test_blend = alpha_array * test_h76 + (1.0 - alpha_array) * test_vol
    result = {
        "protocol": {
            "fit_subjects": [1, 5, 6, 7, 8],
            "test_subjects": [9, 11],
            "fit_is_sample_independent": True,
            "fit_parameters": 17,
            "fit_frames": int(len(train_h76)),
            "test_frames": int(len(test_h76)),
            "train_h76_pkl": str(Path(args.train_h76_pkl).resolve()),
            "train_vol_npz": str(Path(args.train_vol_npz).resolve()),
            "test_h76_pkl": str(Path(args.test_h76_pkl).resolve()),
            "test_vol_npz": str(Path(args.test_vol_npz).resolve()),
            "train_gt_alignment_mm": train_gt_delta.tolist(),
            "test_gt_alignment_mm": test_gt_delta.tolist(),
        },
        "h76_alpha_per_joint": alphas,
        "fit_joint_losses_mm": train_losses,
        "train": {
            "H76": summarize(train_h76, train_target, train_actions),
            "Volumetric": summarize(train_vol, train_target, train_actions),
            "fixed_joint_blend": summarize(
                train_blend, train_target, train_actions
            ),
        },
        "test": {
            "H76": summarize(test_h76, test_target, test_actions),
            "Volumetric": summarize(test_vol, test_target, test_actions),
            "fixed_joint_blend": summarize(
                test_blend, test_target, test_actions
            ),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
