#!/usr/bin/env python3
"""Measure whether H76 and official Volumetric LT are worth fusing.

This is an upper-bound diagnostic, not a paper result: any strategy that uses
validation GT to choose a method or blend is explicitly labelled oracle.
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
    parser.add_argument("--h76-pkl", required=True)
    parser.add_argument("--vol-npz", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def action_equal(frame_errors: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        frame_errors[actions == action].mean() for action in ACTION_IDS
    ]))


def summarize(prediction: np.ndarray, target: np.ndarray, actions: np.ndarray) -> dict:
    frame_errors = np.linalg.norm(prediction - target, axis=-1).mean(-1)
    return {
        "frame_weighted_mm": float(frame_errors.mean()),
        "action_equal_mm": action_equal(frame_errors, actions),
    }


def main() -> None:
    args = parse_args()
    with open(args.h76_pkl, "rb") as stream:
        h76_data = pickle.load(stream)
    vol_data = np.load(args.vol_npz)

    h76_prediction = np.asarray(h76_data["pred"], dtype=np.float64) * 1000.0
    h76_target = np.asarray(h76_data["gt"], dtype=np.float64) * 1000.0
    if h76_prediction.shape[0] % len(COMBOS):
        raise ValueError("H76 V2 records are not divisible by six camera pairs")
    groups = h76_prediction.shape[0] // len(COMBOS)
    h76_prediction = h76_prediction.reshape(groups, len(COMBOS), 17, 3)
    h76_target = h76_target.reshape(groups, len(COMBOS), 17, 3)

    vol_target = np.asarray(vol_data["targets"], dtype=np.float64)
    actions = np.asarray(vol_data["actions"], dtype=np.int64)
    vol_prediction = np.stack([
        np.asarray(vol_data[VOL_PREFIX + combo.replace("-", "_")], dtype=np.float64)
        for combo in COMBOS
    ], axis=1)
    if vol_prediction.shape != h76_prediction.shape:
        raise ValueError(
            f"prediction shape mismatch: H76 {h76_prediction.shape}, "
            f"Volumetric {vol_prediction.shape}"
        )
    if vol_target.shape != h76_target[:, 0].shape:
        raise ValueError(
            f"target shape mismatch: H76 {h76_target[:, 0].shape}, "
            f"Volumetric {vol_target.shape}"
        )
    repeated_target_spread = np.abs(
        h76_target - h76_target[:, :1]
    ).max()
    target_delta = np.linalg.norm(
        h76_target[:, 0] - vol_target, axis=-1
    )
    if repeated_target_spread > 1e-3 or target_delta.max() > 1e-2:
        raise ValueError(
            "GT alignment failed: "
            f"repeat_max={repeated_target_spread}, cross_pipeline_max={target_delta.max()}"
        )
    target = np.broadcast_to(vol_target[:, None], vol_prediction.shape)
    tiled_actions = np.broadcast_to(actions[:, None], (groups, len(COMBOS)))

    h76_joint_error = np.linalg.norm(h76_prediction - target, axis=-1)
    vol_joint_error = np.linalg.norm(vol_prediction - target, axis=-1)
    h76_frame_error = h76_joint_error.mean(-1)
    vol_frame_error = vol_joint_error.mean(-1)

    result = {
        "protocol": {
            "h76_pkl": str(Path(args.h76_pkl).resolve()),
            "vol_npz": str(Path(args.vol_npz).resolve()),
            "groups": groups,
            "combos": list(COMBOS),
            "gt_repeat_max_mm": float(repeated_target_spread),
            "gt_cross_pipeline_mean_mm": float(target_delta.mean()),
            "gt_cross_pipeline_max_mm": float(target_delta.max()),
            "warning": "GT-selected blends and selectors are oracle diagnostics only",
        },
        "methods": {
            "H76": summarize(
                h76_prediction.reshape(-1, 17, 3),
                target.reshape(-1, 17, 3),
                tiled_actions.reshape(-1),
            ),
            "Volumetric": summarize(
                vol_prediction.reshape(-1, 17, 3),
                target.reshape(-1, 17, 3),
                tiled_actions.reshape(-1),
            ),
        },
        "per_pair": {},
    }

    # Plain fixed blends do not use GT at inference.  The alpha sweep is still
    # selected on validation GT and therefore only estimates potential.
    fixed_blends = {}
    for alpha in np.linspace(0.0, 1.0, 21):
        prediction = alpha * h76_prediction + (1.0 - alpha) * vol_prediction
        fixed_blends[f"h76_{alpha:.2f}"] = summarize(
            prediction.reshape(-1, 17, 3),
            target.reshape(-1, 17, 3),
            tiled_actions.reshape(-1),
        )
    best_fixed_key = min(
        fixed_blends,
        key=lambda key: fixed_blends[key]["action_equal_mm"],
    )
    result["fixed_blend_validation_sweep"] = {
        "best": best_fixed_key,
        "best_metrics": fixed_blends[best_fixed_key],
        "all": fixed_blends,
    }

    # Joint-specific but sample-independent convex weights are a much weaker
    # model than a learned utility head.  Report both the validation optimum
    # and leave-one-action-out weights to test whether complementarity is
    # stable across actions rather than driven by direct per-frame GT access.
    alpha_grid = np.linspace(0.0, 1.0, 101)
    joint_alphas = []
    for joint in range(17):
        losses = [
            np.linalg.norm(
                alpha * h76_prediction[..., joint, :]
                + (1.0 - alpha) * vol_prediction[..., joint, :]
                - target[..., joint, :],
                axis=-1,
            ).mean()
            for alpha in alpha_grid
        ]
        joint_alphas.append(float(alpha_grid[int(np.argmin(losses))]))
    joint_alphas_array = np.asarray(joint_alphas)[None, None, :, None]
    fixed_joint_prediction = (
        joint_alphas_array * h76_prediction
        + (1.0 - joint_alphas_array) * vol_prediction
    )
    result["fixed_per_joint_blend_validation_sweep"] = {
        "h76_alphas": joint_alphas,
        "metrics": summarize(
            fixed_joint_prediction.reshape(-1, 17, 3),
            target.reshape(-1, 17, 3),
            tiled_actions.reshape(-1),
        ),
    }

    loa_predictions = np.empty_like(h76_prediction)
    loa_alphas = {}
    for held_out_action in ACTION_IDS:
        training_mask = actions != held_out_action
        testing_mask = actions == held_out_action
        action_alphas = []
        for joint in range(17):
            losses = [
                np.linalg.norm(
                    alpha * h76_prediction[training_mask, :, joint, :]
                    + (1.0 - alpha)
                    * vol_prediction[training_mask, :, joint, :]
                    - target[training_mask, :, joint, :],
                    axis=-1,
                ).mean()
                for alpha in alpha_grid
            ]
            action_alphas.append(float(alpha_grid[int(np.argmin(losses))]))
        alpha_array = np.asarray(action_alphas)[None, None, :, None]
        loa_predictions[testing_mask] = (
            alpha_array * h76_prediction[testing_mask]
            + (1.0 - alpha_array) * vol_prediction[testing_mask]
        )
        loa_alphas[str(held_out_action)] = action_alphas
    result["fixed_per_joint_blend_leave_one_action_out"] = {
        "h76_alphas_by_held_out_action": loa_alphas,
        "metrics": summarize(
            loa_predictions.reshape(-1, 17, 3),
            target.reshape(-1, 17, 3),
            tiled_actions.reshape(-1),
        ),
    }

    pair_selected_prediction = np.empty_like(h76_prediction)
    for combo_index, combo in enumerate(COMBOS):
        h76_summary = summarize(
            h76_prediction[:, combo_index], vol_target, actions
        )
        vol_summary = summarize(
            vol_prediction[:, combo_index], vol_target, actions
        )
        selected = (
            "H76" if h76_summary["action_equal_mm"]
            <= vol_summary["action_equal_mm"] else "Volumetric"
        )
        pair_selected_prediction[:, combo_index] = (
            h76_prediction[:, combo_index]
            if selected == "H76" else vol_prediction[:, combo_index]
        )
        result["per_pair"][combo] = {
            "H76": h76_summary,
            "Volumetric": vol_summary,
            "validation_oracle_selected_method": selected,
            "H76_frame_win_rate": float(
                (h76_frame_error[:, combo_index]
                 < vol_frame_error[:, combo_index]).mean()
            ),
            "frame_error_pearson": float(np.corrcoef(
                h76_frame_error[:, combo_index],
                vol_frame_error[:, combo_index],
            )[0, 1]),
        }
    result["pair_identity_selector_validation_oracle"] = summarize(
        pair_selected_prediction.reshape(-1, 17, 3),
        target.reshape(-1, 17, 3),
        tiled_actions.reshape(-1),
    )

    choose_h76_frame = h76_frame_error <= vol_frame_error
    frame_oracle = np.where(
        choose_h76_frame[..., None, None], h76_prediction, vol_prediction
    )
    result["frame_method_selector_oracle"] = summarize(
        frame_oracle.reshape(-1, 17, 3),
        target.reshape(-1, 17, 3),
        tiled_actions.reshape(-1),
    )
    result["frame_oracle_H76_choice_rate"] = float(choose_h76_frame.mean())

    choose_h76_joint = h76_joint_error <= vol_joint_error
    joint_oracle = np.where(
        choose_h76_joint[..., None], h76_prediction, vol_prediction
    )
    result["joint_method_selector_oracle"] = summarize(
        joint_oracle.reshape(-1, 17, 3),
        target.reshape(-1, 17, 3),
        tiled_actions.reshape(-1),
    )
    result["joint_oracle_H76_choice_rate"] = float(choose_h76_joint.mean())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
