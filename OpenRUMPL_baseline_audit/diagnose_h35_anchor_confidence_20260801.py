#!/usr/bin/env python3
"""Diagnose whether H35 is limited by its ray anchor or learned residual.

This script uses the exact H36M grouping/order used by the H23 diagnostics.
It is read-only: no model is trained and all scale sweeps are explicitly
labelled validation diagnostics rather than deployable selected settings.
"""

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
    KP_STAR,
    rumpl_anchor_h36m,
    swap_lower_body,
    target_world_metres,
)
from eval_h36m_sparse_epipolar_topk import build_four_view_groups
from pose_query_geometry import triangulate_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--rumpl-input-pkl", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--confidence-epsilon", type=float, default=0.05)
    return parser.parse_args()


def add(
    store: dict[str, dict[str, list[float]]],
    method: str,
    action: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    joint_error = np.linalg.norm(prediction - target, axis=-1) * 1000.0
    store[method][action].append(float(joint_error.mean()))
    store[method + "__kp"][action].append(
        float(joint_error[list(KP_STAR)].mean())
    )


def summarize(store: dict[str, dict[str, list[float]]]) -> dict:
    result = {}
    for method, action_values in sorted(store.items()):
        if method.endswith("__kp"):
            continue
        per_action = {
            action: float(np.mean(values))
            for action, values in sorted(action_values.items())
        }
        kp_values = store[method + "__kp"]
        per_action_kp = {
            action: float(np.mean(values))
            for action, values in sorted(kp_values.items())
        }
        flat = [value for values in action_values.values() for value in values]
        result[method] = {
            "frame_weighted_all17_mm": float(np.mean(flat)),
            "action_equal_all17_mm": float(np.mean(list(per_action.values()))),
            "action_equal_kp_star_mm": float(
                np.mean(list(per_action_kp.values()))
            ),
            "per_action_all17_mm": per_action,
        }
    return result


def load_prediction(root: Path, views: int) -> dict:
    matches = sorted((root / f"V{views}").glob("preds_gt_*_dict.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"V{views}: expected one prediction, got {matches}")
    with matches[0].open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = parse_args()
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    with open(args.rumpl_input_pkl, "rb") as handle:
        rumpl_records = pickle.load(handle)
    if len(records) != len(rumpl_records):
        raise ValueError("GT and RUMPL input lengths differ")
    for index, (gt_record, input_record) in enumerate(zip(records, rumpl_records)):
        if gt_record["image"] != input_record["image"]:
            raise ValueError(f"record {index} image mismatch")

    groups = build_four_view_groups(records)
    result = {
        "warning": (
            "Residual scales and confidence transforms are validation-only "
            "diagnostics, not selected deployable hyperparameters."
        ),
        "input_pkl": args.input_pkl,
        "rumpl_input_pkl": args.rumpl_input_pkl,
        "prediction_root": args.prediction_root,
        "views": {},
    }
    for views in (2, 3, 4):
        frozen = load_prediction(Path(args.prediction_root), views)
        predictions = np.asarray(frozen["pred"], dtype=np.float64)
        targets = np.asarray(frozen["gt"], dtype=np.float64)
        work = [
            (group, combination)
            for group in groups
            for combination in itertools.combinations(range(4), views)
        ]
        if len(work) != len(predictions):
            raise ValueError(
                f"V{views}: combinations={len(work)} predictions={len(predictions)}"
            )
        metrics: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        residual_norms = []
        for prediction_index, (group, combination) in enumerate(work):
            indices = [group[position] for position in combination]
            group_records = [records[index] for index in indices]
            group_inputs = [rumpl_records[index] for index in indices]
            xy = np.stack(
                [np.asarray(item["joints_2d"], dtype=np.float64) for item in group_inputs]
            )
            confidence = np.stack(
                [
                    np.asarray(item["joints_2d_conf"], dtype=np.float64).reshape(17)
                    for item in group_inputs
                ]
            )
            target = targets[prediction_index]
            record_target = swap_lower_body(
                target_world_metres(group_records[0])[None]
            )[0]
            if np.max(np.abs(record_target - target)) > 1e-5:
                raise ValueError("prediction target ordering mismatch")
            action = ACTION_NAMES[int(group_records[0]["action"])]

            def anchor(weights: np.ndarray, epsilon: float) -> np.ndarray:
                value = rumpl_anchor_h36m(
                    group_records,
                    xy,
                    weights,
                    epsilon,
                    args.regularization,
                )
                return swap_lower_body(value[None])[0]

            baseline_anchor = anchor(confidence, args.confidence_epsilon)
            uniform_anchor = anchor(np.ones_like(confidence), 0.0)
            sqrt_anchor = anchor(np.sqrt(np.clip(confidence, 0.0, 1.0)), 0.0)
            squared_anchor = anchor(np.square(np.clip(confidence, 0.0, 1.0)), 0.0)
            irls = triangulate_points(group_records, xy, confidence, 5)
            irls = swap_lower_body(irls[None])[0]

            prediction = predictions[prediction_index]
            add(metrics, "rumpl_prediction", action, prediction, target)
            add(metrics, "anchor_conf_eps005", action, baseline_anchor, target)
            add(metrics, "anchor_uniform", action, uniform_anchor, target)
            add(metrics, "anchor_sqrt_conf", action, sqrt_anchor, target)
            add(metrics, "anchor_squared_conf", action, squared_anchor, target)
            add(metrics, "anchor_irls5", action, irls, target)

            residual = prediction - baseline_anchor
            residual_norms.append(float(np.linalg.norm(residual, axis=-1).mean() * 1000.0))
            for scale in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
                add(
                    metrics,
                    f"anchor_plus_{scale:g}x_rumpl_residual",
                    action,
                    baseline_anchor + scale * residual,
                    target,
                )

        result["views"][f"V{views}"] = {
            "records": len(work),
            "mean_rumpl_residual_norm_mm": float(np.mean(residual_norms)),
            "metrics": summarize(metrics),
        }
        print(
            f"V{views}",
            json.dumps(
                {
                    name: round(values["action_equal_all17_mm"], 3)
                    for name, values in result["views"][f"V{views}"]["metrics"].items()
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
