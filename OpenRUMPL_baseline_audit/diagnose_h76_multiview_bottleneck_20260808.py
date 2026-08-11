#!/usr/bin/env python3
"""Audit where H76 loses multi-view information on real Human3.6M.

This is a read-only diagnostic.  It does not train a model and does not use
test GT to choose a view.  It pairs the already exported H76 predictions with
the exact rays produced by the real-H36M dataset and reports:

* nested camera-subset gains (V2 -> V3 -> V4),
* absolute, root-relative, and per-joint errors,
* the confidence-weighted triangulation anchor used by H76,
* geometry/confidence statistics and their sample-level correlations with
  anchor/final error.

The anchor formula intentionally mirrors MultiView_RUMPL.forward():
confidence weight = confidence + 0.05 and Tikhonov regularization = 1e-4.
The prediction files are the existing action-equal H76 validation exports;
no prediction is regenerated here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))

import dataset  # noqa: E402
from core.config import config, update_config  # noqa: E402


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
JOINT_NAMES = [
    "root", "rhip", "rknee", "rankle", "lhip", "lknee", "lankle",
    "belly", "neck", "nose", "head", "lshoulder", "lelbow", "lwrist",
    "rshoulder", "relbow", "rwrist",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mmpose-type", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="optional prefix smoke test; 0 evaluates every validation group",
    )
    return parser.parse_args()


def build_four_view_groups(records: list[dict]) -> list[list[int]]:
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    for index, record in enumerate(records):
        key = (
            int(record["subject"]),
            int(record["action"]),
            int(record["subaction"]),
            int(record["image_id"]),
        )
        grouped.setdefault(key, [-1, -1, -1, -1])[
            int(record["camera_id"])
        ] = index
    return [group for group in grouped.values() if min(group) >= 0]


def action_equal(values: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Mean over each action first, then arithmetic mean over 15 actions."""
    per_action = []
    for action in ACTION_NAMES:
        selected = actions == action
        if not np.any(selected):
            # A --max-groups smoke run may intentionally contain only a prefix
            # of actions.  Full runs still cover all 15 actions and therefore
            # retain the exact Table-II arithmetic mean.
            continue
        per_action.append(np.mean(values[selected], axis=0))
    if not per_action:
        raise ValueError("no recognized H36M action in diagnostic records")
    return np.mean(np.stack(per_action, axis=0), axis=0)


def action_equal_scalar(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.asarray(action_equal(values, actions)).mean())


def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) < 1e-12 or np.std(y[valid]) < 1e-12:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def solve_anchor(rays: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mirror the H76 confidence-weighted least-squares ray anchor.

    rays: (N,J,V,7), [direction(3), point-on-ray(3), confidence(1)].
    """
    direction = rays[..., :3]
    point = rays[..., 3:6]
    confidence = np.clip(rays[..., 6:7], 0.0, 1.0)
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True).clip(
        1e-7
    )
    eye = np.eye(3, dtype=np.float64)
    projection = eye - direction[..., :, None] * direction[..., None, :]
    weights = confidence + 0.05
    weighted = weights[..., None] * projection
    lhs = weighted.sum(axis=2) + 1e-4 * eye
    rhs = np.sum(weighted @ point[..., None], axis=2)[..., 0]
    anchor = np.linalg.solve(lhs, rhs)

    eigenvalues = np.linalg.eigvalsh(lhs - 1e-4 * eye)
    fractions = eigenvalues / np.clip(eigenvalues.sum(axis=-1, keepdims=True), 1e-9, None)
    # Use the symmetric skew-line distance for the reported geometry statistic.
    d_i = direction[:, :, :, None, :]
    d_j = direction[:, :, None, :, :]
    p_diff = point[:, :, None, :, :] - point[:, :, :, None, :]
    cross = np.cross(d_i, d_j)
    cross_norm = np.linalg.norm(cross, axis=-1)
    skew = np.abs(np.sum(p_diff * cross, axis=-1)) / np.clip(cross_norm, 1e-7, None)
    parallel = np.linalg.norm(np.cross(p_diff, d_i), axis=-1)
    line_distance = np.where(cross_norm > 1e-7, skew, parallel)
    diagonal = np.eye(line_distance.shape[-1], dtype=bool)
    line_distance[..., diagonal] = 0.0
    return anchor, {
        "mean_confidence": confidence.mean(axis=2)[..., 0],
        "mean_line_distance": line_distance.sum(axis=(-2, -1))
        / np.maximum(2 * rays.shape[2] * (rays.shape[2] - 1), 1),
        "lambda_min_fraction": fractions[..., 0],
        "lambda_condition": fractions[..., -1]
        / np.clip(fractions[..., 0], 1e-9, None),
    }


def load_predictions(root: Path, views: int) -> tuple[np.ndarray, np.ndarray]:
    matches = list((root / f"V{views}").glob("preds_gt_*_dict.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{root}/V{views}: expected one prediction export, found {matches}"
        )
    with matches[0].open("rb") as stream:
        payload = pickle.load(stream)
    return np.asarray(payload["pred"], dtype=np.float64), np.asarray(
        payload["gt"], dtype=np.float64
    )


def serialize_array(values: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(values).reshape(-1)]


def main() -> None:
    args = parse_args()
    update_config(args.cfg)
    config.DATASET.TEST_H36M_DATASET_NAME = args.dataset_name
    config.DATASET.TEST_MMPOSE_TYPE = args.mmpose_type
    config.DATASET.USE_MMPOSE_VAL = True
    config.DATASET.USE_MMPOSE_TEST = True
    config.DATASET.TEST_ON_ALL_CAMERAS = True
    config.DATASET.TEST_VIEWS = [1, 2, 3, 4]
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4

    dataset_class = getattr(dataset, config.DATASET.TEST_DATASET)
    base_dataset = dataset_class(config, config.DATASET.TEST_SUBSET, False, None)
    groups = build_four_view_groups(base_dataset.db)
    if len(groups) != len(base_dataset.grouping):
        raise RuntimeError(
            f"raw groups {len(groups)} != dataset groups {len(base_dataset.grouping)}"
        )
    if args.max_groups:
        groups = groups[: args.max_groups]
    group_actions = np.asarray(
        [int(base_dataset.db[group[0]]["action"]) for group in groups], dtype=np.int64
    )

    # Calling the dataset once per group gives exactly the canonical 4-view ray
    # tensor used by RUMPL evaluation, including the A1D/H21 lower-body mapping.
    rays_all = []
    targets_all = []
    for group_index in range(len(groups)):
        _, _, target, rays, _, _ = base_dataset[group_index]
        rays_all.append(rays.numpy())
        targets_all.append(target.numpy())
    rays_all = np.asarray(rays_all, dtype=np.float64)
    targets_all = np.asarray(targets_all, dtype=np.float64)

    output_root = Path(args.output).resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    result: dict = {
        "metric": "Human3.6M S9/S11 action-equal All-17 MPJPE (mm)",
        "cfg": str(Path(args.cfg).resolve()),
        "dataset_name": args.dataset_name,
        "mmpose_type": args.mmpose_type,
        "groups": int(len(groups)),
        "joint_order": JOINT_NAMES,
        "anchor_formula": {
            "confidence_offset": 0.05,
            "regularization": 1e-4,
            "description": "H76 confidence-weighted least-squares ray intersection",
        },
        "views": {},
    }

    for views in (2, 3, 4):
        prediction, target_from_export = load_predictions(
            Path(args.prediction_root), views
        )
        combinations = list(itertools.combinations(range(4), views))
        expected = len(groups) * len(combinations)
        if prediction.shape[0] != expected and not (
            args.max_groups and prediction.shape[0] >= expected
        ):
            raise ValueError(
                f"V{views}: predictions={prediction.shape[0]} but expected "
                f"{expected} for {len(groups)} groups and {len(combinations)} subsets"
            )
        if args.max_groups:
            prediction = prediction[:expected]
            target_from_export = target_from_export[:expected]
        target = target_from_export
        # The RUMPL validation exporter is group-major: for each synchronized
        # frame it emits all C(4,k) subsets.  Keep that ordering explicit.
        expected_target = np.repeat(
            targets_all[:, None, :, :], len(combinations), axis=1
        ).reshape(expected, 17, 3)
        if not np.allclose(target, expected_target, atol=2e-4):
            raise ValueError(
                f"V{views}: exported GT is not aligned with dataset grouping; "
                f"max diff={np.max(np.abs(target - expected_target)):.6g}"
            )

        subset_result: dict = {"subsets": {}, "nested_marginal": {}}
        prediction_by_group = prediction.reshape(len(groups), len(combinations), 17, 3)
        target_by_group = target.reshape(len(groups), len(combinations), 17, 3)
        final_errors = np.linalg.norm(prediction - target, axis=-1) * 1000.0
        final_root_relative = np.linalg.norm(
            (prediction - prediction[:, :1]) - (target - target[:, :1]), axis=-1
        ) * 1000.0

        for subset_index, combination in enumerate(combinations):
            ray_subset = rays_all[:, :, combination, :]
            anchor, geometry = solve_anchor(ray_subset)
            # Predictions are group-major, so a subset is the second axis.
            pred_subset = prediction_by_group[:, subset_index]
            target_subset = target_by_group[:, subset_index]
            anchor_error = np.linalg.norm(anchor - target_subset, axis=-1) * 1000.0
            final_error = np.linalg.norm(pred_subset - target_subset, axis=-1) * 1000.0
            anchor_root_relative = np.linalg.norm(
                (anchor - anchor[:, :1]) - (target_subset - target_subset[:, :1]),
                axis=-1,
            ) * 1000.0
            final_rr = np.linalg.norm(
                (pred_subset - pred_subset[:, :1])
                - (target_subset - target_subset[:, :1]),
                axis=-1,
            ) * 1000.0
            combo_name = "-".join(str(x + 1) for x in combination)
            subset_result["subsets"][combo_name] = {
                "groups": int(len(groups)),
                "final_all17_mm": action_equal_scalar(final_error, group_actions),
                "anchor_all17_mm": action_equal_scalar(anchor_error, group_actions),
                "final_root_mm": float(action_equal(final_error, group_actions)[0]),
                "anchor_root_mm": float(action_equal(anchor_error, group_actions)[0]),
                "final_root_relative_nonroot16_mm": float(
                    action_equal(final_rr, group_actions)[1:].mean()
                ),
                "anchor_root_relative_nonroot16_mm": float(
                    action_equal(anchor_root_relative, group_actions)[1:].mean()
                ),
                "final_minus_anchor_mm": action_equal_scalar(
                    final_error - anchor_error, group_actions
                ),
                "residual_norm_mm": action_equal_scalar(
                    np.linalg.norm(pred_subset - anchor, axis=-1) * 1000.0,
                    group_actions,
                ),
                "per_joint_final_mm": {
                    name: float(value)
                    for name, value in zip(
                        JOINT_NAMES, action_equal(final_error, group_actions)
                    )
                },
                "per_joint_anchor_mm": {
                    name: float(value)
                    for name, value in zip(
                        JOINT_NAMES, action_equal(anchor_error, group_actions)
                    )
                },
                "geometry_action_equal": {
                    key: float(np.mean(action_equal(value, group_actions)))
                    for key, value in geometry.items()
                },
                "sample_correlations": {
                    key: safe_corr(value.mean(axis=1), final_error.mean(axis=1))
                    for key, value in geometry.items()
                },
            }

        subset_result["action_equal_overall"] = {
            "final_all17_mm": action_equal_scalar(final_errors, np.repeat(group_actions, len(combinations))),
            "final_root_relative_nonroot16_mm": float(
                action_equal(final_root_relative, np.repeat(group_actions, len(combinations)))[1:].mean()
            ),
        }
        result["views"][f"V{views}"] = subset_result

    # Nested gains are computed only after all cardinalities are available and
    # use the same group/action ordering.  A negative delta means improvement.
    nested: dict[str, dict[str, float]] = {}
    for pair in itertools.combinations(range(4), 2):
        pair_name = "-".join(str(x + 1) for x in pair)
        pair_value = result["views"]["V2"]["subsets"][pair_name]["final_all17_mm"]
        for added in sorted(set(range(4)) - set(pair)):
            triple = tuple(sorted((*pair, added)))
            triple_name = "-".join(str(x + 1) for x in triple)
            triple_value = result["views"]["V3"]["subsets"][triple_name]["final_all17_mm"]
            nested[f"V2:{pair_name}+{added + 1}->{triple_name}"] = {
                "from_mm": pair_value,
                "to_mm": triple_value,
                "delta_mm": triple_value - pair_value,
            }
    full_value = result["views"]["V4"]["subsets"]["1-2-3-4"]["final_all17_mm"]
    for triple in itertools.combinations(range(4), 3):
        triple_name = "-".join(str(x + 1) for x in triple)
        triple_value = result["views"]["V3"]["subsets"][triple_name]["final_all17_mm"]
        nested[f"V3:{triple_name}+missing->1-2-3-4"] = {
            "from_mm": triple_value,
            "to_mm": full_value,
            "delta_mm": full_value - triple_value,
        }
    result["nested_marginal"] = nested

    output_root.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown = output_root.with_suffix(".md")
    lines = [
        "# H76 real-H36M multi-view bottleneck audit",
        "",
        f"- Groups: {len(groups)}",
        f"- Dataset: `{args.dataset_name}`",
        f"- Input: `{args.mmpose_type}`",
        "- Negative nested delta means the added camera improved MPJPE.",
        "",
        "## Nested camera gains",
        "",
        "| Transition | From (mm) | To (mm) | Delta (mm) |",
        "|---|---:|---:|---:|",
    ]
    for key, value in result["nested_marginal"].items():
        lines.append(
            f"| {key} | {value['from_mm']:.3f} | "
            f"{value['to_mm']:.3f} | {value['delta_mm']:+.3f} |"
        )
    lines.extend(["", "## Per-subset final and anchor MPJPE", ""])
    lines.extend([
        "| Views | Subset | Final (mm) | Anchor (mm) | Final−anchor (mm) |",
        "|---|---|---:|---:|---:|",
    ])
    for views in (2, 3, 4):
        for subset, value in result["views"][f"V{views}"]["subsets"].items():
            lines.append(
                f"| V{views} | {subset} | {value['final_all17_mm']:.3f} | "
                f"{value['anchor_all17_mm']:.3f} | {value['final_minus_anchor_mm']:+.3f} |"
            )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_root)
    print(markdown)


if __name__ == "__main__":
    main()
