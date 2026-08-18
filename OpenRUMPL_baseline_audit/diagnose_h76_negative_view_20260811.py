#!/usr/bin/env python3
"""Measure joint-level negative-view effects for the frozen H76 baseline.

This is a read-only Stage-A diagnostic.  It aligns the already exported strict
V2/V3/V4 H76 predictions by synchronized Human3.6M frame and camera subset,
then computes the counterfactual change caused by adding one camera.  Positive
``delta_error_mm`` means the added view made the prediction worse.

The compressed NPZ is deliberately suitable for training a later utility head:
it contains per-transition, per-frame and per-joint counterfactual targets plus
geometry/confidence diagnostics, but no learned predictions are produced here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))

import dataset  # noqa: E402
from core.config import config, update_config  # noqa: E402

from diagnose_h76_multiview_bottleneck_20260808 import (  # noqa: E402
    ACTION_NAMES,
    JOINT_NAMES,
    build_four_view_groups,
    load_predictions,
    safe_corr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mmpose-type", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-groups", type=int, default=0)
    return parser.parse_args()


def action_equal_mean(values: np.ndarray, actions: np.ndarray) -> float:
    per_action = [
        float(np.mean(values[actions == action]))
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]
    return float(np.mean(per_action))


def condition_number(rays: np.ndarray) -> np.ndarray:
    """Condition number of the unregularized ray normal matrix, shape N,J."""
    direction = rays[..., :3]
    direction = direction / np.linalg.norm(
        direction, axis=-1, keepdims=True
    ).clip(1e-8)
    confidence = np.clip(rays[..., 6:7], 0.0, 1.0) + 0.05
    eye = np.eye(3, dtype=np.float64)
    projection = eye - direction[..., :, None] * direction[..., None, :]
    normal = (confidence[..., None] * projection).sum(axis=2)
    eigenvalues = np.linalg.eigvalsh(normal)
    return eigenvalues[..., -1] / eigenvalues[..., 0].clip(1e-9)


def added_view_features(
    rays_all: np.ndarray,
    base_combo: tuple[int, ...],
    larger_combo: tuple[int, ...],
    added_view: int,
    base_prediction: np.ndarray,
) -> dict[str, np.ndarray]:
    base_rays = rays_all[:, :, base_combo, :]
    larger_rays = rays_all[:, :, larger_combo, :]
    added_ray = rays_all[:, :, added_view, :]

    # Work on copies: ``added_ray`` is a view into ``rays_all`` and in-place
    # normalization would corrupt later transitions' geometry diagnostics.
    base_direction = base_rays[..., :3].copy()
    base_direction = base_direction / np.linalg.norm(
        base_direction, axis=-1, keepdims=True
    ).clip(1e-8)
    added_direction = added_ray[..., :3].copy()
    added_direction = added_direction / np.linalg.norm(
        added_direction, axis=-1, keepdims=True
    ).clip(1e-8)

    cosine = np.sum(
        base_direction * added_direction[:, :, None, :], axis=-1
    ).clip(-1.0, 1.0)
    parallax = np.degrees(np.arccos(cosine))

    added_point = added_ray[..., 3:6]
    prediction_offset = base_prediction - added_point
    counterfactual_residual = np.linalg.norm(
        np.cross(prediction_offset, added_direction), axis=-1
    ) * 1000.0

    return {
        "added_confidence": np.clip(added_ray[..., 6], 0.0, 1.0),
        "parallax_mean_deg": parallax.mean(axis=2),
        "parallax_min_deg": parallax.min(axis=2),
        "base_prediction_to_added_ray_mm": counterfactual_residual,
        "condition_before": condition_number(base_rays),
        "condition_after": condition_number(larger_rays),
    }


def summarize_transition(
    delta: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    features: dict[str, np.ndarray],
    actions: np.ndarray,
    before_relative: np.ndarray | None = None,
    after_relative: np.ndarray | None = None,
) -> dict:
    oracle_selective = np.minimum(before, after)
    summary = {
        "mean_delta_mm": action_equal_mean(delta, actions),
        "median_delta_mm": float(np.median(delta)),
        "p10_delta_mm": float(np.percentile(delta, 10)),
        "p90_delta_mm": float(np.percentile(delta, 90)),
        "negative_view_rate": action_equal_mean(delta > 0.0, actions),
        "negative_view_rate_gt1mm": action_equal_mean(delta > 1.0, actions),
        "negative_view_rate_gt5mm": action_equal_mean(delta > 5.0, actions),
        "pose_negative_rate": action_equal_mean(
            delta.mean(axis=1) > 0.0, actions
        ),
        "before_mm": action_equal_mean(before, actions),
        "after_mm": action_equal_mean(after, actions),
        "oracle_selective_mm": action_equal_mean(oracle_selective, actions),
        "oracle_gain_over_after_mm": action_equal_mean(
            after - oracle_selective, actions
        ),
        "harm_mean_when_negative_mm": float(np.mean(delta[delta > 0.0])),
        "gain_mean_when_positive_mm": float(np.mean(-delta[delta < 0.0])),
        "feature_correlations_with_delta": {
            name: safe_corr(value, delta) for name, value in features.items()
        },
        "per_joint": {},
        "per_action": {},
    }
    if before_relative is not None and after_relative is not None:
        summary["root_relative"] = {
            "before_mm": action_equal_mean(before_relative, actions),
            "after_mm": action_equal_mean(after_relative, actions),
            "mean_delta_mm": action_equal_mean(
                after_relative - before_relative, actions
            ),
            "negative_view_rate": action_equal_mean(
                after_relative > before_relative, actions
            ),
        }
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        joint_delta = delta[:, joint_index]
        summary["per_joint"][joint_name] = {
            "mean_delta_mm": action_equal_mean(joint_delta, actions),
            "negative_view_rate": action_equal_mean(
                joint_delta > 0.0, actions
            ),
    }
    for action, action_name in ACTION_NAMES.items():
        selected = actions == action
        if not np.any(selected):
            continue
        summary["per_action"][action_name] = {
            "mean_delta_mm": float(np.mean(delta[selected])),
            "negative_view_rate": float(np.mean(delta[selected] > 0.0)),
        }
    return summary


def aggregate_summary(
    indices: list[int],
    arrays: dict[str, np.ndarray],
    actions: np.ndarray,
) -> dict:
    delta = arrays["delta_error_mm"][indices].reshape(-1, len(JOINT_NAMES))
    before = arrays["before_error_mm"][indices].reshape(-1, len(JOINT_NAMES))
    after = arrays["after_error_mm"][indices].reshape(-1, len(JOINT_NAMES))
    before_relative = arrays["before_root_relative_mm"][indices].reshape(
        -1, len(JOINT_NAMES)
    )
    after_relative = arrays["after_root_relative_mm"][indices].reshape(
        -1, len(JOINT_NAMES)
    )
    repeated_actions = np.tile(actions, len(indices))
    features = {
        key: value[indices].reshape(-1, len(JOINT_NAMES))
        for key, value in arrays.items()
        if key not in {
            "delta_error_mm", "before_error_mm", "after_error_mm",
            "before_root_relative_mm", "after_root_relative_mm",
        }
    }
    return summarize_transition(
        delta, before, after, features, repeated_actions,
        before_relative, after_relative,
    )


def plot_results(
    output_dir: Path,
    transition_names: list[str],
    transition_summaries: list[dict],
    arrays: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rates = [summary["negative_view_rate"] * 100 for summary in transition_summaries]
    deltas = [summary["mean_delta_mm"] for summary in transition_summaries]
    colors = ["#d95f02" if delta > 0 else "#1b9e77" for delta in deltas]
    figure, axis = plt.subplots(figsize=(13, 6))
    axis.bar(np.arange(len(rates)), rates, color=colors)
    axis.set_xticks(np.arange(len(rates)), transition_names, rotation=70, ha="right")
    axis.set_ylabel("Negative View Rate (%)")
    axis.set_title("H76 joint-level harm rate when adding one camera")
    axis.axhline(50, color="black", linewidth=0.8, linestyle="--")
    figure.tight_layout()
    figure.savefig(output_dir / "negative_view_rate_by_transition.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for stage, indices, color in (
        ("V2→V3", list(range(12)), "#7570b3"),
        ("V3→V4", list(range(12, 16)), "#e7298a"),
    ):
        values = arrays["delta_error_mm"][indices].reshape(-1)
        axis.hist(
            np.clip(values, -30, 30), bins=100, density=True,
            alpha=0.45, label=stage, color=color,
        )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_xlabel("Δ joint error after adding view (mm; positive = harm)")
    axis.set_ylabel("Density")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "counterfactual_delta_distribution.png", dpi=180)
    plt.close(figure)

    feature_names = (
        "added_confidence", "parallax_mean_deg",
        "base_prediction_to_added_ray_mm", "condition_after",
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    delta = arrays["delta_error_mm"].reshape(-1)
    for axis, feature_name in zip(axes.flat, feature_names):
        feature = arrays[feature_name].reshape(-1)
        finite = np.isfinite(feature) & np.isfinite(delta)
        feature = feature[finite]
        target = delta[finite]
        quantiles = np.quantile(feature, np.linspace(0, 1, 11))
        x_values, y_values = [], []
        for low, high in zip(quantiles[:-1], quantiles[1:]):
            selected = (feature >= low) & (feature <= high)
            if np.any(selected):
                x_values.append(float(np.mean(feature[selected])))
                y_values.append(float(np.mean(target[selected])))
        axis.plot(x_values, y_values, marker="o")
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlabel(feature_name)
        axis.set_ylabel("Mean Δ error (mm)")
    figure.suptitle("Geometry diagnostics vs counterfactual view contribution")
    figure.tight_layout()
    figure.savefig(output_dir / "geometry_binned_counterfactual_curves.png", dpi=180)
    plt.close(figure)


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
    if args.max_groups:
        groups = groups[: args.max_groups]
    actions = np.asarray(
        [int(base_dataset.db[group[0]]["action"]) for group in groups],
        dtype=np.int64,
    )
    rays_all = []
    targets_all = []
    for group_index in range(len(groups)):
        _, _, target, rays, _, _ = base_dataset[group_index]
        rays_all.append(rays.numpy())
        targets_all.append(target.numpy())
    rays_all = np.asarray(rays_all, dtype=np.float64)
    targets_all = np.asarray(targets_all, dtype=np.float64)

    predictions = {}
    combinations = {}
    for views in (2, 3, 4):
        pred, target = load_predictions(Path(args.prediction_root), views)
        combinations[views] = list(itertools.combinations(range(4), views))
        expected = len(groups) * len(combinations[views])
        pred = pred[:expected].reshape(len(groups), len(combinations[views]), 17, 3)
        target = target[:expected].reshape(len(groups), len(combinations[views]), 17, 3)
        expected_target = np.repeat(
            targets_all[:, None], len(combinations[views]), axis=1
        )
        if not np.allclose(target, expected_target, atol=2e-4):
            raise ValueError(f"V{views} target ordering mismatch")
        predictions[views] = pred

    transition_names = []
    metadata = []
    storage: dict[str, list[np.ndarray]] = {
        "before_error_mm": [], "after_error_mm": [], "delta_error_mm": [],
        "before_root_relative_mm": [], "after_root_relative_mm": [],
        "added_confidence": [], "parallax_mean_deg": [],
        "parallax_min_deg": [], "base_prediction_to_added_ray_mm": [],
        "condition_before": [], "condition_after": [],
    }

    for base_views in (2, 3):
        larger_views = base_views + 1
        for base_index, base_combo in enumerate(combinations[base_views]):
            for added_view in sorted(set(range(4)) - set(base_combo)):
                larger_combo = tuple(sorted((*base_combo, added_view)))
                larger_index = combinations[larger_views].index(larger_combo)
                before_prediction = predictions[base_views][:, base_index]
                after_prediction = predictions[larger_views][:, larger_index]
                before_error = np.linalg.norm(
                    before_prediction - targets_all, axis=-1
                ) * 1000.0
                after_error = np.linalg.norm(
                    after_prediction - targets_all, axis=-1
                ) * 1000.0
                before_relative = np.linalg.norm(
                    (before_prediction - before_prediction[:, :1])
                    - (targets_all - targets_all[:, :1]), axis=-1,
                ) * 1000.0
                after_relative = np.linalg.norm(
                    (after_prediction - after_prediction[:, :1])
                    - (targets_all - targets_all[:, :1]), axis=-1,
                ) * 1000.0
                features = added_view_features(
                    rays_all, base_combo, larger_combo, added_view,
                    before_prediction,
                )
                name = (
                    f"V{base_views}:"
                    f"{'-'.join(str(x + 1) for x in base_combo)}+{added_view + 1}"
                    f"->{'-'.join(str(x + 1) for x in larger_combo)}"
                )
                transition_names.append(name)
                metadata.append({
                    "name": name,
                    "base_views": base_views,
                    "base_combo_zero_based": list(base_combo),
                    "larger_combo_zero_based": list(larger_combo),
                    "added_view_zero_based": added_view,
                })
                storage["before_error_mm"].append(before_error)
                storage["after_error_mm"].append(after_error)
                storage["delta_error_mm"].append(after_error - before_error)
                storage["before_root_relative_mm"].append(before_relative)
                storage["after_root_relative_mm"].append(after_relative)
                for key, value in features.items():
                    storage[key].append(value)

    arrays = {key: np.stack(value) for key, value in storage.items()}
    transition_summaries = []
    for transition_index, transition in enumerate(metadata):
        features = {
            key: arrays[key][transition_index]
            for key in (
                "added_confidence", "parallax_mean_deg", "parallax_min_deg",
                "base_prediction_to_added_ray_mm", "condition_before",
                "condition_after",
            )
        }
        transition["summary"] = summarize_transition(
            arrays["delta_error_mm"][transition_index],
            arrays["before_error_mm"][transition_index],
            arrays["after_error_mm"][transition_index],
            features,
            actions,
            arrays["before_root_relative_mm"][transition_index],
            arrays["after_root_relative_mm"][transition_index],
        )
        transition_summaries.append(transition["summary"])

    result = {
        "metric": "H36M S9/S11 frozen-H76 counterfactual added-view audit",
        "groups": len(groups),
        "joint_order": JOINT_NAMES,
        "positive_delta_definition": "added camera increased absolute joint error",
        "transitions": metadata,
        "aggregate": {
            "V2_to_V3": aggregate_summary(list(range(12)), arrays, actions),
            "V3_to_V4": aggregate_summary(list(range(12, 16)), arrays, actions),
            "all": aggregate_summary(list(range(16)), arrays, actions),
        },
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output.with_suffix(".npz"),
        actions=actions,
        **arrays,
    )
    plot_results(
        output.parent / f"{output.stem}_figures",
        transition_names,
        transition_summaries,
        arrays,
    )

    lines = [
        "# H76 counterfactual negative-view audit",
        "",
        f"- Synchronized groups: {len(groups)}",
        "- Positive delta means the added camera made a joint worse.",
        "",
        "| Stage | Mean delta (mm) | Negative View Rate | >1 mm | >5 mm | Pose NVR | Oracle gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("V2_to_V3", "V3_to_V4", "all"):
        summary = result["aggregate"][stage]
        lines.append(
            f"| {stage} | {summary['mean_delta_mm']:.3f} | "
            f"{100 * summary['negative_view_rate']:.2f}% | "
            f"{100 * summary['negative_view_rate_gt1mm']:.2f}% | "
            f"{100 * summary['negative_view_rate_gt5mm']:.2f}% | "
            f"{100 * summary['pose_negative_rate']:.2f}% | "
            f"{summary['oracle_gain_over_after_mm']:.3f} mm |"
        )
    lines.extend([
        "", "## Per transition", "",
        "| Transition | Mean delta (mm) | NVR | Pose NVR |",
        "|---|---:|---:|---:|",
    ])
    for transition in metadata:
        summary = transition["summary"]
        lines.append(
            f"| {transition['name']} | {summary['mean_delta_mm']:.3f} | "
            f"{100 * summary['negative_view_rate']:.2f}% | "
            f"{100 * summary['pose_negative_rate']:.2f}% |"
        )
    output.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate"], indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
