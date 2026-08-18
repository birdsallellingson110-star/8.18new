#!/usr/bin/env python3
"""Train one camera-identity-free H76/Volumetric gate for V2/V3/V4.

All architecture choices are fixed from the per-view leave-one-training-subject
experiments.  The three view counts contribute equally to the objective even
though they contain 6/4/1 camera combinations.  S9/S11 are evaluation-only.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from train_adaptive_joint_branch_gate_20260813 import (
    ACTION_IDS,
    AdaptiveJointGate,
    PARENTS,
    load_pair,
    make_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for split in ("train", "test"):
        for source in ("h76", "vol"):
            for views in (2, 3, 4):
                parser.add_argument(f"--{split}-{source}-v{views}", required=True)
        for views in (2, 3, 4):
            parser.add_argument(f"--{split}-geometry-v{views}")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--gate-depth", type=int, choices=(1, 2), default=1)
    parser.add_argument("--hidden2", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--utility-lambda", type=float, default=0.0)
    parser.add_argument("--utility-temperature-mm", type=float, default=10.0)
    parser.add_argument(
        "--geometry-feature-set",
        choices=(
            "full", "ray_geometry", "confidence_statistics",
            "ray_angles", "ray_distances", "normal_condition",
        ),
        default="full",
    )
    parser.add_argument(
        "--view-count-feature", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--training-subject-cv", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def combinations(views: int) -> tuple[str, ...]:
    return tuple(
        "-".join(str(index + 1) for index in combo)
        for combo in itertools.combinations(range(4), views)
    )


def flatten_view(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    views: int,
    combos: tuple[str, ...],
    add_view_count: bool,
) -> dict[str, np.ndarray]:
    groups, combo_count = data["h76"].shape[:2]
    if combo_count != len(combos):
        raise ValueError("camera-combination count mismatch")
    if add_view_count:
        view_one_hot = np.zeros((*features.shape[:-1], 3), dtype=np.float32)
        view_one_hot[..., views - 2] = 1.0
        features = np.concatenate([features, view_one_hot], axis=-1)
    return {
        "features": features.reshape(groups * combo_count, 17, -1),
        "h76": data["h76"].reshape(groups * combo_count, 17, 3),
        "vol": data["vol"].reshape(groups * combo_count, 17, 3),
        "target": np.repeat(data["target"], combo_count, axis=0),
        "subjects": np.repeat(data["subjects"], combo_count),
        "actions": np.repeat(data["actions"], combo_count),
        "combo_index": np.tile(np.arange(combo_count), groups),
        "view_index": np.full(groups * combo_count, views - 2, dtype=np.int64),
    }


def load_geometry(
    path: str, groups: int, combo_count: int, feature_set: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    payload = np.load(path)
    if bool(np.asarray(payload["uses_ground_truth"]).item()):
        raise ValueError(f"geometry feature file declares GT use: {path}")
    features = np.asarray(payload["features"], dtype=np.float32)
    expected = (groups * combo_count, 17)
    if features.shape[:2] != expected:
        raise ValueError(
            f"geometry alignment failed for {path}: {features.shape[:2]} != {expected}"
        )
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite geometry features in {path}")
    names = tuple(str(value) for value in payload["feature_names"])
    if feature_set == "ray_geometry":
        selected = [index for index, name in enumerate(names) if not name.startswith("confidence_")]
    elif feature_set == "confidence_statistics":
        selected = [index for index, name in enumerate(names) if name.startswith("confidence_")]
    elif feature_set == "ray_angles":
        selected = [index for index, name in enumerate(names) if name.startswith("ray_angle_")]
    elif feature_set == "ray_distances":
        selected = [index for index, name in enumerate(names) if "ray_distance_" in name]
    elif feature_set == "normal_condition":
        selected = [index for index, name in enumerate(names) if name.startswith("normal_eigen_")]
    else:
        selected = list(range(len(names)))
    features = features[..., selected]
    names = tuple(names[index] for index in selected)
    return features.reshape(groups, combo_count, 17, -1), names


def concatenate(parts: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([parts[views][key] for views in (2, 3, 4)], axis=0)
        for key in parts[2]
    }


def static_logits_equal_view(data: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    logits = np.zeros(17, dtype=np.float32)
    for joint in range(17):
        losses = []
        for alpha in grid:
            error = np.linalg.norm(
                alpha * data["h76"][mask, joint]
                + (1.0 - alpha) * data["vol"][mask, joint]
                - data["target"][mask, joint],
                axis=-1,
            )
            view_index = data["view_index"][mask]
            losses.append(np.mean([
                error[view_index == index].mean() for index in range(3)
            ]))
        selected = float(grid[int(np.argmin(losses))])
        selected = np.clip(selected, 0.01, 0.99)
        logits[joint] = np.log(selected / (1.0 - selected))
    return logits


def standardize(
    features: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    flat = features[train_mask].reshape(-1, features.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0)
    std[std < 1e-6] = 1.0
    return (features - mean) / std, np.stack([mean, std])


def fit_and_evaluate(
    data: dict[str, np.ndarray],
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
    utility_lambda: float,
    utility_temperature_mm: float,
    hidden: int,
    gate_depth: int,
    hidden2: int | None,
) -> tuple[dict, dict[str, torch.Tensor], np.ndarray]:
    torch.manual_seed(seed)
    normalized, normalization = standardize(data["features"], train_mask)
    model = AdaptiveJointGate(
        normalized.shape[-1], hidden=hidden, max_delta=2.0,
        logits=static_logits_equal_view(data, train_mask),
        depth=gate_depth,
        hidden2=hidden2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(normalized[train_mask]).to(device)
    h76 = torch.from_numpy(data["h76"][train_mask]).to(device)
    vol = torch.from_numpy(data["vol"][train_mask]).to(device)
    target = torch.from_numpy(data["target"][train_mask]).to(device)
    train_views = torch.from_numpy(data["view_index"][train_mask]).to(device)
    for _ in range(epochs):
        alpha, dynamic = model(x)
        prediction = alpha[..., None] * h76 + (1.0 - alpha[..., None]) * vol
        point_error = torch.linalg.vector_norm(prediction - target, dim=-1)
        data_loss = torch.stack([
            point_error[train_views == index].mean() for index in range(3)
        ]).mean()
        h76_error = torch.linalg.vector_norm(h76 - target, dim=-1)
        vol_error = torch.linalg.vector_norm(vol - target, dim=-1)
        utility_target = torch.sigmoid(
            (vol_error - h76_error) / utility_temperature_mm
        )
        utility_point = torch.nn.functional.binary_cross_entropy(
            alpha.clamp(1e-6, 1.0 - 1e-6), utility_target, reduction="none"
        )
        utility_bce = torch.stack([
            utility_point[train_views == index].mean() for index in range(3)
        ]).mean()
        loss = (
            data_loss
            + utility_lambda * utility_bce
            + 1e-3 * dynamic.square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_x = torch.from_numpy(normalized[eval_mask]).to(device)
        eval_h = torch.from_numpy(data["h76"][eval_mask]).to(device)
        eval_v = torch.from_numpy(data["vol"][eval_mask]).to(device)
        eval_t = torch.from_numpy(data["target"][eval_mask]).to(device)
        alpha, dynamic = model(eval_x)
        prediction = alpha[..., None] * eval_h + (1.0 - alpha[..., None]) * eval_v
        frame_error = torch.linalg.vector_norm(prediction - eval_t, dim=-1).mean(-1)
        frame_error = frame_error.cpu().numpy()
    actions = data["actions"][eval_mask]
    view_index = data["view_index"][eval_mask]
    combo_index = data["combo_index"][eval_mask]
    metrics = {}
    for index, views in enumerate((2, 3, 4)):
        view_mask = view_index == index
        combos = combinations(views)
        metrics[f"V{views}"] = {
            "frame_weighted_mm": float(frame_error[view_mask].mean()),
            "action_equal_mm": float(np.mean([
                frame_error[view_mask & (actions == action)].mean()
                for action in ACTION_IDS if np.any(view_mask & (actions == action))
            ])),
            "per_combination_action_equal_mm": {
                combo: float(np.mean([
                    frame_error[
                        view_mask & (combo_index == combo_id) & (actions == action)
                    ].mean()
                    for action in ACTION_IDS
                    if np.any(
                        view_mask & (combo_index == combo_id) & (actions == action)
                    )
                ]))
                for combo_id, combo in enumerate(combos)
            },
        }
    metrics["dynamic_abs_mean"] = float(dynamic.abs().mean().cpu())
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return metrics, state, normalization


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    loaded: dict[str, dict[int, dict[str, np.ndarray]]] = {"train": {}, "test": {}}
    geometry_arguments = [
        getattr(args, f"{split}_geometry_v{views}")
        for split in ("train", "test") for views in (2, 3, 4)
    ]
    if any(geometry_arguments) and not all(geometry_arguments):
        raise ValueError("provide all six train/test V2/V3/V4 geometry files or none")
    geometry_names: tuple[str, ...] = ()
    for split in ("train", "test"):
        for views in (2, 3, 4):
            combos = combinations(views)
            data = load_pair(
                getattr(args, f"{split}_h76_v{views}"),
                getattr(args, f"{split}_vol_v{views}"), views, combos,
            )
            geometry_path = getattr(args, f"{split}_geometry_v{views}")
            if geometry_path:
                current, names = load_geometry(
                    geometry_path, len(data["h76"]), len(combos),
                    args.geometry_feature_set,
                )
                if geometry_names and names != geometry_names:
                    raise ValueError("geometry feature schemas differ across files")
                geometry_names = names
                data["geometry"] = current
            loaded[split][views] = data
    train_targets = loaded["train"][2]["target"]
    target_bones = train_targets - train_targets[:, PARENTS]
    canonical_bones = np.linalg.norm(target_bones, axis=-1).mean(0)
    flat = {"train": {}, "test": {}}
    for split in ("train", "test"):
        for views in (2, 3, 4):
            data = loaded[split][views]
            features = make_features(
                data, canonical_bones, "no_pair_id", len(combinations(views))
            )
            if "geometry" in data:
                features = np.concatenate((features, data["geometry"]), axis=-1)
            flat[split][views] = flatten_view(
                data, features, views, combinations(views), args.view_count_feature
            )
    train = concatenate(flat["train"])
    test = concatenate(flat["test"])
    cv_folds = []
    if args.training_subject_cv:
        for held_out in sorted(int(x) for x in np.unique(train["subjects"])):
            fit_mask = train["subjects"] != held_out
            eval_mask = ~fit_mask
            fold_metrics, _, _ = fit_and_evaluate(
                train, fit_mask, eval_mask, args.epochs,
                args.seed + held_out, device,
                args.utility_lambda, args.utility_temperature_mm,
                args.hidden,
                args.gate_depth,
                args.hidden2,
            )
            cv_folds.append({
                "held_out_subject": held_out,
                "metrics": fold_metrics,
                "mean_action_equal_across_view_counts_mm": float(np.mean([
                    fold_metrics[f"V{views}"]["action_equal_mm"]
                    for views in (2, 3, 4)
                ])),
            })
    combined = {
        key: np.concatenate([train[key], test[key]], axis=0) for key in train
    }
    train_mask = np.zeros(len(combined["h76"]), dtype=bool)
    train_mask[:len(train["h76"])] = True
    test_mask = ~train_mask
    metrics, state, normalization = fit_and_evaluate(
        combined, train_mask, test_mask, args.epochs, args.seed + 100, device,
        args.utility_lambda, args.utility_temperature_mm,
        args.hidden,
        args.gate_depth,
        args.hidden2,
    )
    result = {
        "protocol": {
            "fit_subjects": sorted(int(x) for x in np.unique(train["subjects"])),
            "test_subjects": sorted(int(x) for x in np.unique(test["subjects"])),
            "single_model_for_views": [2, 3, 4],
            "equal_loss_weight_per_view_count": True,
            "camera_identity_feature": False,
            "view_count_feature": args.view_count_feature,
            "ray_geometry_features": list(geometry_names),
            "architecture_fixed_from_training_subject_cv": {
                "hidden": args.hidden, "depth": args.gate_depth,
                "hidden2": args.hidden2,
                "max_delta": 2.0, "regularization": 1e-3,
            },
            "seed": args.seed,
            "epochs": args.epochs,
            "utility_auxiliary": {
                "lambda": args.utility_lambda,
                "temperature_mm": args.utility_temperature_mm,
                "target": "soft branch preference from training GT only",
            },
        },
        "test": metrics,
    }
    if cv_folds:
        result["training_subject_cv"] = {
            "folds": cv_folds,
            "mean_action_equal_across_subjects_and_view_counts_mm": float(
                np.mean([
                    fold["mean_action_equal_across_view_counts_mm"]
                    for fold in cv_folds
                ])
            ),
            "per_view_mean_action_equal_mm": {
                f"V{views}": float(np.mean([
                    fold["metrics"][f"V{views}"]["action_equal_mm"]
                    for fold in cv_folds
                ]))
                for views in (2, 3, 4)
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    torch.save({
        "model": state,
        "normalization": torch.from_numpy(normalization),
        "canonical_bone_lengths_mm": torch.from_numpy(canonical_bones),
        "protocol": result["protocol"],
    }, args.weights_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
