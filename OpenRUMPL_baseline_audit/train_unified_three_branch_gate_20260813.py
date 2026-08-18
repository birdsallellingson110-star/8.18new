#!/usr/bin/env python3
"""Train a unified H76/Volumetric/Algebraic joint-quality gate.

Architecture and features mirror the selected two-branch gate.  The only
methodological change is a third official Algebraic LT candidate and a
three-way softmax, enabling a controlled candidate-complementarity test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_adaptive_joint_branch_gate_20260813 import ACTION_IDS, PARENTS, load_pair, make_features
from train_unified_multiview_joint_gate_20260813 import (
    combinations, concatenate, load_geometry, standardize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for split in ("train", "test"):
        for source in ("h76", "vol"):
            for views in (2, 3, 4):
                parser.add_argument(f"--{split}-{source}-v{views}", required=True)
        for views in (2, 3, 4):
            parser.add_argument(f"--{split}-geometry-v{views}", required=True)
            parser.add_argument(f"--{split}-candidate-residual-v{views}")
        parser.add_argument(f"--{split}-alg", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--expected-risk-weight", type=float, default=0.0)
    parser.add_argument("--fused-mpjpe-weight", type=float, default=1.0)
    parser.add_argument("--soft-target-weight", type=float, default=0.0)
    parser.add_argument("--soft-target-temperature-mm", type=float, default=5.0)
    parser.add_argument(
        "--ray-geometry", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--candidate-residual-feature-set",
        choices=("full", "distance", "angle"), default="full",
    )
    return parser.parse_args()


def add_algebraic(data: dict[str, np.ndarray], path: str, views: int) -> None:
    payload = np.load(path)
    combos = combinations(views)
    algebraic = np.stack([
        np.asarray(payload[f"prediction_V{views}_{combo.replace('-', '_')}"])
        for combo in combos
    ], axis=1).astype(np.float32)
    target = np.asarray(payload["targets"], dtype=np.float32)
    actions = np.asarray(payload["actions"], dtype=np.int64)
    if algebraic.shape != data["h76"].shape:
        raise ValueError(f"Algebraic shape mismatch: {algebraic.shape} != {data['h76'].shape}")
    if np.max(np.abs(target - data["target"])) > 1e-3 or not np.array_equal(actions, data["actions"]):
        raise ValueError("Algebraic target/action alignment failed")
    data["alg"] = algebraic


def flatten(data: dict[str, np.ndarray], features: np.ndarray, views: int) -> dict[str, np.ndarray]:
    groups, combo_count = data["h76"].shape[:2]
    return {
        "features": features.reshape(groups * combo_count, 17, -1),
        "h76": data["h76"].reshape(groups * combo_count, 17, 3),
        "vol": data["vol"].reshape(groups * combo_count, 17, 3),
        "alg": data["alg"].reshape(groups * combo_count, 17, 3),
        "target": np.repeat(data["target"], combo_count, axis=0),
        "subjects": np.repeat(data["subjects"], combo_count),
        "actions": np.repeat(data["actions"], combo_count),
        "combo_index": np.tile(np.arange(combo_count), groups),
        "view_index": np.full(groups * combo_count, views - 2, dtype=np.int64),
    }


def three_branch_features(
    data: dict[str, np.ndarray], canonical_bones: np.ndarray, geometry: np.ndarray,
) -> np.ndarray:
    base = make_features(data, canonical_bones, "no_pair_id", data["h76"].shape[1])
    alg = data["alg"]
    h76, vol = data["h76"], data["vol"]
    alg_h = (alg - h76) / 1000.0
    alg_v = (alg - vol) / 1000.0
    alg_relative = (alg - alg[:, :, :1]) / 1000.0
    alg_bones = alg - alg[:, :, PARENTS]
    alg_bone_error = (
        np.linalg.norm(alg_bones, axis=-1) - canonical_bones[None, None]
    )[..., None] / 1000.0
    extra = (
        alg_h, np.linalg.norm(alg_h, axis=-1, keepdims=True),
        alg_v, np.linalg.norm(alg_v, axis=-1, keepdims=True),
        alg_relative, alg_bone_error, geometry,
    )
    return np.concatenate((base, *extra), axis=-1).astype(np.float32)


def static_logits(
    data: dict[str, np.ndarray], mask: np.ndarray, device: torch.device,
) -> np.ndarray:
    candidates = torch.from_numpy(np.stack([
        data["h76"][mask], data["vol"][mask], data["alg"][mask]
    ], axis=-2)).to(device)
    target = torch.from_numpy(data["target"][mask]).to(device)
    view_index = torch.from_numpy(data["view_index"][mask]).to(device)
    logits = nn.Parameter(torch.zeros(17, 3, device=device))
    optimizer = torch.optim.Adam([logits], lr=0.05)
    for _ in range(300):
        weights = logits.softmax(-1)[None, :, :, None]
        prediction = (weights * candidates).sum(-2)
        error = torch.linalg.vector_norm(prediction - target, dim=-1)
        loss = torch.stack([error[view_index == index].mean() for index in range(3)]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return logits.detach().cpu().numpy().astype(np.float32)


class ThreeBranchGate(nn.Module):
    def __init__(self, feature_dim: int, hidden: int, hidden2: int, logits: np.ndarray):
        super().__init__()
        self.register_buffer("static_logits", torch.from_numpy(logits))
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden2), nn.ReLU(inplace=True),
            nn.Linear(hidden2, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic = 2.0 * torch.tanh(self.network(features))
        weights = torch.softmax(self.static_logits[None] + dynamic, dim=-1)
        return weights, dynamic


def fit(
    data: dict[str, np.ndarray], train_mask: np.ndarray, eval_mask: np.ndarray,
    args: argparse.Namespace, seed: int,
) -> tuple[dict, dict[str, torch.Tensor], np.ndarray]:
    device = torch.device(args.device)
    torch.manual_seed(seed)
    normalized, normalization = standardize(data["features"], train_mask)
    model = ThreeBranchGate(
        normalized.shape[-1], args.hidden, args.hidden2,
        static_logits(data, train_mask, device),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.from_numpy(normalized[train_mask]).to(device)
    candidates = torch.from_numpy(np.stack([
        data["h76"][train_mask], data["vol"][train_mask], data["alg"][train_mask]
    ], axis=-2)).to(device)
    target = torch.from_numpy(data["target"][train_mask]).to(device)
    view_index = torch.from_numpy(data["view_index"][train_mask]).to(device)
    for _ in range(args.epochs):
        weights, dynamic = model(x)
        prediction = (weights[..., None] * candidates).sum(-2)
        error = torch.linalg.vector_norm(prediction - target, dim=-1)
        data_loss = torch.stack([
            error[view_index == index].mean() for index in range(3)
        ]).mean()
        candidate_error = torch.linalg.vector_norm(
            candidates - target[:, :, None], dim=-1
        )
        expected_risk_point = (weights * candidate_error).sum(-1)
        expected_risk = torch.stack([
            expected_risk_point[view_index == index].mean()
            for index in range(3)
        ]).mean()
        soft_target = torch.softmax(
            -candidate_error / args.soft_target_temperature_mm, dim=-1
        )
        soft_target_point = -(
            soft_target * weights.clamp_min(1e-8).log()
        ).sum(-1)
        soft_target_loss = torch.stack([
            soft_target_point[view_index == index].mean()
            for index in range(3)
        ]).mean()
        loss = (
            args.fused_mpjpe_weight * data_loss
            + args.expected_risk_weight * expected_risk
            + args.soft_target_weight * soft_target_loss
        )
        loss = loss + 1e-3 * dynamic.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        weights, dynamic = model(torch.from_numpy(normalized[eval_mask]).to(device))
        candidates = torch.from_numpy(np.stack([
            data["h76"][eval_mask], data["vol"][eval_mask], data["alg"][eval_mask]
        ], axis=-2)).to(device)
        prediction = (weights[..., None] * candidates).sum(-2)
        target = torch.from_numpy(data["target"][eval_mask]).to(device)
        joint_error = torch.linalg.vector_norm(prediction - target, dim=-1).cpu().numpy()
        frame_error = joint_error.mean(-1)
        weights_numpy = weights.cpu().numpy()
    actions, views, combo_index = (
        data["actions"][eval_mask], data["view_index"][eval_mask],
        data["combo_index"][eval_mask],
    )
    metrics = {}
    for index, count in enumerate((2, 3, 4)):
        view_mask = views == index
        metrics[f"V{count}"] = {
            "action_equal_mm": float(np.mean([
                frame_error[view_mask & (actions == action)].mean()
                for action in ACTION_IDS if np.any(view_mask & (actions == action))
            ])),
            "per_combination_action_equal_mm": {
                combo: float(np.mean([
                    frame_error[view_mask & (combo_index == combo_id) & (actions == action)].mean()
                    for action in ACTION_IDS
                    if np.any(view_mask & (combo_index == combo_id) & (actions == action))
                ]))
                for combo_id, combo in enumerate(combinations(count))
            },
            "per_action_mm": {
                str(action): float(frame_error[view_mask & (actions == action)].mean())
                for action in ACTION_IDS if np.any(view_mask & (actions == action))
            },
            "per_joint_action_equal_mm": [
                float(np.mean([
                    joint_error[view_mask & (actions == action), joint].mean()
                    for action in ACTION_IDS
                    if np.any(view_mask & (actions == action))
                ]))
                for joint in range(17)
            ],
            "mean_branch_weights": weights_numpy[view_mask].mean(axis=(0, 1)).tolist(),
        }
    metrics["dynamic_abs_mean"] = float(dynamic.abs().mean().cpu())
    return metrics, {k: v.detach().cpu() for k, v in model.state_dict().items()}, normalization


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    loaded = {"train": {}, "test": {}}
    geometry_names = None
    for split in ("train", "test"):
        for views in (2, 3, 4):
            data = load_pair(
                getattr(args, f"{split}_h76_v{views}"),
                getattr(args, f"{split}_vol_v{views}"), views, combinations(views),
            )
            add_algebraic(data, getattr(args, f"{split}_alg"), views)
            geometry, names = load_geometry(
                getattr(args, f"{split}_geometry_v{views}"), len(data["h76"]),
                len(combinations(views)), "ray_geometry",
            )
            if geometry_names is not None and names != geometry_names:
                raise ValueError("geometry schemas differ")
            geometry_names = names
            data["geometry"] = geometry
            residual_path = getattr(args, f"{split}_candidate_residual_v{views}")
            if residual_path:
                residual_payload = np.load(residual_path)
                if bool(np.asarray(residual_payload["uses_ground_truth"]).item()):
                    raise ValueError("candidate residual file declares GT use")
                residual = np.asarray(residual_payload["features"], dtype=np.float32)
                residual_names = tuple(
                    str(value) for value in residual_payload["feature_names"]
                )
                if args.candidate_residual_feature_set == "distance":
                    selected = [i for i, name in enumerate(residual_names) if "distance" in name]
                    residual = residual[..., selected]
                elif args.candidate_residual_feature_set == "angle":
                    selected = [i for i, name in enumerate(residual_names) if "angle" in name]
                    residual = residual[..., selected]
                expected = (len(data["h76"]) * len(combinations(views)), 17)
                if residual.shape[:2] != expected:
                    raise ValueError("candidate residual alignment failed")
                data["candidate_residual"] = residual.reshape(
                    len(data["h76"]), len(combinations(views)), 17, -1
                )
            loaded[split][views] = data

    train_targets = loaded["train"][2]["target"]
    canonical_bones = np.linalg.norm(
        train_targets - train_targets[:, PARENTS], axis=-1
    ).mean(0)
    flat = {"train": {}, "test": {}}
    for split in ("train", "test"):
        for views in (2, 3, 4):
            data = loaded[split][views]
            geometry = (
                data["geometry"] if args.ray_geometry
                else np.empty((*data["geometry"].shape[:-1], 0), dtype=np.float32)
            )
            if "candidate_residual" in data:
                geometry = np.concatenate((geometry, data["candidate_residual"]), axis=-1)
            features = three_branch_features(data, canonical_bones, geometry)
            flat[split][views] = flatten(data, features, views)
    train, test = concatenate(flat["train"]), concatenate(flat["test"])

    folds = []
    for held_out in sorted(int(x) for x in np.unique(train["subjects"])):
        mask = train["subjects"] != held_out
        metrics, _, _ = fit(train, mask, ~mask, args, args.seed + held_out)
        folds.append({
            "held_out_subject": held_out, "metrics": metrics,
            "mean_action_equal_across_view_counts_mm": float(np.mean([
                metrics[f"V{views}"]["action_equal_mm"] for views in (2, 3, 4)
            ])),
        })

    combined = {key: np.concatenate((train[key], test[key]), axis=0) for key in train}
    train_mask = np.arange(len(combined["h76"])) < len(train["h76"])
    metrics, state, normalization = fit(
        combined, train_mask, ~train_mask, args, args.seed + 100
    )
    result = {
        "protocol": {
            "branches": ["H76", "official LT Volumetric", "official LT Algebraic"],
            "single_model_for_views": [2, 3, 4],
            "equal_loss_weight_per_view_count": True,
            "camera_identity_feature": False,
            "view_count_feature": False,
            "ray_geometry_features": (
                list(geometry_names or ()) if args.ray_geometry else []
            ),
            "quality_head": [args.hidden, args.hidden2, 3],
            "seed": args.seed,
            "training_objective": {
                "fused_mpjpe_weight": args.fused_mpjpe_weight,
                "expected_risk_weight": args.expected_risk_weight,
                "soft_target_weight": args.soft_target_weight,
                "soft_target_temperature_mm": args.soft_target_temperature_mm,
            },
        },
        "test": metrics,
        "training_subject_cv": {
            "folds": folds,
            "mean_action_equal_across_subjects_and_view_counts_mm": float(np.mean([
                fold["mean_action_equal_across_view_counts_mm"] for fold in folds
            ])),
            "per_view_mean_action_equal_mm": {
                f"V{views}": float(np.mean([
                    fold["metrics"][f"V{views}"]["action_equal_mm"] for fold in folds
                ])) for views in (2, 3, 4)
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save({
        "model": state, "normalization": torch.from_numpy(normalization),
        "canonical_bone_lengths_mm": torch.from_numpy(canonical_bones),
        "protocol": result["protocol"],
    }, args.weights_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
