#!/usr/bin/env python3
"""Gate a frozen generative pose prior before implementing a normalizing flow.

The diagnostic never predicts a coordinate correction.  It fits a regularized
Gaussian density to root-relative H36M training poses, uses that density only
to re-rank the frozen K96 hypotheses, and selects the mixture coefficient on a
completely held-out training subject.  S9/S11 are evaluated once with the
subject-held-out choice.

This is a deliberately low-capacity premise check for the normalizing-flow pose
prior used by probabilistic 3D pose work.  If even the invariant density signal
does not transfer across subjects, a higher-capacity flow is not launched.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from train_failure_informed_map_20260820 import FrozenK96Anchor
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)
PARENTS = extra.PARENTS.numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--k96-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--holdout-subject", type=int, default=8)
    parser.add_argument(
        "--betas", type=float, nargs="+",
        default=(0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8),
    )
    parser.add_argument(
        "--features", choices=("bone_direction", "root_relative"),
        nargs="+", default=("bone_direction", "root_relative"),
    )
    parser.add_argument("--shrinkage", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def pose_feature_numpy(pose: np.ndarray, mode: str) -> np.ndarray:
    if mode == "bone_direction":
        bone = pose[:, 1:] - pose[:, PARENTS[1:]]
        length = np.linalg.norm(bone, axis=-1, keepdims=True)
        return (bone / np.maximum(length, 1e-6)).reshape(len(pose), -1)
    relative = pose[:, 1:] - pose[:, :1]
    scale = np.linalg.norm(
        pose[:, 1:] - pose[:, PARENTS[1:]], axis=-1
    ).mean(axis=1, keepdims=True)
    return (relative / np.maximum(scale[:, :, None], 1e-6)).reshape(len(pose), -1)


def pose_feature_torch(pose: torch.Tensor, mode: str) -> torch.Tensor:
    # pose: B x K x J x 3
    parents = torch.as_tensor(PARENTS, device=pose.device)
    if mode == "bone_direction":
        bone = pose[:, :, 1:] - pose[:, :, parents[1:]]
        return F.normalize(bone, dim=-1).flatten(2)
    relative = pose[:, :, 1:] - pose[:, :, :1]
    bone = pose[:, :, 1:] - pose[:, :, parents[1:]]
    scale = torch.linalg.vector_norm(bone, dim=-1).mean(dim=-1, keepdim=True)
    return (relative / scale.clamp_min(1e-6)[..., None]).flatten(2)


def fit_density(feature: np.ndarray, shrinkage: float) -> dict[str, torch.Tensor]:
    feature64 = feature.astype(np.float64)
    mean = feature64.mean(axis=0)
    centred = feature64 - mean
    covariance = centred.T @ centred / max(len(feature64) - 1, 1)
    average_variance = float(np.trace(covariance) / covariance.shape[0])
    covariance = (
        (1.0 - shrinkage) * covariance
        + shrinkage * average_variance * np.eye(covariance.shape[0])
    )
    precision = np.linalg.inv(covariance)
    return {
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "precision": torch.from_numpy(precision.astype(np.float32)),
        "average_variance": torch.tensor(average_variance),
    }


def density_energy(
    hypotheses: torch.Tensor, mode: str, density: dict[str, torch.Tensor]
) -> torch.Tensor:
    feature = pose_feature_torch(hypotheses, mode)
    mean = density["mean"].to(feature.device)
    precision = density["precision"].to(feature.device)
    centred = feature - mean
    energy = torch.einsum("bkd,de,bke->bk", centred, precision, centred)
    # Only relative ranking within the GT-free hypothesis set is used.  This
    # also prevents density scale from changing with feature dimensionality.
    return (energy - energy.mean(dim=-1, keepdim=True)) / energy.std(
        dim=-1, keepdim=True
    ).clamp_min(1e-6)


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


@torch.inference_mode()
def evaluate_grid(
    anchor_model: FrozenK96Anchor,
    loader: DataLoader,
    device: torch.device,
    densities: dict[str, dict[str, torch.Tensor]],
    features: list[str],
    betas: list[float],
    seed: int,
) -> dict:
    keys = [(feature, beta) for feature in features for beta in betas]
    stores: dict[tuple[str, float], dict[str, list[np.ndarray]]] = {
        key: {f"V{count}": [] for count in (2, 3, 4)} for key in keys
    }
    actions_by_stage = {f"V{count}": [] for count in (2, 3, 4)}
    torch.manual_seed(10000 + seed)
    torch.cuda.manual_seed_all(10000 + seed)
    for batch_index, (predictions, targets, rays, actions) in enumerate(loader):
        predictions = predictions.to(device)
        targets = targets.to(device)
        rays = rays.to(device)
        for combo in TASKS:
            stage = f"V{len(combo)}"
            hypotheses, scores = anchor_model.hypotheses_and_scores(
                predictions, rays, combo
            )
            score_logits = scores / anchor_model.score_temperature
            energies = {
                mode: density_energy(hypotheses, mode, densities[mode])
                for mode in features
            }
            for mode, beta in keys:
                weights = F.softmax(score_logits - beta * energies[mode], dim=-1)
                fused = torch.einsum("bk,bkjd->bjd", weights, hypotheses)
                error = torch.linalg.vector_norm(
                    fused - targets, dim=-1
                ).cpu().numpy() * 1000.0
                stores[(mode, beta)][stage].append(error)
            actions_by_stage[stage].append(actions.numpy().copy())
    result = {}
    for mode, beta in keys:
        name = f"{mode}:beta={beta:g}"
        result[name] = {}
        for stage in ("V2", "V3", "V4"):
            values = np.concatenate(stores[(mode, beta)][stage])
            actions = np.concatenate(actions_by_stage[stage])
            result[name][stage] = {
                "action_equal_all17_mm": action_equal(values, actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
        result[name]["headline_mm"] = float(np.mean([
            result[name][stage]["action_equal_all17_mm"]
            for stage in ("V2", "V3", "V4")
        ]))
    return result


def main() -> None:
    args = parse_args()
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    device = torch.device(f"cuda:{args.gpu}")
    train = trainer.load_arrays([args.train_cache], 22)
    validation = trainer.load_arrays([args.validation_cache], 22)
    fit_mask = train["subjects"] != args.holdout_subject
    holdout_mask = train["subjects"] == args.holdout_subject
    if not np.any(holdout_mask):
        raise ValueError("holdout subject is absent")
    densities = {}
    density_manifest = {}
    for mode in args.features:
        feature = pose_feature_numpy(train["targets"][fit_mask], mode)
        densities[mode] = fit_density(feature, args.shrinkage)
        density_manifest[mode] = {
            "dimension": int(feature.shape[1]),
            "fit_samples": int(feature.shape[0]),
            "average_variance": float(densities[mode]["average_variance"]),
        }
    if args.smoke_batches:
        maximum = args.batch_size * args.smoke_batches
        holdout_indices = np.flatnonzero(holdout_mask)[:maximum]
        validation = {key: value[:maximum] for key, value in validation.items()}
    else:
        holdout_indices = np.flatnonzero(holdout_mask)
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    validation_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )
    anchor_model = FrozenK96Anchor(args, device)
    holdout = evaluate_grid(
        anchor_model, holdout_loader, device, densities,
        list(args.features), list(args.betas), args.seed,
    )
    best_name = min(holdout, key=lambda key: holdout[key]["headline_mm"])
    baseline_names = [key for key in holdout if key.endswith("beta=0")]
    baseline = holdout[baseline_names[0]]["headline_mm"]
    gain = baseline - holdout[best_name]["headline_mm"]

    # S9/S11 are evaluated once with the S8-selected feature and beta.  beta=0
    # is included as the exact frozen K96 control under the same random stream.
    best_feature, beta_text = best_name.split(":beta=")
    best_beta = float(beta_text)
    validation_grid = evaluate_grid(
        anchor_model, validation_loader, device, densities,
        [best_feature], sorted(set((0.0, best_beta))), args.seed,
    )
    payload = {
        "method": "frozen Gaussian pose density re-ranking of K96 hypotheses",
        "paper_basis": (
            "Probabilistic Monocular 3D Human Pose Estimation with Normalizing "
            "Flows (ICCV 2021), low-capacity premise gate"
        ),
        "official_code_commit": "ad2fdf2",
        "holdout_subject": args.holdout_subject,
        "density": density_manifest,
        "holdout_grid": holdout,
        "selected": {
            "name": best_name,
            "holdout_headline_mm": holdout[best_name]["headline_mm"],
            "baseline_headline_mm": baseline,
            "gain_mm": gain,
            "passes_0p15mm_gate": bool(gain >= 0.15),
        },
        "validation_selected_once": validation_grid,
        "protocol": (
            "density fit S1/S5/S6/S7; beta selected on S8; S9/S11 evaluated once"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["selected"], indent=2), flush=True)
    print(json.dumps(payload["validation_selected_once"], indent=2), flush=True)


if __name__ == "__main__":
    main()
