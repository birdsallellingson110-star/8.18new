#!/usr/bin/env python3
"""Diagnose low-rank HRNet observation bias without changing the input protocol.

The script separates two questions that were previously conflated:

1. Oracle capacity: how much of the 2-D detector error is explained by one
   shared transform per frame/view (translation, scale+translation, similarity,
   or affine)?  Ground truth is used only to fit these diagnostic transforms.
2. Transferability: does a camera-ID-free, per-joint mean residual learned on
   S1/S5/S6/S7 transfer to the completely held-out S8 and then S9/S11?

Every corrected observation is converted back to the exact RUMPL world-ray
convention.  The frozen K96 prediction is used as an anchor and a differentiable
closed-form ray-MAP solve produces a new pose.  The MAP precision and the
shrinkage of the transferable mean correction are selected on S8 only.  S9/S11
is evaluated once with those fixed choices.

Inputs remain frozen HRNet coordinates/confidence, camera calibration and
deterministic derived geometry.  No heatmap, RGB, camera ID embedding or
temporal feature is consumed by the proposed inference controls.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from diagnose_rigr_heatmap_oracle_20260812 import build_four_view_groups
from train_failure_informed_map_20260820 import FrozenK96Anchor
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)
METHODS = ("identity", "translation", "scale_translation", "similarity", "affine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--train-pkl", required=True)
    parser.add_argument("--validation-pkl", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--k96-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-subject", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prior-precisions", nargs="+", type=float,
        default=(
            0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
            300.0, 1000.0, 10000.0,
        ),
    )
    parser.add_argument(
        "--mean-shrinkages", nargs="+", type=float,
        default=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25),
    )
    parser.add_argument(
        "--max-holdout-groups", type=int, default=0,
        help="Debug-only deterministic prefix of S8; zero uses all S8 groups.",
    )
    parser.add_argument(
        "--max-validation-groups", type=int, default=0,
        help="Debug-only deterministic prefix; zero uses all S9/S11 groups.",
    )
    return parser.parse_args()


def load_pickle(path: str) -> list[dict]:
    with Path(path).open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"expected list PKL at {path}, got {type(records)}")
    return records


def group_rotations(records: list[dict], groups: list[list[int]], ids: np.ndarray) -> np.ndarray:
    rotations = np.empty((len(ids), 4, 3, 3), dtype=np.float64)
    for row, group_id in enumerate(ids):
        group = groups[int(group_id)]
        for view, record_index in enumerate(group):
            record = records[record_index]
            if int(record["camera_id"]) != view:
                raise ValueError(f"camera order mismatch at group={group_id}, view={view}")
            rotations[row, view] = np.asarray(
                record["camera"]["R"], dtype=np.float64
            ).reshape(3, 3)
    return rotations


def normalized_observations_and_targets(
    rays: np.ndarray, targets: np.ndarray, rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover normalized camera coordinates from exact cached RUMPL rays."""
    direction = np.asarray(rays[..., :3], dtype=np.float64)
    origin = np.asarray(rays[..., 3:6], dtype=np.float64)
    # RUMPL direction is camera_center - world_point.  For a row vector,
    # -direction @ R.T recovers a scalar multiple of [x_norm,y_norm,1].
    observed_camera = np.einsum("njva,nvba->njvb", -direction, rotations)
    observed = observed_camera[..., :2] / observed_camera[..., 2:3]
    target_offset = targets[:, :, None].astype(np.float64) - origin
    target_camera = np.einsum("njva,nvba->njvb", target_offset, rotations)
    projected = target_camera[..., :2] / target_camera[..., 2:3]
    if not np.isfinite(observed).all() or not np.isfinite(projected).all():
        raise FloatingPointError("non-finite normalized image coordinates")
    return observed, projected


def weighted_mean(value: np.ndarray, weight: np.ndarray, axis: int) -> np.ndarray:
    return np.sum(value * weight, axis=axis) / np.sum(weight, axis=axis).clip(1e-8)


def fit_oracle_transforms(
    observed: np.ndarray, projected: np.ndarray, confidence: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Fit shared transforms independently for every frame and camera view."""
    # Convert N,J,V,C to N,V,J,C for compact batched linear algebra.
    p = observed.transpose(0, 2, 1, 3)
    q = projected.transpose(0, 2, 1, 3)
    w = confidence.transpose(0, 2, 1)[..., None].astype(np.float64).clip(0.01, 1.0)
    p_mean = weighted_mean(p, w, axis=2)[:, :, None]
    q_mean = weighted_mean(q, w, axis=2)[:, :, None]
    pc, qc = p - p_mean, q - q_mean

    corrected: dict[str, np.ndarray] = {"identity": observed.copy()}
    translation = p + (q_mean - p_mean)
    corrected["translation"] = translation.transpose(0, 2, 1, 3)

    numerator = np.sum(w * pc * qc, axis=(2, 3))
    denominator = np.sum(w * pc * pc, axis=(2, 3)).clip(1e-12)
    scale = numerator / denominator
    scaled = scale[:, :, None, None] * pc + q_mean
    corrected["scale_translation"] = scaled.transpose(0, 2, 1, 3)

    cross = np.einsum("nvji,nvjk->nvik", pc * w, qc)
    u, singular, vt = np.linalg.svd(cross)
    rotation = np.einsum("nvij,nvjk->nvik", u, vt)
    determinant = np.linalg.det(rotation)
    reflection = np.tile(np.eye(2), (len(p), p.shape[1], 1, 1))
    reflection[:, :, 1, 1] = np.where(determinant < 0.0, -1.0, 1.0)
    rotation = np.einsum("nvij,nvjk,nvkl->nvil", u, reflection, vt)
    similarity_scale = np.sum(singular * np.stack(
        (np.ones_like(determinant), reflection[:, :, 1, 1]), axis=-1
    ), axis=-1) / denominator
    similar = (
        similarity_scale[:, :, None, None]
        * np.einsum("nvji,nvik->nvjk", pc, rotation)
        + q_mean
    )
    corrected["similarity"] = similar.transpose(0, 2, 1, 3)

    design = np.concatenate((p, np.ones((*p.shape[:-1], 1))), axis=-1)
    normal = np.einsum("nvji,nvjk->nvik", design * w, design)
    rhs = np.einsum("nvji,nvjk->nvik", design * w, q)
    eye = np.eye(3, dtype=np.float64)[None, None]
    matrix = np.linalg.solve(normal + 1e-10 * eye, rhs)
    affine = np.einsum("nvji,nvik->nvjk", design, matrix)
    corrected["affine"] = affine.transpose(0, 2, 1, 3)

    diagnostics: dict[str, dict[str, float]] = {}
    for name, value in corrected.items():
        residual = np.linalg.norm(value - projected, axis=-1)
        diagnostics[name] = {
            "normalized_2d_mean": float(residual.mean()),
            "normalized_2d_median": float(np.median(residual)),
            "normalized_2d_p95": float(np.quantile(residual, 0.95)),
        }
    diagnostics["parameters"] = {
        "translation_norm_mean": float(np.linalg.norm(q_mean - p_mean, axis=-1).mean()),
        "isotropic_scale_mean": float(scale.mean()),
        "isotropic_scale_std": float(scale.std()),
        "similarity_scale_mean": float(similarity_scale.mean()),
        "affine_deviation_from_identity_mean": float(
            np.linalg.norm(matrix[..., :2, :] - np.eye(2), axis=(-2, -1)).mean()
        ),
    }
    return corrected, diagnostics


def correction_to_rays(
    corrected: np.ndarray, rotations: np.ndarray, rays: np.ndarray,
) -> np.ndarray:
    camera = np.concatenate(
        (corrected, np.ones((*corrected.shape[:-1], 1), dtype=np.float64)), axis=-1
    )
    # camera row vector @ R gives the world line direction; RUMPL stores its negative.
    direction = -np.einsum("njva,nvab->njvb", camera, rotations)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True).clip(1e-12)
    output = np.empty_like(rays[..., :7], dtype=np.float64)
    output[..., :3] = direction / norm
    output[..., 3:6] = rays[..., 3:6]
    output[..., 6] = rays[..., 6]
    return output


def map_solve_grid(
    anchor: np.ndarray, rays: np.ndarray, combo: tuple[int, ...],
    precisions: tuple[float, ...],
) -> np.ndarray:
    """Solve every isotropic prior precision from one symmetric eigensystem."""
    selected = rays[:, :, list(combo)]
    direction = selected[..., :3]
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True).clip(1e-12)
    origin = selected[..., 3:6]
    weight = selected[..., 6].clip(0.01, 1.0)
    eye = np.eye(3, dtype=np.float64)
    projector = eye - direction[..., :, None] * direction[..., None, :]
    weighted = weight[..., None, None] * projector
    normal = weighted.sum(axis=2)
    rhs = np.einsum("njvab,njvb->nja", weighted, origin)
    eigenvalue, eigenvector = np.linalg.eigh(normal)
    rhs_local = np.einsum("...ki,...k->...i", eigenvector, rhs)
    anchor_local = np.einsum("...ki,...k->...i", eigenvector, anchor)
    precision = np.asarray(precisions, dtype=np.float64)
    local = (
        rhs_local[None] + precision[:, None, None, None] * anchor_local[None]
    ) / (
        eigenvalue[None] + precision[:, None, None, None]
    ).clip(1e-12)
    return np.einsum("...ki,p...i->p...k", eigenvector, local)


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def pose_metrics(pose: np.ndarray, target: np.ndarray, actions: np.ndarray) -> dict:
    absolute = np.linalg.norm(pose - target, axis=-1) * 1000.0
    relative_pose = pose - pose[:, :1]
    relative_target = target - target[:, :1]
    relative = np.linalg.norm(relative_pose - relative_target, axis=-1) * 1000.0
    return {
        "action_equal_all17_mm": action_equal(absolute, actions),
        "frame_weighted_all17_mm": float(absolute.mean()),
        "action_equal_root_mm": action_equal(absolute[:, :1], actions),
        "action_equal_relative_nonroot_mm": action_equal(relative[:, 1:], actions),
    }


@torch.inference_mode()
def frozen_k96_predictions(
    arrays: dict[str, np.ndarray], indices: np.ndarray, frozen: FrozenK96Anchor,
    device: torch.device, batch_size: int, workers: int, seed: int,
) -> np.ndarray:
    loader = DataLoader(
        ArrayDataset(arrays, indices), batch_size=batch_size, shuffle=False,
        num_workers=workers,
    )
    outputs = [[] for _ in TASKS]
    torch.manual_seed(10000 + seed)
    torch.cuda.manual_seed_all(10000 + seed)
    for predictions, _, rays, _ in loader:
        predictions = predictions.to(device)
        rays = rays.to(device)
        for task_id, combo in enumerate(TASKS):
            outputs[task_id].append(
                frozen(predictions, rays, combo).cpu().numpy().astype(np.float32)
            )
    return np.stack([np.concatenate(chunks) for chunks in outputs], axis=1)


def evaluate_grid(
    anchors: np.ndarray, target: np.ndarray, rays_by_method: dict[str, np.ndarray],
    actions: np.ndarray, precisions: tuple[float, ...],
) -> dict[str, dict[str, dict]]:
    result: dict[str, dict[str, dict]] = {}
    for method, rays in rays_by_method.items():
        stages_by_precision = [
            {f"V{x}": [] for x in (2, 3, 4)} for _ in precisions
        ]
        pairs_by_precision = [dict() for _ in precisions]
        for task_id, combo in enumerate(TASKS):
            pose_grid = map_solve_grid(
                anchors[:, task_id], rays, combo, precisions
            )
            for precision_id, pose in enumerate(pose_grid):
                metric = pose_metrics(pose, target, actions)
                stages_by_precision[precision_id][f"V{len(combo)}"].append(metric)
                if len(combo) == 2:
                    pairs_by_precision[precision_id][
                        f"{combo[0] + 1}-{combo[1] + 1}"
                    ] = metric
            del pose_grid
        rows: dict[str, dict] = {}
        for precision_id, precision in enumerate(precisions):
            key = f"{precision:g}"
            stages = stages_by_precision[precision_id]
            row: dict[str, dict] = {}
            for stage, metrics in stages.items():
                row[stage] = {
                    name: float(np.mean([item[name] for item in metrics]))
                    for name in metrics[0]
                }
            row["V2_pairs"] = pairs_by_precision[precision_id]
            row["headline_v234_mean_mm"] = float(np.mean([
                row[stage]["action_equal_all17_mm"] for stage in ("V2", "V3", "V4")
            ]))
            rows[key] = row
        result[method] = rows
    return result


def anchor_metrics(
    anchors: np.ndarray, target: np.ndarray, actions: np.ndarray,
) -> dict:
    result = {f"V{x}": [] for x in (2, 3, 4)}
    pairs = {}
    for task_id, combo in enumerate(TASKS):
        metric = pose_metrics(anchors[:, task_id], target, actions)
        result[f"V{len(combo)}"].append(metric)
        if len(combo) == 2:
            pairs[f"{combo[0] + 1}-{combo[1] + 1}"] = metric
    output = {}
    for stage, metrics in result.items():
        output[stage] = {
            name: float(np.mean([item[name] for item in metrics]))
            for name in metrics[0]
        }
    output["V2_pairs"] = pairs
    output["headline_v234_mean_mm"] = float(np.mean([
        output[stage]["action_equal_all17_mm"] for stage in ("V2", "V3", "V4")
    ]))
    return output


def select_precision(grid: dict[str, dict[str, dict]]) -> dict[str, float]:
    return {
        method: float(min(rows, key=lambda key: rows[key]["headline_v234_mean_mm"]))
        for method, rows in grid.items()
    }


def geometry_for_split(
    records: list[dict], arrays: dict[str, np.ndarray], indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    groups = build_four_view_groups(records)
    group_ids = arrays["group_indices"][indices].astype(np.int64)
    if group_ids.max(initial=-1) >= len(groups):
        raise ValueError("cache group index exceeds PKL grouping")
    rotations = group_rotations(records, groups, group_ids)
    rays = arrays["rays"][indices].astype(np.float64)
    targets = arrays["targets"][indices].astype(np.float64)
    observed, projected = normalized_observations_and_targets(rays, targets, rotations)
    corrected, diagnostic = fit_oracle_transforms(
        observed, projected, rays[..., 6]
    )
    return rotations, observed, projected, corrected, diagnostic


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    device = torch.device(f"cuda:{args.gpu}")

    train = trainer.load_arrays([args.train_cache], 22)
    validation = trainer.load_arrays([args.validation_cache], 22)
    train_records = load_pickle(args.train_pkl)
    validation_records = load_pickle(args.validation_pkl)
    train_groups = build_four_view_groups(train_records)
    validation_groups = build_four_view_groups(validation_records)
    if len(train_groups) != len(train["targets"]):
        raise ValueError(f"train PKL/cache group mismatch {len(train_groups)} vs {len(train['targets'])}")
    if len(validation_groups) != len(validation["targets"]):
        raise ValueError(
            f"validation PKL/cache group mismatch {len(validation_groups)} vs "
            f"{len(validation['targets'])}"
        )

    holdout_indices = np.flatnonzero(train["subjects"] == args.holdout_subject)
    fit_indices = np.flatnonzero(train["subjects"] != args.holdout_subject)
    validation_indices = np.arange(len(validation["targets"]), dtype=np.int64)
    if args.max_holdout_groups:
        holdout_indices = holdout_indices[:args.max_holdout_groups]
    if args.max_validation_groups:
        validation_indices = validation_indices[:args.max_validation_groups]

    # Fit the camera-ID-free per-joint detector bias only on S1/S5/S6/S7.
    fit_rotations = group_rotations(
        train_records, train_groups, train["group_indices"][fit_indices]
    )
    fit_observed, fit_projected = normalized_observations_and_targets(
        train["rays"][fit_indices], train["targets"][fit_indices], fit_rotations
    )
    fit_weight = train["rays"][fit_indices, ..., 6].astype(np.float64).clip(0.01, 1.0)
    mean_joint_bias = weighted_mean(
        fit_projected - fit_observed, fit_weight[..., None], axis=(0, 2)
    )
    if mean_joint_bias.shape != (17, 2):
        raise ValueError(f"unexpected mean bias shape {mean_joint_bias.shape}")
    del fit_rotations, fit_observed, fit_projected, fit_weight

    hold_rot, hold_obs, hold_gt2d, hold_corrected, hold_2d = geometry_for_split(
        train_records, train, holdout_indices
    )
    val_rot, val_obs, val_gt2d, val_corrected, val_2d = geometry_for_split(
        validation_records, validation, validation_indices
    )

    holdout_rays = train["rays"][holdout_indices].astype(np.float64)
    validation_rays = validation["rays"][validation_indices].astype(np.float64)
    hold_rays_by_method = {
        name: correction_to_rays(value, hold_rot, holdout_rays)
        for name, value in hold_corrected.items()
    }
    val_rays_by_method = {
        name: correction_to_rays(value, val_rot, validation_rays)
        for name, value in val_corrected.items()
    }

    # Transferable mean-bias shrinkage is a separate method family.  Select
    # both shrinkage and MAP precision on S8; freeze both before S9/S11.
    for shrinkage in args.mean_shrinkages:
        name = f"train_mean_joint_bias_s{shrinkage:g}"
        hold_value = hold_obs + float(shrinkage) * mean_joint_bias[None, :, None]
        val_value = val_obs + float(shrinkage) * mean_joint_bias[None, :, None]
        hold_rays_by_method[name] = correction_to_rays(
            hold_value, hold_rot, holdout_rays
        )
        val_rays_by_method[name] = correction_to_rays(
            val_value, val_rot, validation_rays
        )

    frozen_args = SimpleNamespace(
        train_cache=args.train_cache,
        e2_checkpoint=args.e2_checkpoint,
        proposal_checkpoint=args.proposal_checkpoint,
        k96_checkpoint=args.k96_checkpoint,
    )
    frozen = FrozenK96Anchor(frozen_args, device)
    print(json.dumps({"status": "exporting frozen K96 S8 anchors", "groups": len(holdout_indices)}), flush=True)
    hold_anchors = frozen_k96_predictions(
        train, holdout_indices, frozen, device, args.batch_size, args.workers, args.seed
    )
    print(json.dumps({"status": "exporting frozen K96 S9/S11 anchors", "groups": len(validation_indices)}), flush=True)
    validation_anchors = frozen_k96_predictions(
        validation, validation_indices, frozen, device,
        args.batch_size, args.workers, args.seed,
    )

    hold_actions = train["actions"][holdout_indices]
    validation_actions = validation["actions"][validation_indices]
    hold_target = train["targets"][holdout_indices].astype(np.float64)
    validation_target = validation["targets"][validation_indices].astype(np.float64)
    precisions = tuple(float(value) for value in args.prior_precisions)
    hold_grid = evaluate_grid(
        hold_anchors, hold_target, hold_rays_by_method, hold_actions, precisions
    )
    selected = select_precision(hold_grid)
    validation_selected = {}
    for method, precision in selected.items():
        key = f"{precision:g}"
        validation_grid = evaluate_grid(
            validation_anchors, validation_target,
            {method: val_rays_by_method[method]}, validation_actions, (precision,),
        )
        validation_selected[method] = {
            "selected_prior_precision_on_s8": precision,
            **validation_grid[method][key],
        }

    anchor_holdout = anchor_metrics(hold_anchors, hold_target, hold_actions)
    anchor_validation = anchor_metrics(
        validation_anchors, validation_target, validation_actions
    )
    baseline = anchor_validation["headline_v234_mean_mm"]
    gains = {
        method: baseline - row["headline_v234_mean_mm"]
        for method, row in validation_selected.items()
    }
    identity_headline = validation_selected["identity"]["headline_v234_mean_mm"]
    incremental_vs_identity = {
        method: identity_headline - row["headline_v234_mean_mm"]
        for method, row in validation_selected.items()
    }
    best_oracle = max(
        METHODS, key=lambda method: gains[method]
    )
    mean_candidates = [
        name for name in validation_selected
        if name.startswith("train_mean_joint_bias") and not name.endswith("_s0")
    ]
    best_mean = max(mean_candidates, key=lambda method: gains[method])
    payload = {
        "method": "low-rank 2D observation-bias oracle + frozen K96 ray-MAP",
        "input_protocol": (
            "frozen HRNet coordinates/confidence + cameras + deterministic geometry only"
        ),
        "scientific_boundary": {
            "oracle_methods": (
                "translation/scale/similarity/affine fit S8 or S9/S11 GT projections; "
                "diagnostic upper bounds only, never deployable results"
            ),
            "transferable_control": (
                "per-joint mean normalized-camera residual fit on S1/S5/S6/S7; "
                "shrinkage and MAP precision selected on complete S8"
            ),
            "test_selection": "none; S9/S11 evaluated once after S8 selection",
        },
        "counts": {
            "fit_s1_s5_s6_s7": int(len(fit_indices)),
            "holdout_s8": int(len(holdout_indices)),
            "validation_s9_s11": int(len(validation_indices)),
        },
        "mean_joint_bias_normalized_camera": mean_joint_bias.tolist(),
        "s8": {
            "k96_anchor": anchor_holdout,
            "two_dimensional_fit": hold_2d,
            "precision_grid": hold_grid,
            "selected_precision": selected,
        },
        "s9_s11": {
            "k96_anchor": anchor_validation,
            "two_dimensional_fit": val_2d,
            "selected_from_s8": validation_selected,
            "headline_gain_vs_k96_mm": gains,
            "incremental_gain_vs_identity_map_mm": incremental_vs_identity,
        },
        "decision": {
            "best_oracle_method": best_oracle,
            "best_oracle_headline_gain_mm": gains[best_oracle],
            "best_oracle_incremental_gain_vs_identity_map_mm": (
                incremental_vs_identity[best_oracle]
            ),
            "oracle_passes_1p5mm_gate": bool(
                gains[best_oracle] >= 1.5
                and incremental_vs_identity[best_oracle] >= 1.0
            ),
            "best_transferable_mean_method": best_mean,
            "best_transferable_mean_headline_gain_mm": gains[best_mean],
            "best_transferable_mean_incremental_gain_vs_identity_map_mm": (
                incremental_vs_identity[best_mean]
            ),
            "transferable_mean_passes_0p3mm_gate": bool(
                incremental_vs_identity[best_mean] >= 0.3
            ),
        },
        "args": vars(args),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "result.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(json.dumps({"result": str(output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
