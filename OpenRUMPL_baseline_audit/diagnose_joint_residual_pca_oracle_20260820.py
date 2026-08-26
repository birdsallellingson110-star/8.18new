#!/usr/bin/env python3
"""Oracle audit for camera-ID-free low-rank joint detector residuals.

The PCA basis is fit only on H36M S1/S5/S6/S7.  For K>0, coefficients are
projected from the ground-truth 2-D residual of each held-out frame/view and
are therefore diagnostic oracle variables, never deployable inputs.  S8 is
used to choose the ray-MAP prior precision; S9/S11 is evaluated once with that
choice.  This establishes whether a learned low-dimensional coefficient head
could possibly produce a material gain before training one.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import diagnose_lowrank_observation_bias_20260820 as low
import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from diagnose_rigr_heatmap_oracle_20260812 import build_four_view_groups
from train_failure_informed_map_20260820 import FrozenK96Anchor


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
        "--components", nargs="+", type=int,
        default=(0, 1, 2, 4, 8, 16, 32),
    )
    parser.add_argument(
        "--prior-precisions", nargs="+", type=float,
        default=(0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0,
                 100.0, 300.0, 1000.0, 10000.0),
    )
    parser.add_argument("--max-holdout-groups", type=int, default=0)
    parser.add_argument("--max-validation-groups", type=int, default=0)
    return parser.parse_args()


def fit_pca(residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a shared 34-D joint residual basis over every frame and view."""
    samples = residual.transpose(0, 2, 1, 3).reshape(-1, 34).astype(np.float64)
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = np.maximum(eigenvalue[order], 0.0)
    basis = eigenvector[:, order]
    return mean, basis, eigenvalue


def pca_oracle_correction(
    observed: np.ndarray,
    projected: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
    components: int,
) -> tuple[np.ndarray, dict[str, float]]:
    residual = (
        projected - observed
    ).transpose(0, 2, 1, 3).reshape(-1, 34).astype(np.float64)
    if components == 0:
        reconstruction = np.broadcast_to(mean, residual.shape)
    else:
        selected = basis[:, :components]
        reconstruction = mean + (residual - mean) @ selected @ selected.T
    corrected = observed + reconstruction.reshape(
        len(observed), observed.shape[2], observed.shape[1], 2
    ).transpose(0, 2, 1, 3)
    raw_joint = np.linalg.norm(residual.reshape(-1, 17, 2), axis=-1)
    remaining = residual - reconstruction
    remaining_joint = np.linalg.norm(remaining.reshape(-1, 17, 2), axis=-1)
    return corrected, {
        "raw_normalized_2d_mean": float(raw_joint.mean()),
        "remaining_normalized_2d_mean": float(remaining_joint.mean()),
        "remaining_normalized_2d_median": float(np.median(remaining_joint)),
        "remaining_normalized_2d_p95": float(np.quantile(remaining_joint, 0.95)),
        "fraction_squared_residual_removed": float(
            1.0 - np.sum(remaining * remaining) / np.sum(residual * residual).clip(1e-20)
        ),
    }


def split_geometry(records, groups, arrays, indices):
    rotations = low.group_rotations(
        records, groups, arrays["group_indices"][indices].astype(np.int64)
    )
    rays = arrays["rays"][indices].astype(np.float64)
    targets = arrays["targets"][indices].astype(np.float64)
    observed, projected = low.normalized_observations_and_targets(
        rays, targets, rotations
    )
    return rotations, rays, observed, projected


def selected_rows(grid, selected):
    return {
        method: {
            "selected_prior_precision_on_s8": precision,
            **grid[method][f"{precision:g}"],
        }
        for method, precision in selected.items()
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    device = torch.device(f"cuda:{args.gpu}")

    components = tuple(sorted(set(args.components)))
    if not components or components[0] < 0 or components[-1] > 34:
        raise ValueError("PCA component counts must lie in [0, 34]")

    train = trainer.load_arrays([args.train_cache], 22)
    validation = trainer.load_arrays([args.validation_cache], 22)
    train_records = low.load_pickle(args.train_pkl)
    validation_records = low.load_pickle(args.validation_pkl)
    train_groups = build_four_view_groups(train_records)
    validation_groups = build_four_view_groups(validation_records)
    if len(train_groups) != len(train["targets"]):
        raise ValueError("train PKL/cache group mismatch")
    if len(validation_groups) != len(validation["targets"]):
        raise ValueError("validation PKL/cache group mismatch")

    fit_indices = np.flatnonzero(train["subjects"] != args.holdout_subject)
    holdout_indices = np.flatnonzero(train["subjects"] == args.holdout_subject)
    validation_indices = np.arange(len(validation["targets"]), dtype=np.int64)
    if args.max_holdout_groups:
        holdout_indices = holdout_indices[:args.max_holdout_groups]
    if args.max_validation_groups:
        validation_indices = validation_indices[:args.max_validation_groups]

    print(json.dumps({"status": "fitting 34-D residual PCA", "groups": len(fit_indices)}), flush=True)
    fit_rot, _, fit_observed, fit_projected = split_geometry(
        train_records, train_groups, train, fit_indices
    )
    mean, basis, eigenvalue = fit_pca(fit_projected - fit_observed)
    del fit_rot, fit_observed, fit_projected
    total_variance = eigenvalue.sum().clip(1e-20)
    explained = {
        f"K{k}": float(eigenvalue[:k].sum() / total_variance) if k else 0.0
        for k in components
    }

    hold_rot, hold_original_rays, hold_obs, hold_projected = split_geometry(
        train_records, train_groups, train, holdout_indices
    )
    val_rot, val_original_rays, val_obs, val_projected = split_geometry(
        validation_records, validation_groups, validation, validation_indices
    )
    hold_rays = {}
    validation_rays = {}
    hold_2d = {}
    validation_2d = {}
    for k in components:
        name = f"pca_k{k}"
        hold_corrected, hold_2d[name] = pca_oracle_correction(
            hold_obs, hold_projected, mean, basis, k
        )
        val_corrected, validation_2d[name] = pca_oracle_correction(
            val_obs, val_projected, mean, basis, k
        )
        hold_rays[name] = low.correction_to_rays(
            hold_corrected, hold_rot, hold_original_rays
        )
        validation_rays[name] = low.correction_to_rays(
            val_corrected, val_rot, val_original_rays
        )
        del hold_corrected, val_corrected

    frozen_args = SimpleNamespace(
        train_cache=args.train_cache,
        e2_checkpoint=args.e2_checkpoint,
        proposal_checkpoint=args.proposal_checkpoint,
        k96_checkpoint=args.k96_checkpoint,
    )
    frozen = FrozenK96Anchor(frozen_args, device)
    print(json.dumps({"status": "exporting frozen K96 S8 anchors", "groups": len(holdout_indices)}), flush=True)
    hold_anchors = low.frozen_k96_predictions(
        train, holdout_indices, frozen, device, args.batch_size, args.workers, args.seed
    )
    print(json.dumps({"status": "exporting frozen K96 S9/S11 anchors", "groups": len(validation_indices)}), flush=True)
    val_anchors = low.frozen_k96_predictions(
        validation, validation_indices, frozen, device,
        args.batch_size, args.workers, args.seed
    )

    hold_target = train["targets"][holdout_indices].astype(np.float64)
    val_target = validation["targets"][validation_indices].astype(np.float64)
    hold_actions = train["actions"][holdout_indices]
    val_actions = validation["actions"][validation_indices]
    precisions = tuple(float(value) for value in args.prior_precisions)
    print(json.dumps({"status": "selecting ray-MAP precision on S8"}), flush=True)
    hold_grid = low.evaluate_grid(
        hold_anchors, hold_target, hold_rays, hold_actions, precisions
    )
    selected = low.select_precision(hold_grid)
    print(json.dumps({"status": "single fixed-choice evaluation on S9/S11"}), flush=True)
    validation_grid = {}
    for method, precision in selected.items():
        validation_grid.update(low.evaluate_grid(
            val_anchors, val_target, {method: validation_rays[method]},
            val_actions, (precision,)
        ))
    validation_selected = selected_rows(validation_grid, selected)

    anchor_s8 = low.anchor_metrics(hold_anchors, hold_target, hold_actions)
    anchor_validation = low.anchor_metrics(val_anchors, val_target, val_actions)
    baseline = anchor_validation["headline_v234_mean_mm"]
    gains = {
        method: baseline - row["headline_v234_mean_mm"]
        for method, row in validation_selected.items()
    }
    k0_headline = validation_selected["pca_k0"]["headline_v234_mean_mm"]
    incremental = {
        method: k0_headline - row["headline_v234_mean_mm"]
        for method, row in validation_selected.items()
    }
    deployable_capacity = [f"pca_k{k}" for k in components if 1 <= k <= 8]
    best_low_k = max(deployable_capacity, key=lambda method: gains[method])
    high_k = [f"pca_k{k}" for k in components if k > 8]
    best_any = max(validation_selected, key=lambda method: gains[method])

    payload = {
        "method": "train-subject residual PCA basis + held-out oracle coefficients + frozen K96 ray-MAP",
        "input_protocol": "frozen HRNet coordinates/confidence + cameras; no heatmap/RGB/camera ID/temporal input",
        "scientific_boundary": {
            "basis": "34-D normalized-camera joint residual PCA fit only on S1/S5/S6/S7",
            "k0": "deployable train-subject mean residual control",
            "k_gt_0": "coefficients use held-out GT 2-D residual; diagnostic upper bounds only",
            "selection": "MAP precision selected on complete S8; S9/S11 evaluated once",
        },
        "counts": {
            "fit_s1_s5_s6_s7": int(len(fit_indices)),
            "holdout_s8": int(len(holdout_indices)),
            "validation_s9_s11": int(len(validation_indices)),
            "pca_samples_frame_view": int(len(fit_indices) * 4),
        },
        "pca": {
            "eigenvalues": eigenvalue.tolist(),
            "cumulative_explained_variance": explained,
        },
        "s8": {
            "k96_anchor": anchor_s8,
            "two_dimensional_reconstruction": hold_2d,
            "precision_grid": hold_grid,
            "selected_precision": selected,
        },
        "s9_s11": {
            "k96_anchor": anchor_validation,
            "two_dimensional_reconstruction": validation_2d,
            "selected_from_s8": validation_selected,
            "headline_gain_vs_k96_mm": gains,
            "incremental_gain_vs_k0_map_mm": incremental,
        },
        "decision": {
            "best_k_le_8": best_low_k,
            "best_k_le_8_gain_vs_k96_mm": gains[best_low_k],
            "best_k_le_8_incremental_vs_k0_map_mm": incremental[best_low_k],
            "k_le_8_passes_predictor_gate": bool(
                gains[best_low_k] >= 1.5 and incremental[best_low_k] >= 1.0
            ),
            "best_any_k": best_any,
            "best_any_gain_vs_k96_mm": gains[best_any],
            "k_gt_8_is_upper_bound_only": high_k,
        },
        "args": vars(args),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "pca_basis.npz",
        mean=mean.astype(np.float32), basis=basis.astype(np.float32),
        eigenvalue=eigenvalue.astype(np.float32),
    )
    output = output_dir / "result.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["decision"], indent=2), flush=True)
    print(json.dumps({"result": str(output.resolve())}), flush=True)


if __name__ == "__main__":
    main()
