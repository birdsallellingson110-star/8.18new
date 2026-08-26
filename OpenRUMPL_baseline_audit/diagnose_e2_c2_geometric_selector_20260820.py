#!/usr/bin/env python3
"""GT-free candidate selectors that try to approach the expanded-pool oracle.

The learned scorer keeps preferring the full-view H76 because its features
include view_fraction and excluded-view residual, which punish leave-one-out
and 1-view snap/bone hypotheses.  This diagnostic never reads GT when
selecting; GT is only used to report MPJPE.  Selectors are paper-backed
reprojection / robust residual / confidence / triangulation-angle rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_e2_c2_viewsnap_bone_oracle_20260819 import (
    ORIGINAL,
    action_equal,
    extra_candidates,
    combo_name,
    unit,
)
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-cache",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/"
            "e2_c2_input_protocol_v2/train_c2_22c.npz"
        ),
    )
    parser.add_argument(
        "--validation-cache",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260815/"
            "e2_c2_input_protocol_v2/validation_c2_22c.npz"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/mnt/data/cjyoutput/gbt_aligned_hrnet_20260819/"
            "e2_c2_geometric_selector/result.json"
        ),
    )
    return parser.parse_args()


def summarize_mm(error_m: np.ndarray, actions: np.ndarray) -> float:
    return action_equal(error_m * 1000.0, actions)


def ray_residual(poses: np.ndarray, rays: np.ndarray, views: tuple[int, ...]) -> np.ndarray:
    """poses: N,C,J,3 -> residual N,C,J,Vtask (meters)."""
    residuals = []
    for view in views:
        direction = unit(rays[:, :, view, :3])
        point = rays[:, :, view, 3:6]
        offset = poses - point[:, None]
        cross = np.cross(offset, direction[:, None])
        residuals.append(np.linalg.norm(cross, axis=-1))
    return np.stack(residuals, axis=-1)


def softmax_fuse(poses: np.ndarray, scores: np.ndarray, temperature: float) -> np.ndarray:
    # scores: N,C,J  (lower is better)
    weights = np.exp(-(scores - scores.min(axis=1, keepdims=True)) / max(temperature, 1e-6))
    weights = weights / weights.sum(axis=1, keepdims=True).clip(1e-8)
    return np.einsum("ncj,ncjd->njd", weights, poses)


def hard_pick(poses: np.ndarray, scores: np.ndarray) -> np.ndarray:
    index = scores.argmin(axis=1)
    gather = np.take_along_axis(
        poses, index[:, None, :, None].repeat(3, axis=-1), axis=1
    )
    return gather[:, 0]


def huber(values: np.ndarray, delta: float) -> np.ndarray:
    abs_v = np.abs(values)
    return np.where(abs_v <= delta, 0.5 * abs_v * abs_v, delta * (abs_v - 0.5 * delta))


def task_pool(predictions, rays, combo, bone_lengths):
    available = [
        index for index, candidate in enumerate(list(ORIGINAL) + list(ORIGINAL))
        if set(candidate).issubset(combo)
    ]
    extras = extra_candidates(predictions, rays, combo, bone_lengths)
    poses = np.concatenate(
        (predictions[:, available], extras["snap"], extras["bone"]), axis=1
    )
    n_pool = len(available)
    n_snap = extras["snap"].shape[1]
    families = (
        ["h76_or_conf"] * n_pool
        + ["snap"] * n_snap
        + ["bone"] * n_snap
    )
    return poses, families, ORIGINAL.index(combo)


def pairwise_angles(rays: np.ndarray, combo: tuple[int, ...]) -> float:
    if len(combo) < 2:
        return 90.0
    angles = []
    for i, a in enumerate(combo):
        da = unit(rays[:, :, a, :3])
        for b in combo[i + 1:]:
            db = unit(rays[:, :, b, :3])
            cosine = np.einsum("njd,njd->nj", da, db).clip(-1.0, 1.0)
            angles.append(np.degrees(np.arccos(np.abs(cosine))).mean())
    return float(np.min(angles))


def main() -> None:
    args = parse_args()
    train = np.load(args.train_cache)
    val = np.load(args.validation_cache)
    parents = np.asarray([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
    bone_lengths = np.linalg.norm(
        train["targets"].astype(np.float64) - train["targets"].astype(np.float64)[:, parents],
        axis=-1,
    ).mean(axis=0)
    bone_lengths[0] = 0.0
    predictions = val["predictions"].astype(np.float64)
    targets = val["targets"].astype(np.float64)
    rays = val["rays"].astype(np.float64)
    actions = val["actions"]
    conf = rays[..., 6]

    selector_names = [
        "baseline_h76",
        "oracle",
        "hard_mean_task_reproj",
        "soft_mean_task_reproj_t5mm",
        "hard_huber_task_reproj",
        "hard_median_task_reproj",
        "hard_min_task_reproj",
        "hard_inlier20_then_mean",
        "angle_gate_bone_maxconf",
        "angle_gate_soft_reproj",
    ]
    stores = {
        combo: {name: [] for name in selector_names}
        for combo in ORIGINAL
    }

    for combo in ORIGINAL:
        poses, families, baseline_local = task_pool(
            predictions, rays, combo, bone_lengths
        )
        residual = ray_residual(poses, rays, combo)
        mean_res = residual.mean(axis=-1)
        median_res = np.median(residual, axis=-1)
        min_res = residual.min(axis=-1)
        huber_res = huber(residual, 0.02).mean(axis=-1)
        inliers = (residual < 0.02).sum(axis=-1)
        inlier_score = -inliers.astype(np.float64) * 10.0 + mean_res
        true_err = np.linalg.norm(poses - targets[:, None], axis=-1)
        baseline = poses[:, baseline_local]
        angle = pairwise_angles(rays, combo)

        maxconf_view = int(np.argmax([conf[:, :, view].mean() for view in combo]))
        bone_offset = len(families) - len(combo)
        bone_maxconf = poses[:, bone_offset + maxconf_view]
        use_gate = angle < 25.0
        gated_bone = np.where(use_gate, bone_maxconf, baseline)
        gated_soft = np.where(
            use_gate,
            softmax_fuse(poses, mean_res, 0.005),
            baseline,
        )

        stores[combo]["baseline_h76"] = np.linalg.norm(baseline - targets, axis=-1)
        stores[combo]["oracle"] = true_err.min(axis=1)
        stores[combo]["hard_mean_task_reproj"] = np.linalg.norm(
            hard_pick(poses, mean_res) - targets, axis=-1
        )
        stores[combo]["soft_mean_task_reproj_t5mm"] = np.linalg.norm(
            softmax_fuse(poses, mean_res, 0.005) - targets, axis=-1
        )
        stores[combo]["hard_huber_task_reproj"] = np.linalg.norm(
            hard_pick(poses, huber_res) - targets, axis=-1
        )
        stores[combo]["hard_median_task_reproj"] = np.linalg.norm(
            hard_pick(poses, median_res) - targets, axis=-1
        )
        stores[combo]["hard_min_task_reproj"] = np.linalg.norm(
            hard_pick(poses, min_res) - targets, axis=-1
        )
        stores[combo]["hard_inlier20_then_mean"] = np.linalg.norm(
            hard_pick(poses, inlier_score) - targets, axis=-1
        )
        stores[combo]["angle_gate_bone_maxconf"] = np.linalg.norm(
            gated_bone - targets, axis=-1
        )
        stores[combo]["angle_gate_soft_reproj"] = np.linalg.norm(
            gated_soft - targets, axis=-1
        )

    def stage_mean(name, views):
        return float(np.mean([
            summarize_mm(stores[combo][name], actions)
            for combo in ORIGINAL if len(combo) == views
        ]))

    headline = {
        stage: {name: stage_mean(name, views) for name in selector_names}
        for stage, views in (("V2", 2), ("V3", 3), ("V4", 4))
    }
    combos = []
    for combo in ORIGINAL:
        combos.append({
            "combo": combo_name(combo),
            "views": len(combo),
            **{name: summarize_mm(stores[combo][name], actions) for name in selector_names},
        })

    payload = {
        "method": "GT-free geometric candidate selection on E2-C2 + snap/bone pool",
        "protocol": "clean H36M S9/S11 action-equal All-17; GT not used for selection",
        "e2_c2_soft_cal": {"V2": 38.700, "V3": 29.486, "V4": 27.274},
        "headline": headline,
        "combos": combos,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"headline": headline, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
