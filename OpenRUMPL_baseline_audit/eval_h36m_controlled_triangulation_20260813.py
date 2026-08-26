#!/usr/bin/env python3
"""Evaluate public triangulation controls on one fixed H36M 2D cache.

This is deliberately independent of RUMPL checkpoints.  Every method receives
the same synchronized 2D coordinates, confidences and calibrated cameras.  It
reports all 6/4/1 camera combinations for V2/V3/V4 and both action-equal and
frame-weighted absolute MPJPE.  GT-2D controls are included to catch convention,
distortion, synchronization and unit errors before comparing learned models.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np


ACTION_NAMES = {
    2: "Direction", 3: "Discuss", 4: "Eating", 5: "Greet", 6: "Phone",
    7: "Photo", 8: "Pose", 9: "Purchase", 10: "Sitting",
    11: "SittingDown", 12: "Smoke", 13: "Wait", 14: "WalkDog",
    15: "Walk", 16: "WalkTwo",
}
JOINT_NAMES = (
    "root", "rhip", "rkne", "rank", "lhip", "lkne", "lank", "belly",
    "neck", "nose", "head", "lsho", "lelb", "lwri", "rsho", "relb", "rwri",
)
# Joints directly observed by COCO rather than constructed by averaging/extrapolation.
DIRECT_COCO_H36M = (1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-pkl", required=True)
    parser.add_argument("--pred-pkl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--irls-iters", type=int, default=5)
    parser.add_argument("--huber-threshold-mm", type=float, default=20.0)
    parser.add_argument("--ransac-iters", type=int, default=10)
    parser.add_argument("--ransac-epsilon-px", type=float, default=20.0)
    parser.add_argument("--ransac-seed", type=int, default=42)
    parser.add_argument(
        "--pred-only-auto",
        action="store_true",
        help=(
            "Evaluate only predicted 2D in the coordinate system explicitly "
            "stored by its cache. This skips the four GT/raw diagnostic arms "
            "and is the fast path for a protocol that already passed them."
        ),
    )
    parser.add_argument(
        "--algebraic-only",
        action="store_true",
        help=(
            "Evaluate only confidence-weighted algebraic DLT. This avoids the "
            "much slower per-joint RANSAC/IRLS controls for dense benchmarks."
        ),
    )
    return parser.parse_args()


def group_records(records: list[dict]) -> OrderedDict[tuple, list[dict]]:
    grouped: OrderedDict[tuple, dict[int, dict]] = OrderedDict()
    for record in records:
        key = (
            int(record["subject"]), int(record["action"]),
            int(record["subaction"]), int(record["image_id"]),
        )
        grouped.setdefault(key, {})[int(record["camera_id"])] = record
    output = OrderedDict()
    for key, by_camera in grouped.items():
        if set(by_camera) == {0, 1, 2, 3}:
            output[key] = [by_camera[index] for index in range(4)]
    return output


def camera_matrices(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = record["camera"]
    k = np.asarray(camera.get("K"), dtype=np.float64)
    if k.shape != (3, 3):
        k = np.array([
            [float(camera["fx"]), 0.0, float(camera["cx"])],
            [0.0, float(camera["fy"]), float(camera["cy"])],
            [0.0, 0.0, 1.0],
        ])
    rotation = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    center = np.asarray(camera["T"], dtype=np.float64).reshape(3)
    radial = np.asarray(camera.get("k", np.zeros(3)), dtype=np.float64).reshape(-1)
    tangential = np.asarray(camera.get("p", np.zeros(2)), dtype=np.float64).reshape(-1)
    distortion = np.array([
        radial[0] if radial.size > 0 else 0.0,
        radial[1] if radial.size > 1 else 0.0,
        tangential[0] if tangential.size > 0 else 0.0,
        tangential[1] if tangential.size > 1 else 0.0,
        radial[2] if radial.size > 2 else 0.0,
    ], dtype=np.float64)
    return k, rotation, center, distortion


def undistort_pixels(pixels: np.ndarray, k: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    shape = pixels.shape
    return cv2.undistortPoints(
        pixels.reshape(-1, 1, 2).astype(np.float64), k, distortion, P=k
    ).reshape(shape)


def make_geometry(records: list[dict], pixels: np.ndarray, undistort: bool):
    centers, directions, projections = [], [], []
    corrected = []
    for record, points in zip(records, pixels):
        k, rotation, center, distortion = camera_matrices(record)
        points = undistort_pixels(points, k, distortion) if undistort else points
        homogeneous = np.concatenate(
            [points.astype(np.float64), np.ones((len(points), 1))], axis=1
        )
        camera_rays = homogeneous @ np.linalg.inv(k).T
        world_rays = camera_rays @ rotation
        world_rays /= np.linalg.norm(world_rays, axis=1, keepdims=True).clip(1e-12)
        # H36M camera convention: X_cam = R (X_world - C).
        translation = -rotation @ center.reshape(3, 1)
        projections.append(k @ np.concatenate([rotation, translation], axis=1))
        centers.append(center)
        directions.append(world_rays)
        corrected.append(points)
    return (
        np.asarray(centers), np.asarray(directions),
        np.asarray(projections), np.asarray(corrected),
    )


def ray_intersection(
    centers: np.ndarray, directions: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    # centers: V,3; directions: V,J,3; weights: V,J
    identity = np.eye(3, dtype=np.float64)
    projection = identity[None, None] - directions[..., :, None] * directions[..., None, :]
    weighted = projection * weights[..., None, None]
    lhs = weighted.sum(axis=0) + 1e-8 * identity[None]
    rhs = np.einsum("vjab,va->jb", weighted, centers)
    return np.linalg.solve(lhs, rhs[..., None])[..., 0]


def solve_ray(
    centers: np.ndarray,
    directions: np.ndarray,
    confidences: np.ndarray,
    mode: str,
    iterations: int,
    huber_threshold_mm: float,
) -> np.ndarray:
    weights = np.ones_like(confidences) if mode == "uniform" else np.clip(confidences, 1e-4, 1.0)
    estimate = ray_intersection(centers, directions, weights)
    if mode != "irls":
        return estimate
    for _ in range(iterations):
        residual = np.linalg.norm(
            np.cross(estimate[None] - centers[:, None], directions), axis=-1
        )
        robust = np.minimum(1.0, huber_threshold_mm / np.maximum(residual, 1e-8))
        estimate = ray_intersection(centers, directions, weights * robust)
    return estimate


def algebraic_dlt(
    projections: np.ndarray, pixels: np.ndarray, confidences: np.ndarray
) -> np.ndarray:
    # Direct Linear Transform with LT-style per-view/per-joint confidence rows.
    n_views, n_joints = pixels.shape[:2]
    estimates = np.empty((n_joints, 3), dtype=np.float64)
    for joint in range(n_joints):
        rows = []
        for view in range(n_views):
            p = projections[view]
            x, y = pixels[view, joint]
            weight = float(np.clip(confidences[view, joint], 1e-4, 1.0))
            rows.extend([weight * (x * p[2] - p[0]), weight * (y * p[2] - p[1])])
        _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
        homogeneous = vh[-1]
        estimates[joint] = homogeneous[:3] / homogeneous[3]
    return estimates


def project_points(projections: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points_3d, np.ones((len(points_3d), 1), dtype=np.float64)], axis=1
    )
    projected = np.einsum("vab,jb->vja", projections, homogeneous)
    return projected[..., :2] / np.clip(projected[..., 2:3], 1e-12, None)


def adafuse_ransac(
    projections: np.ndarray,
    pixels: np.ndarray,
    iterations: int,
    epsilon_px: float,
    rng: random.Random,
) -> np.ndarray:
    """Coordinate-equivalent port of AdaFuse's public test-time RANSAC.

    AdaFuse samples two views, keeps the hypothesis with the largest inlier set
    under a pixel reprojection threshold, then retriangulates all inliers with
    unweighted DLT.  The upstream implementation uses ``niter=10`` and
    ``epsilon=20``.  We expose and record the RNG seed because upstream relies
    on Python's global RNG and does not publish the evaluation seed.
    """
    n_views, n_joints = pixels.shape[:2]
    if n_views < 2:
        raise ValueError("RANSAC triangulation needs at least two views")
    estimates = np.empty((n_joints, 3), dtype=np.float64)
    view_ids = list(range(n_views))
    unit_conf = np.ones((n_views, n_joints), dtype=np.float64)
    for joint in range(n_joints):
        best_inliers: list[int] = []
        for _ in range(iterations):
            sampled = sorted(rng.sample(view_ids, 2))
            hypothesis = algebraic_dlt(
                projections[sampled], pixels[sampled, joint : joint + 1],
                unit_conf[sampled, joint : joint + 1]
            )[0]
            reprojection = project_points(
                projections, hypothesis.reshape(1, 3)
            )[:, 0]
            errors = np.linalg.norm(reprojection - pixels[:, joint], axis=-1)
            inliers = np.flatnonzero(errors < epsilon_px).tolist()
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
            if len(best_inliers) == n_views:
                break
        if len(best_inliers) < 2:
            best_inliers = view_ids
        estimates[joint] = algebraic_dlt(
            projections[best_inliers], pixels[best_inliers, joint : joint + 1],
            unit_conf[best_inliers, joint : joint + 1]
        )[0]
    return estimates


def summarize(errors: np.ndarray, actions: np.ndarray) -> dict:
    # errors: N,J in mm
    per_action = {
        ACTION_NAMES[action]: float(errors[actions == action].mean())
        for action in ACTION_NAMES if np.any(actions == action)
    }
    direct_per_action = {
        ACTION_NAMES[action]: float(errors[actions == action][:, DIRECT_COCO_H36M].mean())
        for action in ACTION_NAMES if np.any(actions == action)
    }
    per_joint = errors.mean(axis=0)
    return {
        "action_equal_all17_mm": float(np.mean(list(per_action.values()))),
        "frame_weighted_all17_mm": float(errors.mean()),
        "action_equal_direct13_mm": float(np.mean(list(direct_per_action.values()))),
        "frame_weighted_direct13_mm": float(errors[:, DIRECT_COCO_H36M].mean()),
        "num_samples": int(len(errors)),
        "per_action_all17_mm": per_action,
        "per_joint_frame_weighted_mm": {
            name: float(value) for name, value in zip(JOINT_NAMES, per_joint)
        },
    }


def main() -> None:
    args = parse_args()
    ransac_rng = random.Random(args.ransac_seed)
    with open(args.gt_pkl, "rb") as handle:
        gt_groups = group_records(pickle.load(handle))
    with open(args.pred_pkl, "rb") as handle:
        pred_groups = group_records(pickle.load(handle))
    common_keys = [key for key in gt_groups if key in pred_groups]
    if args.limit_groups:
        common_keys = common_keys[: args.limit_groups]
    if not common_keys:
        raise RuntimeError("No complete synchronized four-view groups in common")

    accum: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    actions: dict[str, list[int]] = defaultdict(list)
    two_d_errors = []
    for key in common_keys:
        gt_records = gt_groups[key]
        pred_records = pred_groups[key]
        if [x["image"] for x in gt_records] != [x["image"] for x in pred_records]:
            raise RuntimeError(f"Image alignment mismatch at {key}")
        gt_pixels = np.asarray([x["joints_2d"] for x in gt_records], dtype=np.float64)
        pred_pixels = np.asarray([x["joints_2d"] for x in pred_records], dtype=np.float64)
        confidences = np.asarray([
            x.get("joints_2d_conf", np.ones((17, 1))) for x in pred_records
        ], dtype=np.float64).reshape(4, 17)
        target = np.asarray(gt_records[0]["joints_3d"], dtype=np.float64)
        two_d_errors.append(np.linalg.norm(pred_pixels - gt_pixels, axis=-1))

        for n_views in (2, 3, 4):
            for subset in itertools.combinations(range(4), n_views):
                subset_records = [gt_records[index] for index in subset]
                subset_pred = pred_pixels[list(subset)]
                subset_gt = gt_pixels[list(subset)]
                subset_conf = confidences[list(subset)]
                stage = f"V{n_views}"
                source_inputs = (("pred2d", subset_pred),) if args.pred_only_auto else (
                    ("pred2d", subset_pred), ("gt2d", subset_gt)
                )
                for source, pixels in source_inputs:
                    # A coordinate cache may already be undistorted and carry
                    # an explicit zero-distortion camera.  Use that camera for
                    # predicted points; otherwise an ``undistort=True`` audit
                    # would undistort the exported points a second time.  The
                    # historical caches have no marker and retain the old
                    # gt-record behavior for backwards compatibility.
                    if source == "pred2d" and any(
                        "camera_2d_coordinate_system" in x for x in pred_records
                    ):
                        geometry_records = [pred_records[index] for index in subset]
                    else:
                        geometry_records = subset_records
                    cache_has_explicit_coordinates = source == "pred2d" and any(
                        "camera_2d_coordinate_system" in x for x in pred_records
                    )
                    coordinate_modes = (
                        ((False,) if cache_has_explicit_coordinates else (True,))
                        if args.pred_only_auto else (False, True)
                    )
                    for undistort in coordinate_modes:
                        tag = (
                            "cache_coordinates" if args.pred_only_auto else
                            ("undistorted" if undistort else "raw_distorted")
                        )
                        centers, directions, projections, corrected = make_geometry(
                            geometry_records, pixels, undistort
                        )
                        algebraic_key = f"{source}_{tag}_algebraic_confidence"
                        if args.algebraic_only:
                            estimates = {
                                algebraic_key: algebraic_dlt(
                                    projections, corrected, subset_conf
                                )
                            }
                        else:
                            estimates = {
                                f"{source}_{tag}_ray_uniform": solve_ray(
                                    centers, directions, subset_conf, "uniform",
                                    args.irls_iters, args.huber_threshold_mm,
                                ),
                                f"{source}_{tag}_ray_confidence": solve_ray(
                                    centers, directions, subset_conf, "confidence",
                                    args.irls_iters, args.huber_threshold_mm,
                                ),
                                f"{source}_{tag}_ray_irls": solve_ray(
                                    centers, directions, subset_conf, "irls",
                                    args.irls_iters, args.huber_threshold_mm,
                                ),
                                algebraic_key: algebraic_dlt(
                                    projections, corrected, subset_conf
                                ),
                                f"{source}_{tag}_adafuse_ransac": adafuse_ransac(
                                    projections, corrected, args.ransac_iters,
                                    args.ransac_epsilon_px, ransac_rng,
                                ),
                            }
                        for method, estimate in estimates.items():
                            accum[stage][method].append(
                                np.linalg.norm(estimate - target, axis=-1)
                            )
                            actions[stage].append(key[1]) if method == next(iter(estimates)) else None

    report = {
        "protocol": {
            "metric": "absolute MPJPE, all 17 joints, no alignment",
            "camera_combinations": "all 6/4/1 combinations for V2/V3/V4",
            "gt_pkl": str(Path(args.gt_pkl).resolve()),
            "pred_pkl": str(Path(args.pred_pkl).resolve()),
            "complete_four_view_groups": len(common_keys),
            "irls_iters": args.irls_iters,
            "huber_threshold_mm": args.huber_threshold_mm,
            "ransac": {
                "source": "AdaFuse public adafuse_network.py",
                "iterations": args.ransac_iters,
                "epsilon_px": args.ransac_epsilon_px,
                "seed": args.ransac_seed,
            },
            "pred_only_auto": args.pred_only_auto,
            "algebraic_only": args.algebraic_only,
            "predicted_geometry_camera_policy": (
                "use predicted cache cameras when camera_2d_coordinate_system is present; "
                "otherwise legacy ground-truth camera behavior"
            ),
        },
        "pred2d_vs_gt2d_px": {
            "mean": float(np.asarray(two_d_errors).mean()),
            "median": float(np.median(np.asarray(two_d_errors))),
        },
        "results": {},
    }
    for stage, methods in accum.items():
        repeats = len(next(iter(methods.values()))) // len(common_keys)
        stage_actions = np.repeat(
            np.asarray([key[1] for key in common_keys], dtype=np.int64), repeats
        )
        # itertools loops combinations inside each frame, so actions must tile by frame.
        stage_actions = np.asarray([
            key[1] for key in common_keys for _ in range(repeats)
        ], dtype=np.int64)
        report["results"][stage] = {
            method: summarize(np.asarray(values), stage_actions)
            for method, values in methods.items()
        }

    rendered = json.dumps(report, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
