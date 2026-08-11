#!/usr/bin/env python3
"""Sample strict H36M AMASS shards and verify geometry before training."""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np


BONES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
]


def project(points: np.ndarray, camera: dict, translation_key: str) -> np.ndarray:
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera[translation_key], dtype=np.float64).reshape(3)
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    camera_points = points @ rotation.T + translation
    homogeneous = camera_points @ intrinsic.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glob")
    parser.add_argument("--indices", default="0,49,98")
    parser.add_argument("--samples-per-shard", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob), key=lambda path: int(Path(path).name.split("_")[1]))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")
    requested = [int(value) for value in args.indices.split(",")]
    selected = [files[index] for index in requested]
    rng = np.random.default_rng(args.seed)

    errors_t = []
    errors_T = []
    camera_relation = []
    bone_lengths = []
    detector_errors = []
    detector_confidence = []
    confidence = []
    root_xyz = []
    total_samples = 0

    for path in selected:
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        n = data["joints_3d"].shape[0]
        sample_indices = rng.choice(n, min(n, args.samples_per_shard), replace=False)
        total_samples += len(sample_indices)
        for sample_index in sample_indices:
            points = np.asarray(data["joints_3d"][sample_index], dtype=np.float64)
            root_xyz.append(points[0])
            bone_lengths.extend(np.linalg.norm(points[[b for _, b in BONES]] - points[[a for a, _ in BONES]], axis=1))
            confidence.extend(np.asarray(data["confs_2d_mmpose"][sample_index]).reshape(-1))
            for view_index, camera in enumerate(data["camera_parameters_all"][sample_index]):
                target = np.asarray(data["joints_2d_amass"][sample_index, view_index], dtype=np.float64)
                detected = np.asarray(data["joints_2d_mmpose"][sample_index, view_index], dtype=np.float64)
                errors_t.extend(np.linalg.norm(project(points, camera, "t") - target, axis=1))
                errors_T.extend(np.linalg.norm(project(points, camera, "T") - target, axis=1))
                detector_errors.extend(np.linalg.norm(detected - target, axis=1))
                detector_confidence.extend(
                    np.asarray(data["confs_2d_mmpose"][sample_index, view_index]).reshape(-1)
                )
                rotation = np.asarray(camera["R"], dtype=np.float64)
                center = np.asarray(camera["T"], dtype=np.float64).reshape(3)
                translation = np.asarray(camera["t"], dtype=np.float64).reshape(3)
                camera_relation.append(np.linalg.norm(translation + rotation @ center))

    def describe(name: str, values: list[float]) -> None:
        array = np.asarray(values)
        print(
            f"{name}: mean={array.mean():.6f} median={np.median(array):.6f} "
            f"p95={np.percentile(array, 95):.6f} max={array.max():.6f}"
        )

    print(f"files={len(files)} sampled_shards={[Path(path).name for path in selected]}")
    print(f"sampled_poses={total_samples} sampled_views={total_samples * 20}")
    describe("reprojection_Rx_plus_t_px", errors_t)
    describe("reprojection_Rx_plus_T_px", errors_T)
    describe("camera_relation_norm_t_plus_RT", camera_relation)
    describe("mmpose_vs_gt_px", detector_errors)
    detector_errors_array = np.asarray(detector_errors)
    detector_confidence_array = np.asarray(detector_confidence)
    for threshold in (0.05, 0.5, 0.8):
        valid = detector_confidence_array > threshold
        print(f"mmpose_conf_gt_{threshold}: ratio={valid.mean():.6f}")
        describe(f"mmpose_vs_gt_conf_gt_{threshold}_px", detector_errors_array[valid].tolist())
    describe("bone_length_m", bone_lengths)
    describe("confidence", confidence)
    roots = np.asarray(root_xyz)
    print(f"root_xyz_min={roots.min(axis=0).tolist()} root_xyz_max={roots.max(axis=0).tolist()}")
    print(f"all_finite={all(np.isfinite(np.asarray(values)).all() for values in [errors_t, errors_T, bone_lengths, confidence, roots])}")


if __name__ == "__main__":
    main()
