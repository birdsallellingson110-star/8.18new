#!/usr/bin/env python3
"""Export GT-free ray-geometry quality features for frozen H36M subsets.

The record order is defined by the selection manifest emitted by
``eval_rumpl_checkpoint.py``.  Only predicted 2D keypoints, detector
confidence and camera calibration are used; 3D targets are deliberately never
read.  The output therefore can be consumed by a train/test gate without
ground-truth leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "ray_angle_min_rad",
    "ray_angle_mean_rad",
    "ray_angle_max_rad",
    "log1p_ray_distance_mean_mm",
    "log1p_ray_distance_max_mm",
    "normal_eigen_log_fraction_0",
    "normal_eigen_log_fraction_1",
    "normal_eigen_log_fraction_2",
    "confidence_min",
    "confidence_mean",
    "confidence_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_rays(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit world rays, camera centers (mm), and confidence."""
    camera = record["camera"]
    points = np.asarray(record["joints_2d"], dtype=np.float64)
    fx, fy = float(camera["fx"]), float(camera["fy"])
    cx, cy = float(camera["cx"]), float(camera["cy"])
    camera_points = np.stack(
        ((points[:, 0] - cx) / fx, (points[:, 1] - cy) / fy,
         np.ones(len(points), dtype=np.float64)),
        axis=-1,
    )
    rotation = np.asarray(camera["R"], dtype=np.float64)
    directions = camera_points @ rotation
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-12)
    centers = np.broadcast_to(
        np.asarray(camera["T"], dtype=np.float64).reshape(1, 3),
        directions.shape,
    )
    confidence = np.asarray(
        record.get("joints_2d_conf", np.ones((len(points), 1))),
        dtype=np.float64,
    ).reshape(len(points), -1)[:, 0]
    return directions, centers, confidence


def geometry_features(records: list[dict]) -> np.ndarray:
    directions, centers, confidence = zip(*(camera_rays(x) for x in records))
    # (J,V,3), (J,V,3), (J,V)
    directions = np.stack(directions, axis=1)
    centers = np.stack(centers, axis=1)
    confidence = np.stack(confidence, axis=1)
    joints, views = directions.shape[:2]
    if views < 2:
        raise ValueError("ray geometry requires at least two views")
    pair_indices = np.asarray(
        [(left, right) for left in range(views) for right in range(left + 1, views)],
        dtype=np.int64,
    )
    left, right = pair_indices[:, 0], pair_indices[:, 1]
    cross = np.cross(directions[:, left], directions[:, right])
    cross_norm = np.linalg.norm(cross, axis=-1)
    dot = np.sum(directions[:, left] * directions[:, right], axis=-1)
    # Triangulation conditioning belongs to unoriented lines: theta and
    # pi-theta describe the same line pair.  Use the acute intersection angle.
    angles = np.arctan2(cross_norm, np.abs(np.clip(dot, -1.0, 1.0)))

    center_delta = centers[:, right] - centers[:, left]
    skew_distance = np.abs(np.sum(center_delta * cross, axis=-1)) / np.clip(
        cross_norm, 1e-12, None
    )
    parallel_distance = np.linalg.norm(
        np.cross(center_delta, directions[:, left]), axis=-1
    )
    ray_distance = np.where(cross_norm > 1e-8, skew_distance, parallel_distance)

    eye = np.eye(3, dtype=np.float64)
    projection = eye - directions[..., :, None] * directions[..., None, :]
    # Confidence controls contribution, while normalization prevents the tiny
    # raw LT confidence scale from collapsing the matrix numerically.
    weights = np.clip(confidence, 0.0, None)
    weights /= np.clip(weights.mean(axis=1, keepdims=True), 1e-12, None)
    normal = np.sum(weights[..., None, None] * projection, axis=1)
    eigenvalues = np.linalg.eigvalsh(normal).clip(1e-12)
    eigen_fractions = eigenvalues / eigenvalues.sum(axis=-1, keepdims=True)

    features = np.stack(
        (
            angles.min(axis=1),
            angles.mean(axis=1),
            angles.max(axis=1),
            np.log1p(ray_distance.mean(axis=1)),
            np.log1p(ray_distance.max(axis=1)),
            np.log(eigen_fractions[:, 0]),
            np.log(eigen_fractions[:, 1]),
            np.log(eigen_fractions[:, 2]),
            confidence.min(axis=1),
            confidence.mean(axis=1),
            confidence.max(axis=1),
        ),
        axis=-1,
    )
    if features.shape != (joints, len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise ValueError(f"invalid feature matrix: {features.shape}")
    return features.astype(np.float32)


def main() -> None:
    args = parse_args()
    with open(args.input_pkl, "rb") as stream:
        database = pickle.load(stream)
    if not isinstance(database, list):
        raise TypeError("expected the H36M pickle to contain a record list")
    with open(args.selection_manifest, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    output_features = []
    camera_ids = []
    for position, group in enumerate(manifest["groups"]):
        indices = [int(value) for value in group["record_indices"]]
        records = [database[index] for index in indices]
        images = [record["image"] for record in records]
        if images != group["images"]:
            raise ValueError(f"manifest/database alignment failed at group {position}")
        ids = [int(record["camera_id"]) for record in records]
        if ids != [int(value) for value in group["camera_ids"]]:
            raise ValueError(f"camera ID alignment failed at group {position}")
        output_features.append(geometry_features(records))
        camera_ids.append(ids)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            features=np.stack(output_features),
            feature_names=np.asarray(FEATURE_NAMES),
            camera_ids=np.asarray(camera_ids, dtype=np.int8),
            record_indices=np.asarray(
                [group["record_indices"] for group in manifest["groups"]],
                dtype=np.int32,
            ),
            input_pkl_sha256=np.asarray(sha256(args.input_pkl)),
            manifest_sha256=np.asarray(sha256(args.selection_manifest)),
            uses_ground_truth=np.asarray(False),
        )
    temporary.replace(output)
    print(json.dumps({
        "output": str(output),
        "shape": list(np.stack(output_features).shape),
        "features": FEATURE_NAMES,
        "uses_ground_truth": False,
    }, indent=2))


if __name__ == "__main__":
    main()
