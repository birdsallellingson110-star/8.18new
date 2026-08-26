#!/usr/bin/env python3
"""Deterministic image-level H36M-Occl protocol shared by both 2D frontends.

Geometry-Biased Transformer (Moliner et al., FG 2024) defines H36M-Occl as
white square masks placed over every projected 2D joint independently with
probability 0.1.  The paper does not publish code, square size, random seed,
or explicitly state whether the mask is inserted before the RGB detector or
directly into the 2D tensor.  This module follows the image-level
interpretation suggested by the cited synthetic-occlusion work and makes all
missing choices explicit and reproducible.  It is intentionally shared by the
HRNet and ResNet exporters so both frontends see pixel-identical masks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


PROTOCOL_NAME = "GBT-H36M-Occl-white-square-p01-deterministic-v1"


@dataclass(frozen=True)
class OcclusionResult:
    image: np.ndarray
    masked_joints: np.ndarray
    centers_xy: np.ndarray
    square_side_px: int


def distortion_from_record(record: dict[str, Any]) -> np.ndarray:
    """Convert RUMPL/H36M k,p storage to OpenCV coefficient order."""
    camera = record["camera"]
    radial = np.asarray(camera.get("k", np.zeros(3)), dtype=np.float64).reshape(-1)
    tangential = np.asarray(
        camera.get("p", np.zeros(2)), dtype=np.float64
    ).reshape(-1)
    if radial.size < 3 or tangential.size < 2:
        raise ValueError("H36M camera must contain k[3] and p[2]")
    return np.asarray(
        [radial[0], radial[1], tangential[0], tangential[1], radial[2]],
        dtype=np.float64,
    )


def undistorted_joint_pixels(record: dict[str, Any]) -> np.ndarray:
    """Return RUMPL-order GT joints in the full undistorted K_new=K image."""
    points = np.asarray(record["joints_2d"], dtype=np.float64).reshape(-1, 2)
    intrinsic = np.asarray(record["camera"]["K"], dtype=np.float64).reshape(3, 3)
    points = cv2.undistortPoints(
        points.reshape(-1, 1, 2),
        intrinsic,
        distortion_from_record(record),
        P=intrinsic,
    ).reshape(-1, 2)
    if points.shape != (17, 2) or not np.isfinite(points).all():
        raise ValueError(f"expected 17 finite undistorted joints, got {points.shape}")
    return points


def _record_identity(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(record["subject"]),
        int(record["action"]),
        int(record["subaction"]),
        int(record["image_id"]),
        int(record["camera_id"]),
    )


def deterministic_uniform(
    record: dict[str, Any], joint_index: int, seed: int
) -> float:
    """Stable U[0,1) independent of process count, order, Python, and NumPy."""
    identity = _record_identity(record)
    payload = ":".join(str(value) for value in (*identity, int(joint_index), int(seed)))
    digest = hashlib.sha256(payload.encode("ascii")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(1 << 64)


def selected_joint_mask(
    record: dict[str, Any], probability: float, seed: int
) -> np.ndarray:
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("occlusion probability must be in [0,1]")
    return np.asarray(
        [
            deterministic_uniform(record, joint_index, seed) < probability
            for joint_index in range(17)
        ],
        dtype=bool,
    )


def square_side_from_record(record: dict[str, Any], fraction: float) -> int:
    """Scale the missing GBT square-size parameter by the annotation box."""
    if not np.isfinite(fraction) or fraction <= 0:
        raise ValueError("occlusion square fraction must be finite and positive")
    box = np.asarray(record["box"], dtype=np.float64).reshape(-1)
    if box.size != 4 or not np.isfinite(box).all():
        raise ValueError("record has no finite four-value annotation box")
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid annotation box {box.tolist()}")
    return max(1, int(round(float(fraction) * max(width, height))))


def apply_white_joint_squares(
    image: np.ndarray,
    record: dict[str, Any],
    probability: float,
    square_fraction: float,
    seed: int,
) -> OcclusionResult:
    """Apply opaque white squares after full-image undistortion.

    ``image`` must already be in the full undistorted K_new=K coordinate
    system.  The input is never modified in place.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {image.shape}")
    result = image.copy()
    centers = undistorted_joint_pixels(record)
    masked = selected_joint_mask(record, probability, seed)
    side = square_side_from_record(record, square_fraction)
    left_extent = side // 2
    right_extent = side - left_extent
    height, width = result.shape[:2]
    for center in centers[masked]:
        x, y = np.rint(center).astype(np.int64)
        x1 = max(0, int(x) - left_extent)
        x2 = min(width, int(x) + right_extent)
        y1 = max(0, int(y) - left_extent)
        y2 = min(height, int(y) + right_extent)
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = 255
    return OcclusionResult(
        image=result,
        masked_joints=np.flatnonzero(masked).astype(np.int16),
        centers_xy=centers.astype(np.float32),
        square_side_px=int(side),
    )


def protocol_manifest(probability: float, square_fraction: float, seed: int) -> dict:
    return {
        "name": PROTOCOL_NAME,
        "paper": "Geometry-Biased Transformer (Moliner et al., FG 2024)",
        "paper_specified": {
            "mask": "opaque white square centered on projected 2D joint",
            "joint_probability": float(probability),
            "model_training": "clean H36M only",
        },
        "paper_unspecified_reconstructed": {
            "square_side": "fraction of longer H36M annotation-box side",
            "square_fraction": float(square_fraction),
            "seed": int(seed),
            "rng": "SHA256(subject,action,subaction,image_id,camera_id,joint,seed)",
            "sampling_correlation": "independent per frame/camera/joint (GBT gives marginal p but not correlation)",
            "mask_insertion_point": "RGB image before detector/crop (operational interpretation; GBT does not state this explicitly)",
        },
        "coordinate_order": "undistort full image K_new=K, then mask, then detector/crop",
        "reference_code": {
            "repository": "https://github.com/isarandi/synthetic-occlusion",
            "local_commit": "3d627bbbeb5dd548d3fecb775c869ab08133f422",
            "boundary": "cited provenance only; repository pastes VOC objects, not GBT white squares",
        },
    }
