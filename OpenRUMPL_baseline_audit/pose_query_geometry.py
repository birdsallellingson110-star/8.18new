#!/usr/bin/env python3
"""Shared calibrated geometry for iterative pose-query experiments."""

from __future__ import annotations

import numpy as np

from eval_h36m_sparse_epipolar_topk import (
    camera_parameters,
    pixels_to_rays,
    robust_intersection,
)


def image_to_heatmap(
    image_xy: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    size = np.asarray([width, height], dtype=np.float64)
    return (
        image_xy - center[:, None] + 0.5 * scale[:, None]
    ) / scale[:, None] * size


def heatmap_to_image(
    heatmap_xy: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    size = np.asarray([width, height], dtype=np.float64)
    return (
        heatmap_xy / size * scale[:, None]
        + center[:, None]
        - 0.5 * scale[:, None]
    )


def triangulate_points(
    records: list[dict],
    image_xy: np.ndarray,
    confidence: np.ndarray,
    irls_iterations: int = 5,
) -> np.ndarray:
    camera_data = [camera_parameters(record) for record in records]
    intrinsics = [item[0] for item in camera_data]
    rotations = [item[1] for item in camera_data]
    centers = np.stack([item[2] for item in camera_data])
    directions = np.stack(
        [
            pixels_to_rays(image_xy[view], intrinsics[view], rotations[view])
            for view in range(len(records))
        ]
    )
    return np.stack(
        [
            robust_intersection(
                centers,
                directions[:, joint],
                confidence[:, joint],
                irls_iterations,
            )
            for joint in range(image_xy.shape[1])
        ]
    )


def project_world_points(
    records: list[dict], world_points_m: np.ndarray
) -> np.ndarray:
    projected = []
    for record in records:
        intrinsic, rotation, center = camera_parameters(record)
        camera_points = (world_points_m - center) @ rotation.T
        homogeneous = camera_points @ intrinsic.T
        projected.append(
            homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-8)
        )
    return np.stack(projected)
