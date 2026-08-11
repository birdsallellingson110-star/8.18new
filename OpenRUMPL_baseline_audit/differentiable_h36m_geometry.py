#!/usr/bin/env python3
"""Differentiable calibrated H36M geometry used by H25.

The functions mirror the verified NumPy evaluator exactly while preserving
gradients from a refined heatmap coordinate through COCO-to-H36M conversion
and confidence-weighted ray intersection.
"""

from __future__ import annotations

import numpy as np
import torch

from eval_h36m_sparse_epipolar_topk import MMPOSE2H36M, camera_parameters


LOWER_BODY_SWAP = torch.tensor(
    [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    dtype=torch.long,
)


def heatmap_to_image_torch(
    heatmap_xy: torch.Tensor,
    center: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Invert the exact MMPose crop transform for VxJx2 coordinates."""
    center_tensor = torch.as_tensor(
        center, dtype=heatmap_xy.dtype, device=heatmap_xy.device
    )
    scale_tensor = torch.as_tensor(
        scale, dtype=heatmap_xy.dtype, device=heatmap_xy.device
    )
    size = heatmap_xy.new_tensor([width, height])
    return (
        heatmap_xy / size * scale_tensor[:, None]
        + center_tensor[:, None]
        - 0.5 * scale_tensor[:, None]
    )


def coco_to_h36m_torch(keypoints: torch.Tensor) -> torch.Tensor:
    """Convert Vx17x2 COCO coordinates with the verified H36M semantics."""
    if keypoints.ndim != 3 or tuple(keypoints.shape[-2:]) != (17, 2):
        raise ValueError(
            f"expected Vx17x2 COCO coordinates, got {tuple(keypoints.shape)}"
        )
    joints = keypoints.new_zeros((keypoints.shape[0], 17, 2))
    for destination, source in MMPOSE2H36M.items():
        joints[:, destination] = keypoints[:, source]
    joints[:, 10] = keypoints[:, 0:5].mean(dim=1)
    joints[:, 8] = keypoints[:, 3:7].mean(dim=1)
    joints[:, 0] = keypoints[:, 11:13].mean(dim=1)
    joints[:, 7] = (joints[:, 8] + joints[:, 0]) / 2.0
    permutation = LOWER_BODY_SWAP.to(device=keypoints.device)
    return joints.index_select(1, permutation)


def weighted_ray_anchor_torch(
    records: list[dict],
    h36m_xy: torch.Tensor,
    h36m_confidence: np.ndarray | torch.Tensor,
    confidence_epsilon: float = 0.05,
    regularization: float = 1e-4,
) -> torch.Tensor:
    """Exact differentiable counterpart of H20/H22's ray anchor."""
    if (
        h36m_xy.ndim != 3
        or h36m_xy.shape[0] != len(records)
        or tuple(h36m_xy.shape[-2:]) != (17, 2)
    ):
        raise ValueError(
            f"expected {len(records)}x17x2 coordinates, got "
            f"{tuple(h36m_xy.shape)}"
        )
    # Double precision keeps the 3x3 solve numerically aligned with the
    # evaluator.  Autograd still propagates to the float32 refiner.
    xy = h36m_xy.double()
    camera_data = [camera_parameters(record) for record in records]
    intrinsics = torch.as_tensor(
        np.stack([item[0] for item in camera_data]),
        dtype=xy.dtype,
        device=xy.device,
    )
    rotations = torch.as_tensor(
        np.stack([item[1] for item in camera_data]),
        dtype=xy.dtype,
        device=xy.device,
    )
    centers = torch.as_tensor(
        np.stack([item[2] for item in camera_data]),
        dtype=xy.dtype,
        device=xy.device,
    )
    homogeneous = torch.cat(
        (xy, torch.ones_like(xy[..., :1])), dim=-1
    )
    inverse_intrinsics = torch.linalg.inv(intrinsics)
    camera_rays = torch.einsum(
        "vjc,vkc->vjk", homogeneous, inverse_intrinsics
    )
    world_rays = torch.einsum(
        "vjc,vcd->vjd", camera_rays, rotations
    )
    directions = torch.nn.functional.normalize(world_rays, dim=-1)

    # Switch to JxV so every joint owns one independent 3x3 system.
    direction = directions.permute(1, 0, 2)
    weight = torch.as_tensor(
        h36m_confidence, dtype=xy.dtype, device=xy.device
    ).reshape(len(records), 17)
    weight = weight.clamp(0.0, 1.0).transpose(0, 1)
    weight = weight + confidence_epsilon
    identity = torch.eye(3, dtype=xy.dtype, device=xy.device)
    projection = (
        identity
        - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    )
    weighted_projection = projection * weight[..., None, None]
    lhs = weighted_projection.sum(dim=1) + regularization * identity
    rhs = (
        weighted_projection
        @ centers[None, :, :, None]
    ).sum(dim=1)
    return torch.linalg.solve(lhs, rhs).squeeze(-1)
