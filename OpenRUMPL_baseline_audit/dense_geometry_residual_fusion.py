#!/usr/bin/env python3
"""Camera-identity-free residual fusion for dense multi-view pose heatmaps."""

from __future__ import annotations

import torch
from torch import nn


class DenseGeometryResidualFusion(nn.Module):
    """Fuse geometry-aligned heatmaps while preserving the detector at init.

    Inputs
    ------
    heatmaps:
        ``V x J x H x W`` non-negative heatmaps, normalized per view/joint.
    pairwise_support:
        ``V x V x J x H x W``.  Entry ``[i, k]`` is view ``k`` warped onto
        view ``i`` along calibrated rays; diagonal entries are ignored.

    The module contains no camera embedding and reduces source views with
    symmetric mean/max operations, so the same weights accept V=2,3,4 and new
    camera ordering.  Both trainable correction paths are exactly zero at
    initialization; the initial argmax is therefore the frozen MMPose result.
    """

    def __init__(
        self,
        num_joints: int = 17,
        joint_embedding_dim: int = 8,
        hidden_channels: int = 32,
        maximum_geometry_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.maximum_geometry_weight = maximum_geometry_weight
        self.joint_embedding = nn.Embedding(
            num_joints, joint_embedding_dim
        )

        # Shared spatial correction.  It sees appearance, geometry support,
        # their agreement, and a semantic joint code, but never camera ID.
        input_channels = 5 + joint_embedding_dim
        self.spatial_residual = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.spatial_residual[-1].weight)
        nn.init.zeros_(self.spatial_residual[-1].bias)

        # Per-view/joint adaptive geometry gate.  Global strength starts at
        # exactly zero, so arbitrary random MLP output cannot alter baseline.
        descriptor_dim = 7 + joint_embedding_dim
        self.local_gate = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.global_geometry_strength = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _entropy(heatmaps: torch.Tensor, epsilon: float) -> torch.Tensor:
        probability = heatmaps.flatten(-2)
        probability = probability / probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(epsilon)
        entropy = -(probability * torch.log(probability + epsilon)).sum(-1)
        return entropy / torch.log(
            entropy.new_tensor(float(probability.shape[-1]))
        )

    def forward(
        self,
        heatmaps: torch.Tensor,
        pairwise_support: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if heatmaps.ndim != 4:
            raise ValueError(f"expected VxJxHxW, got {heatmaps.shape}")
        n_views, n_joints, height, width = heatmaps.shape
        expected = (n_views, n_views, n_joints, height, width)
        if tuple(pairwise_support.shape) != expected:
            raise ValueError(
                f"expected support {expected}, got {pairwise_support.shape}"
            )
        if n_views < 2:
            raise ValueError("dense geometry fusion requires at least 2 views")
        if joint_ids is None:
            joint_ids = torch.arange(n_joints, device=heatmaps.device)
        else:
            joint_ids = joint_ids.to(
                device=heatmaps.device, dtype=torch.long
            )
        if joint_ids.shape != (n_joints,):
            raise ValueError(
                f"expected {n_joints} joint IDs, got {joint_ids.shape}"
            )
        if int(joint_ids.min()) < 0 or int(joint_ids.max()) >= self.num_joints:
            raise ValueError(
                f"joint IDs must be in [0, {self.num_joints - 1}]"
            )

        epsilon = 1e-4
        heatmaps = heatmaps.clamp_min(0.0)
        support = pairwise_support.clamp_min(0.0)
        eye = torch.eye(
            n_views, dtype=torch.bool, device=heatmaps.device
        )
        support = support.masked_fill(
            eye[:, :, None, None, None], 0.0
        )
        cross_mean = support.sum(dim=1) / float(n_views - 1)
        cross_max = support.masked_fill(
            eye[:, :, None, None, None], -torch.inf
        ).amax(dim=1)
        cross_max = torch.nan_to_num(
            cross_max, nan=0.0, neginf=0.0, posinf=0.0
        )

        peak_index = heatmaps.flatten(-2).argmax(dim=-1)
        support_at_peak = cross_mean.flatten(-2).gather(
            -1, peak_index[..., None]
        )[..., 0]
        joint_code = self.joint_embedding(joint_ids)
        joint_code_v = joint_code[None].expand(n_views, -1, -1)
        descriptors = torch.cat(
            (
                heatmaps.flatten(-2).amax(-1)[..., None],
                cross_mean.flatten(-2).amax(-1)[..., None],
                cross_max.flatten(-2).amax(-1)[..., None],
                support_at_peak[..., None],
                (heatmaps * cross_mean).flatten(-2).mean(-1)[..., None],
                self._entropy(heatmaps, epsilon)[..., None],
                heatmaps.new_full(
                    (n_views, n_joints, 1),
                    float(n_views) / 4.0,
                ),
                joint_code_v,
            ),
            dim=-1,
        )
        local_gate = torch.sigmoid(self.local_gate(descriptors)[..., 0])
        geometry_weight = (
            self.maximum_geometry_weight
            * self.global_geometry_strength
            * local_gate
        )

        joint_maps = joint_code_v[..., None, None].expand(
            -1, -1, -1, height, width
        )
        spatial_input = torch.cat(
            (
                heatmaps[..., None, :, :],
                cross_mean[..., None, :, :],
                cross_max[..., None, :, :],
                (heatmaps * cross_mean)[..., None, :, :],
                torch.abs(heatmaps - cross_mean)[..., None, :, :],
                joint_maps,
            ),
            dim=2,
        ).reshape(n_views * n_joints, -1, height, width)
        spatial_residual = self.spatial_residual(spatial_input).reshape(
            n_views, n_joints, height, width
        )

        base_logits = torch.log(heatmaps + epsilon)
        geometry_logits = torch.log(cross_mean + epsilon)
        fused_logits = (
            base_logits
            + geometry_weight[..., None, None] * geometry_logits
            + spatial_residual
        )
        return fused_logits, {
            "geometry_weight": geometry_weight,
            "local_gate": local_gate,
            "spatial_residual": spatial_residual,
            "cross_mean": cross_mean,
        }
