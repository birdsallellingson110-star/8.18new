#!/usr/bin/env python3
"""HeatFormer-inspired iterative pose-query refinement for keypoint heatmaps.

The current triangulated 3D pose is projected into every heatmap and used as
a query.  A local patch around that query, the frozen detector peak, and
cross-view/skeleton attention predict a small detector-coordinate residual.
There is no camera-ID embedding and attention is permutation equivariant over
views, so the same weights support arbitrary camera order and V=2/3/4.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class IterativePoseQueryRefiner(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        patch_size: int = 7,
        dimension: int = 96,
        heads: int = 4,
        view_depth: int = 2,
        skeleton_depth: int = 1,
        maximum_delta_pixels: float = 8.0,
    ) -> None:
        super().__init__()
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        self.num_joints = num_joints
        self.patch_size = patch_size
        self.maximum_delta_pixels = maximum_delta_pixels
        self.joint_embedding = nn.Embedding(num_joints, dimension)
        # patch, detector-query displacement, detector confidence, heatmap
        # value at query, normalized view cardinality
        input_dimension = patch_size * patch_size + 2 + 1 + 1 + 1
        self.input_projection = nn.Sequential(
            nn.Linear(input_dimension, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
        )
        view_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_transformer = nn.TransformerEncoder(
            view_layer, num_layers=view_depth
        )
        skeleton_layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=dimension * 3,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.skeleton_transformer = nn.TransformerEncoder(
            skeleton_layer, num_layers=skeleton_depth
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, 2),
        )
        # Exact detector identity at initialization.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def _query_patches(
        self, heatmaps: torch.Tensor, query_xy: torch.Tensor
    ) -> torch.Tensor:
        n_views, n_joints, height, width = heatmaps.shape
        radius = self.patch_size // 2
        offsets = torch.arange(
            -radius,
            radius + 1,
            dtype=heatmaps.dtype,
            device=heatmaps.device,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        offset_grid = torch.stack((xx, yy), dim=-1)
        grid_xy = query_xy[..., None, None, :] + offset_grid
        normalizer = heatmaps.new_tensor(
            [max(width - 1, 1), max(height - 1, 1)]
        )
        grid = grid_xy / normalizer * 2.0 - 1.0
        source = heatmaps.reshape(n_views * n_joints, 1, height, width)
        sampled = functional.grid_sample(
            source,
            grid.reshape(
                n_views * n_joints,
                self.patch_size,
                self.patch_size,
                2,
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled[:, 0].reshape(
            n_views, n_joints, self.patch_size * self.patch_size
        )

    def forward(
        self,
        heatmaps: torch.Tensor,
        query_xy: torch.Tensor,
        detector_xy: torch.Tensor,
        detector_confidence: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if heatmaps.ndim != 4:
            raise ValueError(f"expected VxJxHxW, got {heatmaps.shape}")
        n_views, n_joints, _, _ = heatmaps.shape
        coordinate_shape = (n_views, n_joints, 2)
        if tuple(query_xy.shape) != coordinate_shape:
            raise ValueError(
                f"expected query {coordinate_shape}, got {query_xy.shape}"
            )
        if tuple(detector_xy.shape) != coordinate_shape:
            raise ValueError(
                f"expected detector {coordinate_shape}, got "
                f"{detector_xy.shape}"
            )
        if tuple(detector_confidence.shape) != (n_views, n_joints):
            raise ValueError("detector_confidence must have shape VxJ")
        if n_views < 2:
            raise ValueError("pose-query refinement needs at least two views")
        if joint_ids is None:
            joint_ids = torch.arange(n_joints, device=heatmaps.device)
        joint_ids = joint_ids.to(device=heatmaps.device, dtype=torch.long)
        if tuple(joint_ids.shape) != (n_joints,):
            raise ValueError("joint_ids must have shape J")

        heatmaps = heatmaps.clamp_min(0.0)
        maximum = heatmaps.flatten(-2).amax(-1, keepdim=True)[..., None]
        normalized = heatmaps / maximum.clamp_min(1e-6)
        patches = self._query_patches(normalized, query_xy)
        query_value = patches[
            :, :, self.patch_size * self.patch_size // 2
        ]
        cardinality = heatmaps.new_full(
            (n_views, n_joints, 1), float(n_views) / 4.0
        )
        features = torch.cat(
            (
                patches,
                detector_xy - query_xy,
                detector_confidence[..., None],
                query_value[..., None],
                cardinality,
            ),
            dim=-1,
        )
        tokens = self.input_projection(features)
        joint_code = self.joint_embedding(joint_ids)
        tokens = tokens + joint_code[None]

        # For each joint, views form an unordered set.
        tokens = self.view_transformer(tokens.transpose(0, 1)).transpose(0, 1)
        # For each view, let the kinematic joint tokens exchange evidence.
        tokens = self.skeleton_transformer(tokens + joint_code[None])
        raw_delta = self.output(tokens)
        delta = self.maximum_delta_pixels * torch.tanh(raw_delta)
        refined = detector_xy + delta
        return refined, {
            "delta": delta,
            "raw_delta": raw_delta,
            "query_patches": patches,
            "query_value": query_value,
        }
