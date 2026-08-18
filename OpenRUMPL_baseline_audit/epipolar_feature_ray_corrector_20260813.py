#!/usr/bin/env python3
"""Identity-initialized feature-epipolar correction of RUMPL input rays.

The module preserves the H76 input/output contract.  It consumes frozen HRNet
epipolar correspondence descriptors and predicts a small tangent-plane update
to each existing observation ray.  There is no camera-ID embedding and the
view transformer is permutation equivariant.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class EpipolarFeatureRayCorrector(nn.Module):
    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int = 96,
        heads: int = 4,
        layers: int = 1,
        max_angle_degrees: float = 0.5,
        use_descriptors: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.descriptor_dim = int(descriptor_dim)
        self.max_angle_radians = math.radians(max_angle_degrees)
        self.use_descriptors = bool(use_descriptors)
        self.joint_embedding = nn.Parameter(torch.zeros(17, 16))
        self.descriptor_encoder = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # direction(3), point-to-pose perpendicular(3), residual norm(1),
        # confidence(1), root-relative pose(3), view fraction(1), joint(16).
        self.geometry_encoder = nn.Sequential(
            nn.Linear(28, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.view_transformer = nn.TransformerEncoder(block, num_layers=layers)
        self.output = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        rays: torch.Tensor,
        baseline_pose: torch.Tensor,
        descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Correct ``B,J,V,7`` rays using ``B,V,J,D`` descriptors."""
        if rays.ndim != 4 or rays.shape[1] != 17 or rays.shape[-1] != 7:
            raise ValueError(f"unexpected rays shape {tuple(rays.shape)}")
        batch, joints, views, _ = rays.shape
        if descriptors.shape != (
            batch, views, joints, self.descriptor_dim
        ):
            raise ValueError(
                f"descriptor/ray mismatch {tuple(descriptors.shape)} vs "
                f"{(batch, views, joints, self.descriptor_dim)}"
            )

        raw_direction = rays[..., :3]
        direction_norm = torch.linalg.vector_norm(
            raw_direction, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        direction = raw_direction / direction_norm
        point = rays[..., 3:6]
        offset = baseline_pose[:, :, None, :] - point
        perpendicular = offset - (offset * direction).sum(
            dim=-1, keepdim=True
        ) * direction
        residual_norm = torch.linalg.vector_norm(
            perpendicular, dim=-1, keepdim=True
        )
        root_relative = baseline_pose - baseline_pose[:, :1]
        root_relative = root_relative[:, :, None].expand(-1, -1, views, -1)
        joint = self.joint_embedding[None, :, None].expand(
            batch, -1, views, -1
        )
        view_fraction = torch.full(
            (batch, joints, views, 1), views / 4.0,
            dtype=rays.dtype, device=rays.device,
        )
        geometry = torch.cat(
            (
                direction,
                perpendicular / 0.05,
                residual_norm / 0.05,
                rays[..., 6:7].clamp(0, 1),
                root_relative / 0.5,
                view_fraction,
                joint,
            ),
            dim=-1,
        )
        if geometry.shape[-1] != 28:
            raise RuntimeError(f"geometry feature size {geometry.shape[-1]}")

        tokens = self.geometry_encoder(geometry)
        if self.use_descriptors:
            descriptor_tokens = descriptors.permute(0, 2, 1, 3)
            tokens = tokens + self.descriptor_encoder(descriptor_tokens)
        tokens = tokens.reshape(batch * joints, views, -1)
        tokens = self.view_transformer(tokens)
        raw = self.output(tokens).reshape(batch, joints, views, 4)

        tangent = raw[..., :3]
        tangent = tangent - (tangent * direction).sum(
            dim=-1, keepdim=True
        ) * direction
        tangent_squared = tangent.square().sum(dim=-1, keepdim=True)
        bounded_tangent = tangent / torch.sqrt(1.0 + tangent_squared)
        alpha = torch.sigmoid(raw[..., 3:4])
        update = self.max_angle_radians * bounded_tangent * alpha
        new_unit = (direction + update) / torch.sqrt(
            1.0 + update.square().sum(dim=-1, keepdim=True)
        )

        corrected = rays.clone()
        # Express as a delta so the zero-initialized model is tensor-identical
        # to H76 input, avoiding numerical drift in near-parallel ray systems.
        corrected[..., :3] = (
            raw_direction + (new_unit - direction) * direction_norm
        )
        return corrected, {
            "angle_radians": torch.atan(
                torch.linalg.vector_norm(update, dim=-1)
            ),
            "alpha": alpha.squeeze(-1),
        }
