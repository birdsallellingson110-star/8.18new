#!/usr/bin/env python3
"""Identity-initialized, trust-region correction of H36M observation rays."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveRayCorrector(nn.Module):
    """Predict a bounded tangent-plane update for each joint/view ray.

    Camera points and detector confidences are never changed.  The last layer is
    initialized to zero, so a newly constructed module is an exact identity map.
    """

    def __init__(self, max_angle_degrees: float = 0.5):
        super().__init__()
        self.max_angle_radians = math.radians(max_angle_degrees)
        self.joint_embedding = nn.Parameter(torch.zeros(17, 16))
        self.network = nn.Sequential(
            nn.Linear(29, 96), nn.ReLU6(),
            nn.Linear(96, 64), nn.ReLU6(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        rays: torch.Tensor,
        baseline_pose: torch.Tensor,
        harm_gate: torch.Tensor,
        use_utility_gate: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return corrected rays and differentiable correction diagnostics.

        Args:
            rays: ``B,J,V,7`` direction, point-on-ray and confidence.
            baseline_pose: frozen H76 pose for this exact view subset, ``B,J,3``.
            harm_gate: C4 counterfactual gate, ``B,J,V`` in [0, 1].
            use_utility_gate: D1 if true, geometry-only D0 otherwise.
        """
        if rays.ndim != 4 or rays.shape[1] != 17 or rays.shape[-1] != 7:
            raise ValueError(f"unexpected rays shape {tuple(rays.shape)}")
        direction_raw = rays[..., :3]
        direction_norm = torch.linalg.vector_norm(
            direction_raw, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        direction = direction_raw / direction_norm
        point = rays[..., 3:6]
        offset = baseline_pose[:, :, None, :] - point
        perpendicular = offset - (offset * direction).sum(
            dim=-1, keepdim=True
        ) * direction
        residual_norm = torch.linalg.vector_norm(
            perpendicular, dim=-1, keepdim=True
        )
        batch, joints, views, _ = rays.shape
        root_relative = baseline_pose - baseline_pose[:, :1]
        root_relative = root_relative[:, :, None].expand(-1, -1, views, -1)
        joint = self.joint_embedding[None, :, None].expand(
            batch, -1, views, -1
        )
        view_fraction = torch.full(
            (batch, joints, views, 1), views / 4.0,
            dtype=rays.dtype, device=rays.device,
        )
        gate_feature = harm_gate[..., None] if use_utility_gate else torch.zeros(
            batch, joints, views, 1, dtype=rays.dtype, device=rays.device
        )
        features = torch.cat(
            (
                direction,
                perpendicular / 0.05,
                residual_norm / 0.05,
                rays[..., 6:7].clamp(0, 1),
                gate_feature,
                view_fraction,
                root_relative / 0.5,
                joint,
            ),
            dim=-1,
        )
        if features.shape[-1] != 29:
            raise RuntimeError(f"ray correction feature size {features.shape[-1]}")

        raw = self.network(features)
        tangent = raw[..., :3]
        tangent = tangent - (tangent * direction).sum(
            dim=-1, keepdim=True
        ) * direction
        tangent_squared = tangent.square().sum(dim=-1, keepdim=True)
        # Smoothly bound a tangent vector without taking ||t|| as the forward
        # parameter.  Unlike norm-times-unit-vector parameterizations, this has
        # a useful derivative at the exact zero/identity initialization.
        bounded_tangent = tangent / torch.sqrt(1.0 + tangent_squared)
        learned_alpha = torch.sigmoid(raw[..., 3:4])
        effective_gate = harm_gate[..., None] if use_utility_gate else 1.0
        tangent_update = (
            self.max_angle_radians * bounded_tangent
            * learned_alpha * effective_gate
        )
        # tangent_update is perpendicular to the unit direction, hence this is
        # the exact normalization factor.  At zero update the expression
        # returns the same `direction` tensor bit-for-bit; F.normalize would do
        # an unnecessary second normalization and break strict identity.
        new_unit = (direction + tangent_update) / torch.sqrt(
            1.0 + tangent_update.square().sum(dim=-1, keepdim=True)
        )
        angle = torch.atan(torch.linalg.vector_norm(
            tangent_update, dim=-1, keepdim=True
        ))
        base_angle = torch.atan(
            self.max_angle_radians
            * torch.linalg.vector_norm(bounded_tangent, dim=-1, keepdim=True)
        )
        # Preserve the original direction norm so theta=0 is a tensor-level
        # identity, not merely equivalent after RUMPL's internal normalization.
        corrected = rays.clone()
        # Express the update as a delta from the already loaded tensor.  This
        # avoids a normalize/rescale round trip at theta=0, which can perturb a
        # nearly parallel ray system enough to move triangulation by millimetres.
        corrected[..., :3] = direction_raw + (new_unit - direction) * direction_norm
        diagnostics = {
            "angle_radians": angle.squeeze(-1),
            "base_angle_radians": base_angle.squeeze(-1),
            "learned_alpha": learned_alpha.squeeze(-1),
            "harm_gate": harm_gate,
        }
        return corrected, diagnostics


def counterfactual_harm_gate(
    utility_model: nn.Module,
    predictions: torch.Tensor,
    rays: torch.Tensor,
    task_combo: tuple[int, ...],
    combinations: tuple[tuple[int, ...], ...],
    task_spec_fn,
) -> torch.Tensor:
    """Convert C4 predicted leave-one-view gains to bounded view harm gates."""
    batch, _, joints, _ = predictions.shape
    if len(task_combo) == 2:
        return torch.ones(
            batch, joints, 2, device=predictions.device, dtype=predictions.dtype
        )
    available, candidate_masks, task_mask = task_spec_fn(
        task_combo, predictions.device
    )
    candidates = predictions[:, available]
    with torch.no_grad():
        raw = utility_model(candidates, rays, candidate_masks, task_mask)
    baseline_global = combinations.index(task_combo)
    baseline_local = available.index(baseline_global)
    predicted_delta = raw - raw[..., baseline_local:baseline_local + 1]
    gates = []
    for view in task_combo:
        dropped = tuple(item for item in task_combo if item != view)
        dropped_global = combinations.index(dropped)
        dropped_local = available.index(dropped_global)
        # C4 delta is normalized by 10 mm.  A negative drop-delta says this
        # view harms the current joint.  Zero/positive deltas close the gate.
        predicted_harm = F.relu(-predicted_delta[..., dropped_local])
        gates.append(1.0 - torch.exp(-predicted_harm))
    return torch.stack(gates, dim=-1)
