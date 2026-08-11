#!/usr/bin/env python3
"""Camera-ID-free gate for an H21 geometric-anchor correction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eval_h36m_sparse_epipolar_topk import camera_parameters, pixels_to_rays


FEATURE_NAMES = (
    "view_cardinality",
    "confidence_mean",
    "confidence_min",
    "confidence_std",
    "ray_residual_mean",
    "ray_residual_max",
    "ray_residual_std",
    "anchor_delta_norm",
)

CONTEXT_FEATURE_NAMES = FEATURE_NAMES + (
    "rumpl_residual_norm",
    "anchor_delta_rumpl_residual_cosine",
    "anchor_delta_to_rumpl_residual_ratio",
    "refined_ray_residual_mean",
    "refined_ray_residual_max",
    "ray_residual_mean_improvement",
    "ray_residual_max_improvement",
)


def _ray_residual(
    records: list[dict], h36m_xy: np.ndarray, anchor: np.ndarray
) -> np.ndarray:
    camera_data = [camera_parameters(record) for record in records]
    intrinsics = [item[0] for item in camera_data]
    rotations = [item[1] for item in camera_data]
    centers = np.stack([item[2] for item in camera_data])
    directions = np.stack(
        [
            pixels_to_rays(
                h36m_xy[view], intrinsics[view], rotations[view]
            )
            for view in range(len(records))
        ]
    )
    offset = anchor[None] - centers[:, None]
    return np.linalg.norm(np.cross(offset, directions), axis=-1)


def anchor_gate_features(
    records: list[dict],
    h36m_xy: np.ndarray,
    h36m_confidence: np.ndarray,
    old_anchor: np.ndarray,
    anchor_delta: np.ndarray,
) -> np.ndarray:
    """Return 17x8 invariant reliability features.

    No camera ID, absolute camera position, or absolute pose coordinate is
    exposed.  Ray distances and anchor changes are normalized to a 5 cm
    reference scale.
    """
    confidence = np.asarray(h36m_confidence, dtype=np.float64).reshape(
        len(records), 17
    )
    ray_residual = _ray_residual(records, h36m_xy, old_anchor)
    reference_m = 0.05
    n_joints = old_anchor.shape[0]
    return np.stack(
        (
            np.full(n_joints, len(records) / 4.0),
            confidence.mean(axis=0),
            confidence.min(axis=0),
            confidence.std(axis=0),
            ray_residual.mean(axis=0) / reference_m,
            ray_residual.max(axis=0) / reference_m,
            ray_residual.std(axis=0) / reference_m,
            np.linalg.norm(anchor_delta, axis=-1) / reference_m,
        ),
        axis=-1,
    ).astype(np.float32)


def anchor_context_gate_features(
    records: list[dict],
    h36m_xy: np.ndarray,
    refined_h36m_xy: np.ndarray,
    h36m_confidence: np.ndarray,
    old_anchor: np.ndarray,
    anchor_delta: np.ndarray,
    rumpl_prediction: np.ndarray,
) -> np.ndarray:
    """Add invariant H22/refinement context to the original H24 features."""
    base = anchor_gate_features(
        records,
        h36m_xy,
        h36m_confidence,
        old_anchor,
        anchor_delta,
    )
    reference_m = 0.05
    rumpl_residual = np.asarray(rumpl_prediction) - np.asarray(old_anchor)
    anchor_delta = np.asarray(anchor_delta)
    residual_norm = np.linalg.norm(rumpl_residual, axis=-1)
    delta_norm = np.linalg.norm(anchor_delta, axis=-1)
    cosine = (
        (anchor_delta * rumpl_residual).sum(axis=-1)
        / np.maximum(delta_norm * residual_norm, 1e-8)
    )
    ratio = delta_norm / np.maximum(residual_norm, 0.005)

    old_ray_residual = _ray_residual(records, h36m_xy, old_anchor)
    refined_anchor = old_anchor + anchor_delta
    refined_ray_residual = _ray_residual(
        records, refined_h36m_xy, refined_anchor
    )
    context = np.stack(
        (
            residual_norm / reference_m,
            cosine,
            np.minimum(ratio, 10.0),
            refined_ray_residual.mean(axis=0) / reference_m,
            refined_ray_residual.max(axis=0) / reference_m,
            (
                old_ray_residual.mean(axis=0)
                - refined_ray_residual.mean(axis=0)
            )
            / reference_m,
            (
                old_ray_residual.max(axis=0)
                - refined_ray_residual.max(axis=0)
            )
            / reference_m,
        ),
        axis=-1,
    )
    return np.concatenate((base, context.astype(np.float32)), axis=-1)


class AnchorDeltaGate(nn.Module):
    def __init__(
        self,
        num_joints: int = 17,
        feature_dimension: int = len(FEATURE_NAMES),
        joint_dimension: int = 16,
        hidden_dimension: int = 32,
        initial_gate: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_dimension = feature_dimension
        self.joint_embedding = nn.Embedding(num_joints, joint_dimension)
        self.network = nn.Sequential(
            nn.Linear(feature_dimension + joint_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Linear(hidden_dimension // 2, 1),
        )
        # Begin from the H23 diagnostic while allowing train-only evidence to
        # move every joint/sample gate independently.
        nn.init.zeros_(self.network[-1].weight)
        initial_logit = np.log(initial_gate / (1.0 - initial_gate))
        nn.init.constant_(self.network[-1].bias, float(initial_logit))

    def forward(
        self,
        features: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            features.ndim != 2
            or features.shape[-1] != self.feature_dimension
        ):
            raise ValueError(
                f"expected Jx{self.feature_dimension} features, got "
                f"{tuple(features.shape)}"
            )
        if joint_ids is None:
            joint_ids = torch.arange(
                features.shape[0], device=features.device
            )
        joint_ids = joint_ids.to(device=features.device, dtype=torch.long)
        tokens = torch.cat(
            (features, self.joint_embedding(joint_ids)), dim=-1
        )
        return torch.sigmoid(self.network(tokens)).squeeze(-1)


class SignedResidualContextGate(nn.Module):
    """Signed [-1, 1] gate initialized exactly from a positive context gate."""

    def __init__(
        self,
        base_gate: AnchorDeltaGate | None = None,
        num_joints: int = 17,
        feature_dimension: int = len(CONTEXT_FEATURE_NAMES),
        joint_dimension: int = 16,
        hidden_dimension: int = 32,
    ) -> None:
        super().__init__()
        if base_gate is None:
            base_gate = AnchorDeltaGate(
                num_joints=num_joints,
                feature_dimension=feature_dimension,
                joint_dimension=joint_dimension,
                hidden_dimension=hidden_dimension,
            )
        self.feature_dimension = feature_dimension
        self.base_gate = base_gate
        for parameter in self.base_gate.parameters():
            parameter.requires_grad_(False)
        self.residual_joint_embedding = nn.Embedding(
            num_joints, joint_dimension
        )
        self.residual_network = nn.Sequential(
            nn.Linear(
                feature_dimension + joint_dimension, hidden_dimension
            ),
            nn.LayerNorm(hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.GELU(),
            nn.Linear(hidden_dimension // 2, 1),
        )
        # tanh(atanh(base) + 0) == base, so H27 starts exactly at H26.
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            features.ndim != 2
            or features.shape[-1] != self.feature_dimension
        ):
            raise ValueError(
                f"expected Jx{self.feature_dimension} features, got "
                f"{tuple(features.shape)}"
            )
        if joint_ids is None:
            joint_ids = torch.arange(
                features.shape[0], device=features.device
            )
        joint_ids = joint_ids.to(device=features.device, dtype=torch.long)
        with torch.no_grad():
            base = self.base_gate(features, joint_ids)
        tokens = torch.cat(
            (
                features,
                self.residual_joint_embedding(joint_ids),
            ),
            dim=-1,
        )
        signed_residual = self.residual_network(tokens).squeeze(-1)
        base_latent = torch.atanh(base.clamp(-0.999999, 0.999999))
        return torch.tanh(base_latent + signed_residual)
