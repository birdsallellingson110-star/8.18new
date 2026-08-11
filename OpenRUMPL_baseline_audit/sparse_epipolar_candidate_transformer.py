"""Camera-identity-free Transformer for sparse heatmap candidate correction.

The module keeps RUMPL's detector fixed.  For one anatomical joint it receives
K heatmap modes from each available view, embeds their calibrated world rays,
and exchanges evidence only across cameras.  Pairwise shortest-ray distance is
added to attention logits as a geometric bias.

The scalar residual gate is initialized to zero, so the untrained model is
exactly the original heatmap ranking.  This avoids the destructive absolute
coordinate readout/cardinality shift observed in H17/H18.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as functional


def pairwise_ray_distance(
    centers: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """Shortest distance between all ray-supporting lines.

    Args:
        centers: ``(B, N, 3)`` camera centers in metres.
        directions: ``(B, N, 3)`` normalized world directions.
    Returns:
        Tensor of shape ``(B, N, N)`` in metres.
    """
    first_direction = directions[:, :, None]
    second_direction = directions[:, None, :]
    baseline = centers[:, None, :] - centers[:, :, None]
    cross = torch.cross(first_direction, second_direction, dim=-1)
    denominator = torch.linalg.vector_norm(cross, dim=-1)
    skew_distance = (
        (baseline * cross).sum(dim=-1).abs()
        / denominator.clamp_min(1e-7)
    )
    # The epipolar constraint degenerates for nearly parallel rays.  Fall back
    # to point-to-line distance instead of generating an unstable large value.
    point_line = torch.linalg.vector_norm(
        torch.cross(baseline, first_direction.expand_as(baseline), dim=-1),
        dim=-1,
    )
    return torch.where(denominator > 1e-5, skew_distance, point_line)


class GeometryBiasedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.geometry_strength = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        tokens: torch.Tensor,
        geometry_bias: torch.Tensor,
        cross_view_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        strength = functional.softplus(self.geometry_strength)[None, :, None, None]
        logits = logits + strength * geometry_bias[:, None]
        logits = logits.masked_fill(~cross_view_mask[:, None], -1e4)
        logits = logits.masked_fill(~valid_mask[:, None, None, :], -1e4)
        attention = functional.softmax(logits, dim=-1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(batch, length, dim)
        output = output * valid_mask.unsqueeze(-1)
        return self.projection(output)


class CandidateTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = GeometryBiasedAttention(dim, num_heads, dropout)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        geometry_bias: torch.Tensor,
        cross_view_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self.attention(
            self.norm_attention(tokens),
            geometry_bias,
            cross_view_mask,
            valid_mask,
        )
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return tokens * valid_mask.unsqueeze(-1)


class SparseEpipolarCandidateTransformer(nn.Module):
    """Rank top-K candidates independently in every available camera."""

    def __init__(
        self,
        num_joints: int = 17,
        dim: int = 96,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # score, log-score, normalized rank, ray xyz, relative camera xyz
        self.token_projection = nn.Linear(9, dim)
        self.joint_embedding = nn.Embedding(num_joints, dim)
        self.blocks = nn.ModuleList(
            [
                CandidateTransformerBlock(
                    dim, num_heads, mlp_ratio, dropout
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        self.residual_head = nn.Linear(dim, 1)
        self.residual_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        scores: torch.Tensor,
        centers: torch.Tensor,
        directions: torch.Tensor,
        joint_ids: torch.Tensor,
        view_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            scores: ``(B,V,K)`` HRNet peak values.
            centers: ``(B,V,3)`` camera centres in metres.
            directions: ``(B,V,K,3)`` normalized world rays.
            joint_ids: ``(B,)`` H36M joint indices.
            view_mask: optional ``(B,V)`` available-camera mask.
        Returns:
            Candidate logits of shape ``(B,V,K)``.
        """
        batch, n_views, topk = scores.shape
        if centers.shape != (batch, n_views, 3):
            raise ValueError(f"bad centers shape: {centers.shape}")
        if directions.shape != (batch, n_views, topk, 3):
            raise ValueError(f"bad directions shape: {directions.shape}")
        if view_mask is None:
            view_mask = torch.ones(
                batch, n_views, dtype=torch.bool, device=scores.device
            )
        valid = view_mask[:, :, None].expand(-1, -1, topk).reshape(batch, -1)
        view_ids = torch.arange(n_views, device=scores.device)
        view_ids = view_ids[:, None].expand(-1, topk).reshape(-1)

        safe_scores = scores.clamp_min(1e-6)
        rank = torch.arange(topk, device=scores.device, dtype=scores.dtype)
        rank = rank / max(topk - 1, 1)
        rank = rank[None, None].expand(batch, n_views, -1)
        centered = centers - (
            (centers * view_mask.unsqueeze(-1)).sum(dim=1, keepdim=True)
            / view_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
        )
        baseline_scale = torch.linalg.vector_norm(centered, dim=-1).amax(
            dim=1, keepdim=True
        ).clamp_min(1e-3)
        centered = centered / baseline_scale.unsqueeze(-1)
        center_features = centered[:, :, None].expand(-1, -1, topk, -1)
        features = torch.cat(
            (
                scores.unsqueeze(-1),
                safe_scores.log().unsqueeze(-1),
                rank.unsqueeze(-1),
                directions,
                center_features,
            ),
            dim=-1,
        ).reshape(batch, n_views * topk, 9)
        tokens = self.token_projection(features)
        tokens = tokens + self.joint_embedding(joint_ids)[:, None]

        flat_centers = centers[:, :, None].expand(-1, -1, topk, -1)
        flat_centers = flat_centers.reshape(batch, n_views * topk, 3)
        flat_directions = directions.reshape(batch, n_views * topk, 3)
        distance = pairwise_ray_distance(flat_centers, flat_directions)
        pair_cross_view = view_ids[:, None] != view_ids[None, :]
        diagonal = torch.eye(
            n_views * topk, dtype=torch.bool, device=scores.device
        )
        geometry_pairs = pair_cross_view[None].expand(batch, -1, -1)
        geometry_pairs = geometry_pairs & valid[:, :, None] & valid[:, None, :]
        cross_view = (pair_cross_view | diagonal)[None].expand(batch, -1, -1)
        cross_view = cross_view & valid[:, :, None] & valid[:, None, :]

        valid_distances = distance.masked_fill(~geometry_pairs, torch.inf)
        geometry_scale = valid_distances.amin(dim=-1)
        geometry_scale = geometry_scale.masked_fill(
            ~torch.isfinite(geometry_scale), 0.02
        )
        geometry_scale = geometry_scale.median(dim=-1, keepdim=True).values
        geometry_scale = geometry_scale.clamp_min(0.005)
        geometry_bias = -distance / geometry_scale[:, :, None]
        geometry_bias = geometry_bias.clamp(min=-20.0, max=0.0)

        for block in self.blocks:
            tokens = block(tokens, geometry_bias, cross_view, valid)
        correction = self.residual_head(self.output_norm(tokens)).squeeze(-1)
        correction = correction.reshape(batch, n_views, topk)
        base_logits = safe_scores.log()
        logits = base_logits + torch.tanh(self.residual_gate) * correction
        return logits.masked_fill(~view_mask[:, :, None], -1e4)


def candidate_loss(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
    view_mask: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over the K alternatives in each valid view."""
    batch, n_views, topk = logits.shape
    losses = functional.cross_entropy(
        logits.reshape(batch * n_views, topk),
        target_indices.reshape(batch * n_views),
        reduction="none",
    ).reshape(batch, n_views)
    if sample_weight is not None:
        losses = losses * sample_weight
    if view_mask is None:
        if sample_weight is None:
            return losses.mean()
        return losses.sum() / sample_weight.sum().clamp_min(1e-6)
    denominator = view_mask
    if sample_weight is not None:
        denominator = denominator * sample_weight
    return (losses * view_mask).sum() / denominator.sum().clamp_min(1e-6)
