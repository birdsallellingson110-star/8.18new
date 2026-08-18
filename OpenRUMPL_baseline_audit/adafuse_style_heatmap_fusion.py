#!/usr/bin/env python3
"""Camera-identity-free AdaFuse-style dense heatmap fusion.

This module follows the public AdaFuse design at the level that matters for
our coordinate-only audit: a shared per-joint/per-view reliability network
uses heatmap appearance and calibrated cross-view consistency, then uses the
predicted view weights to fuse the current heatmap with epipolar-warped
source heatmaps.  It deliberately does not use a camera ID, a fixed view
order, RGB features, or temporal information.

The input ``pairwise_support[target, source]`` is the dense epipolar support
map produced by our calibrated heatmap warper.  Therefore the class accepts
any number of views (>=2) and is usable with unseen view permutations.
"""

from __future__ import annotations

import torch
from torch import nn


class AdaFuseStyleHeatmapFusion(nn.Module):
    """Learn reliability weights and fuse dense epipolar heatmap evidence."""

    def __init__(
        self,
        num_joints: int = 17,
        hidden: int = 96,
        temperature: float = 1.0,
        source_strength: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_joints = int(num_joints)
        self.temperature = float(temperature)
        # A warped heatmap is evidence, not a replacement for the detector's
        # own observation.  The public AdaFuse implementation normalizes its
        # warped maps by the source peak before fusion; our exported maps are
        # already max-normalized but can still contain a full unit peak for
        # every source.  This conservative coefficient is the equivalent
        # scale guard and makes the zero-update model reduce to the measured
        # dense_add(alpha=.25) control instead of silently becoming a pure
        # cross-view map.
        self.source_strength = float(source_strength)

        # The descriptor mirrors AdaFuse's two sources of evidence:
        # heatmap appearance/confidence and cross-view geometric agreement.
        # All operations are shared over views and joints.
        descriptor_dim = 9
        self.reliability = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Equal view weights at initialization are the ScoreFuse-style
        # identity; this prevents an arbitrary camera preference before 2D
        # supervision has provided evidence.
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    @staticmethod
    def _entropy(heatmaps: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        prob = heatmaps.flatten(-2)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(eps)
        entropy = -(prob * torch.log(prob.clamp_min(eps))).sum(dim=-1)
        return entropy / torch.log(
            entropy.new_tensor(float(prob.shape[-1]))
        ).clamp_min(eps)

    @staticmethod
    def _peak(heatmaps: torch.Tensor) -> torch.Tensor:
        return heatmaps.flatten(-2).amax(dim=-1)

    def forward(
        self,
        heatmaps: torch.Tensor,
        pairwise_support: torch.Tensor,
        joint_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if heatmaps.ndim != 4:
            raise ValueError(f"expected VxJxHxW heatmaps, got {heatmaps.shape}")
        n_views, n_joints, height, width = heatmaps.shape
        expected = (n_views, n_views, n_joints, height, width)
        if tuple(pairwise_support.shape) != expected:
            raise ValueError(
                f"expected pairwise support {expected}, got {pairwise_support.shape}"
            )
        if n_views < 2:
            raise ValueError("AdaFuse-style fusion requires at least two views")
        if joint_ids is not None and tuple(joint_ids.shape) != (n_joints,):
            raise ValueError("joint_ids must have shape (J,)")

        eps = 1e-5
        heatmaps = heatmaps.clamp_min(0.0)
        support = pairwise_support.clamp_min(0.0)
        eye = torch.eye(n_views, dtype=torch.bool, device=heatmaps.device)
        support = support.masked_fill(eye[:, :, None, None, None], 0.0)
        # ``target, source`` is the storage convention.  ``target_mean`` is
        # used to fuse each target heatmap; ``source_mean`` is the descriptor
        # for a source view's reliability (AdaFuse scores the source view).
        cross_mean = support.sum(dim=1) / float(n_views - 1)
        source_mean = support.sum(dim=0) / float(n_views - 1)
        cross_max = support.masked_fill(
            eye[:, :, None, None, None], -torch.inf
        ).amax(dim=1)
        cross_max = torch.nan_to_num(cross_max, nan=0.0, neginf=0.0)
        source_max = support.masked_fill(
            eye[:, :, None, None, None], -torch.inf
        ).amax(dim=0)
        source_max = torch.nan_to_num(source_max, nan=0.0, neginf=0.0)

        peak = self._peak(heatmaps)
        cross_peak = self._peak(source_mean)
        peak_index = heatmaps.flatten(-2).argmax(dim=-1)
        support_at_peak = source_mean.flatten(-2).gather(
            -1, peak_index[..., None]
        )[..., 0]
        agreement = (heatmaps * source_mean).flatten(-2).mean(dim=-1)
        entropy = self._entropy(heatmaps)
        # A sharpness statistic is useful because a high peak in a diffuse map
        # should not receive the same reliability as a concentrated peak.
        mass = heatmaps.flatten(-2).mean(dim=-1)
        sharpness = peak / mass.clamp_min(eps)
        descriptors = torch.stack(
            (
                peak,
                entropy,
                sharpness.clamp(max=100.0) / 100.0,
                cross_peak,
                support_at_peak,
                agreement,
                self._peak(source_max),
                heatmaps.new_full((n_views, n_joints), float(n_views) / 4.0),
                (peak - cross_peak).abs(),
            ),
            dim=-1,
        )
        # Reliability logits are camera-ID-free.  Softmax is over the views
        # for each joint, exactly the permutation-equivariant set operation
        # needed by variable-view evaluation.
        reliability_logits = self.reliability(descriptors)[..., 0]
        weights = torch.softmax(
            reliability_logits / max(self.temperature, eps), dim=0
        )
        # Current view's own detector map is the diagonal term; source-view
        # terms are already aligned to the current view by pairwise_support.
        aligned = support.clone()
        aligned[eye] = heatmaps
        # weights[source,j] is shared for every target: this is the same
        # source reliability used by the public AdaFuse implementation.
        weighted_sources = (
            aligned * weights[None, :, :, None, None]
        ).sum(dim=1)
        # Keep the target detector as an explicit identity path.  This is also
        # important for fair comparison: a learned gate may down-weight a bad
        # source, but it cannot erase a good target observation merely because
        # the number of views changed.  At initialization this is a stable
        # additive dense-fusion control, while training can still select the
        # source reliability per joint.
        fused = heatmaps + self.source_strength * weighted_sources
        fused = fused.clamp_min(eps)
        return torch.log(fused), {
            "view_weights": weights,
            "reliability_logits": reliability_logits,
            "cross_mean": cross_mean,
            "fused_heatmaps": fused,
        }
