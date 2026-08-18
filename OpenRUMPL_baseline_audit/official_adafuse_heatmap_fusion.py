#!/usr/bin/env python3
"""AdaFuse-style view weighting for frozen H36M heatmaps.

This is a coordinate/heatmap-only port of the public AdaFuse ``ViewWeightNet``
path.  The detector is frozen.  A shared network predicts one reliability
weight for every joint and view from the joint heatmap and calibrated
cross-view Sampson distances.  The weighted current/source heatmaps are then
decoded by the existing evaluator.

The public implementation also consumes a 256-channel backbone feature.  Our
audit deliberately omits that feature because the exported input contract is
HRNet/LT heatmaps only; this keeps the comparison explicit and camera-ID free.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def sampson_features_from_cameras(
    image_xy: np.ndarray,
    confidence: np.ndarray,
    intrinsics: list[np.ndarray],
    rotations: list[np.ndarray],
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute AdaFuse's per-view Sampson distance/confidence descriptors.

    ``image_xy`` is ``V x J x 2`` in the H36M semantic order.  The returned
    arrays are ``V x J x (V-1)`` with the other-view index in ascending order.
    The fundamental matrix maps the source view's point to the current view's
    epipolar line, matching the public implementation.
    """
    image_xy = np.asarray(image_xy, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    n_views, n_joints = image_xy.shape[:2]
    distances = np.zeros((n_views, n_joints, n_views - 1), dtype=np.float32)
    confidences = np.zeros_like(distances)
    homogeneous = np.concatenate(
        (image_xy, np.ones((n_views, n_joints, 1), dtype=np.float64)), axis=-1
    )
    for target in range(n_views):
        kt = np.asarray(intrinsics[target], dtype=np.float64)
        rt = np.asarray(rotations[target], dtype=np.float64)
        for slot, source in enumerate(
            source for source in range(n_views) if source != target
        ):
            ks = np.asarray(intrinsics[source], dtype=np.float64)
            rs = np.asarray(rotations[source], dtype=np.float64)
            # x_target = K_t R_t (R_s^T K_s^-1 x_source + C_s-C_t).
            relative_r = rt @ rs.T
            relative_t = rt @ (
                np.asarray(centers[source]) - np.asarray(centers[target])
            )
            tx, ty, tz = relative_t
            skew = np.asarray(
                [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]],
                dtype=np.float64,
            )
            fundamental = np.linalg.inv(kt).T @ skew @ relative_r @ np.linalg.inv(ks)
            x_target = homogeneous[target]
            x_source = homogeneous[source]
            line_target = (fundamental @ x_source.T).T
            line_source = (fundamental.T @ x_target.T).T
            numerator = np.abs(np.sum(x_target * line_target, axis=-1))
            denominator = np.sqrt(
                line_target[:, 0] ** 2
                + line_target[:, 1] ** 2
                + line_source[:, 0] ** 2
                + line_source[:, 1] ** 2
            )
            distances[target, :, slot] = (
                numerator / np.maximum(denominator, 1e-9)
            ).astype(np.float32)
            # AdaFuse repeats the current-view confidence for all source pairs.
            confidences[target, :, slot] = confidence[target].astype(np.float32)
    return distances, confidences


class OfficialAdaFuseHeatmapFusion(nn.Module):
    """Permutation-equivariant ViewWeightNet + epipolar heatmap fusion."""

    def __init__(
        self, heatmap_channels: int = 17, signed_heatmaps: bool = False
    ) -> None:
        super().__init__()
        self.heatmap_channels = int(heatmap_channels)
        self.signed_heatmaps = bool(signed_heatmaps)
        # Same channel widths and pooling pattern as the public AdaFuse
        # ViewWeightNet.  Each joint heatmap is encoded independently.
        self.heatmap_feature_net = nn.Sequential(
            nn.Conv2d(1, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )
        self.joint_feature_net = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dist_feature_net = nn.Sequential(
            nn.Conv1d(2, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.conf_out = nn.Sequential(
            nn.Linear(128 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _normalize_maps(
        heatmaps: torch.Tensor, signed_heatmaps: bool = False
    ) -> torch.Tensor:
        if not signed_heatmaps:
            heatmaps = heatmaps.clamp_min(0.0)
        peak = heatmaps.flatten(-2).amax(dim=-1, keepdim=True)
        return heatmaps / peak.clamp_min(1e-6)[..., None]

    def predict_view_weights(
        self,
        heatmaps: torch.Tensor,
        distances: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``V x J`` reliability weights.

        ``distances`` and ``confidences`` are target-view first and have shape
        ``V x J x (V-1)``.  This is the same ordering used in the public
        implementation and permits arbitrary view permutations.
        """
        n_views, n_joints, height, width = heatmaps.shape
        if distances.shape != (n_views, n_joints, n_views - 1):
            raise ValueError(
                f"distances must be VxJx(V-1), got {distances.shape}"
            )
        if confidences.shape != distances.shape:
            raise ValueError("confidences must have the same shape as distances")

        # V*J independent heatmap descriptors -> V x J x 128.
        maps = heatmaps.reshape(n_views * n_joints, 1, height, width)
        map_features = self.heatmap_feature_net(maps)
        map_features = F.adaptive_avg_pool2d(map_features, 1)
        map_features = self.joint_feature_net(map_features)
        map_features = map_features.flatten(1).reshape(n_views, n_joints, 128)

        # The public network applies exp(-distance), concatenates confidence,
        # and pools over the other views.  Clamp only to avoid NaN/Inf for a
        # malformed calibration; normal Sampson distances are left untouched.
        dist = torch.exp(-distances.clamp(min=0.0, max=50.0))
        conf = confidences.clamp(0.0, 1.0)
        dist_input = torch.stack((dist, conf), dim=2)
        dist_input = dist_input.reshape(n_views * n_joints, 2, n_views - 1)
        dist_features = self.dist_feature_net(dist_input).mean(dim=2)
        dist_features = dist_features.reshape(n_views, n_joints, 256)

        logits = self.conf_out(
            torch.cat((map_features, dist_features), dim=-1)
        )[..., 0]
        return logits

    def forward(
        self,
        heatmaps: torch.Tensor,
        pairwise_support: torch.Tensor,
        *,
        distances: torch.Tensor | None = None,
        confidences: torch.Tensor | None = None,
        joint_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del joint_ids  # semantic IDs are not used by the official H36M net
        if heatmaps.ndim != 4:
            raise ValueError(f"expected VxJxHxW heatmaps, got {heatmaps.shape}")
        n_views, n_joints, height, width = heatmaps.shape
        expected = (n_views, n_views, n_joints, height, width)
        if tuple(pairwise_support.shape) != expected:
            raise ValueError(
                f"expected pairwise support {expected}, got {pairwise_support.shape}"
            )
        if n_views < 2:
            raise ValueError("AdaFuse fusion requires at least two views")
        if distances is None:
            # Fallback for diagnostics that do not provide calibrated
            # Sampson features.  Agreement-derived proxies preserve the
            # architecture while making the call safe and deterministic.
            support = (
                pairwise_support
                if self.signed_heatmaps
                else pairwise_support.clamp_min(0.0)
            )
            eye = torch.eye(n_views, dtype=torch.bool, device=heatmaps.device)
            support = support.masked_fill(eye[:, :, None, None, None], 0.0)
            cross = support.sum(dim=1) / float(n_views - 1)
            agreement = (heatmaps * cross).flatten(-2).mean(-1)
            distances = (1.0 - agreement).clamp_min(0.0)[..., None].expand(
                n_views, n_joints, n_views - 1
            )
        if confidences is None:
            peak = heatmaps.flatten(-2).amax(-1)
            confidences = peak[..., None].expand(n_views, n_joints, n_views - 1)

        normalized = self._normalize_maps(
            heatmaps, signed_heatmaps=self.signed_heatmaps
        )
        support = (
            pairwise_support
            if self.signed_heatmaps
            else pairwise_support.clamp_min(0.0)
        )
        eye = torch.eye(n_views, dtype=torch.bool, device=heatmaps.device)
        support = support.masked_fill(eye[:, :, None, None, None], 0.0)
        weights = self.predict_view_weights(normalized, distances, confidences)

        # For every target view, concatenate its own map and the aligned maps
        # from all source views.  Use the source view's learned weight, exactly
        # like AdaFuse's get_warp_weight().
        fused = normalized.new_zeros((n_views, n_joints, height, width))
        for target in range(n_views):
            sources = [target] + [source for source in range(n_views) if source != target]
            maps = torch.cat(
                [normalized[target : target + 1], support[target, sources[1:]]],
                dim=0,
            )
            fused[target] = (
                maps * weights[sources, :, None, None]
            ).sum(dim=0)
        if self.signed_heatmaps:
            output = fused
        else:
            output = torch.log(fused.clamp_min(1e-6))
        return output, {
            "fused_heatmaps": fused,
            "view_weights": weights,
            "distances": distances,
            "confidences": confidences,
        }
