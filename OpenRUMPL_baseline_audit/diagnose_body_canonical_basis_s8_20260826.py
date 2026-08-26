#!/usr/bin/env python3
"""Compare current and robust torso frames on an S8 ray cache."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def anchors_from_rays(rays: torch.Tensor, regularization: float = 1e-2):
    direction = F.normalize(rays[..., :3], dim=-1, eps=1e-7)
    point = rays[..., 3:6]
    confidence = rays[..., 6:7].clamp(0, 1) + 0.05
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    weighted_projection = confidence.unsqueeze(-1) * projection
    lhs = weighted_projection.sum(dim=2)
    rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
    centroid = (
        (confidence * point).sum(dim=2)
        / confidence.sum(dim=2).clamp_min(1e-7)
    )
    pelvis = torch.linalg.solve(
        lhs[:, 0] + regularization * eye,
        rhs[:, 0] + regularization * centroid[:, 0].unsqueeze(-1),
    ).squeeze(-1)
    anchors = torch.linalg.solve(
        lhs + regularization * eye,
        rhs + regularization * pelvis[:, None, :, None],
    ).squeeze(-1)
    return anchors, confidence.squeeze(-1).mean(dim=2)


def basis_current(anchors: torch.Tensor) -> torch.Tensor:
    x_axis = F.normalize(anchors[:, 14] - anchors[:, 11], dim=-1, eps=1e-7)
    up_hint = anchors[:, 8] - anchors[:, 0]
    y_axis = F.normalize(
        up_hint - (up_hint * x_axis).sum(-1, keepdim=True) * x_axis,
        dim=-1,
        eps=1e-7,
    )
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-7)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-7)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def basis_shoulder_mid(anchors: torch.Tensor) -> torch.Tensor:
    """Keep the shoulder x-axis but replace the virtual-neck up hint."""
    x_axis = F.normalize(anchors[:, 14] - anchors[:, 11], dim=-1, eps=1e-7)
    shoulder_mid = 0.5 * (anchors[:, 11] + anchors[:, 14])
    up_hint = shoulder_mid - anchors[:, 0]
    y_axis = F.normalize(
        up_hint - (up_hint * x_axis).sum(-1, keepdim=True) * x_axis,
        dim=-1,
        eps=1e-7,
    )
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-7)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-7)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def basis_robust(anchors: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
    shoulder = F.normalize(anchors[:, 14] - anchors[:, 11], dim=-1, eps=1e-7)
    # Align dynamically because historical detector PKLs did not all use the
    # same lower-body left/right convention.
    hip = F.normalize(anchors[:, 4] - anchors[:, 1], dim=-1, eps=1e-7)
    hip = hip * torch.where(
        (hip * shoulder).sum(dim=-1, keepdim=True) < 0,
        -torch.ones_like(hip[..., :1]),
        torch.ones_like(hip[..., :1]),
    )
    shoulder_weight = torch.sqrt(confidence[:, 14] * confidence[:, 11]).unsqueeze(-1)
    hip_weight = torch.sqrt(confidence[:, 1] * confidence[:, 4]).unsqueeze(-1)
    x_axis = F.normalize(
        shoulder_weight * shoulder + hip_weight * hip, dim=-1, eps=1e-7
    )
    shoulder_mid = 0.5 * (anchors[:, 11] + anchors[:, 14])
    hip_mid = 0.5 * (anchors[:, 1] + anchors[:, 4])
    up_hint = shoulder_mid - hip_mid
    y_axis = F.normalize(
        up_hint - (up_hint * x_axis).sum(-1, keepdim=True) * x_axis,
        dim=-1,
        eps=1e-7,
    )
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-7)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-7)
    return torch.stack((x_axis, y_axis, z_axis), dim=-1)


def angular_errors(estimated: torch.Tensor, target: torch.Tensor):
    per_axis = torch.rad2deg(torch.acos(torch.clamp(
        (estimated * target).sum(dim=1), -1.0, 1.0
    )))
    relative = estimated.transpose(1, 2) @ target
    trace = relative.diagonal(dim1=1, dim2=2).sum(dim=1)
    geodesic = torch.rad2deg(torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0)))
    return per_axis, geodesic


def summarize(values: np.ndarray) -> dict:
    return {
        "mean_deg": float(values.mean()),
        "median_deg": float(np.median(values)),
        "p90_deg": float(np.percentile(values, 90)),
    }


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache).resolve()
    arrays = np.load(cache_path, mmap_mode="r")
    rays = arrays["rays"]
    targets = arrays["targets"]
    subjects = sorted(set(map(int, arrays["subjects"].tolist())))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    rows = {}
    for cardinality in (2, 3, 4):
        current_axes, shoulder_mid_axes, robust_axes = [], [], []
        current_geo, shoulder_mid_geo, robust_geo = [], [], []
        for combo in itertools.combinations(range(4), cardinality):
            for start in range(0, len(rays), args.batch_size):
                stop = min(start + args.batch_size, len(rays))
                batch_rays = torch.from_numpy(np.asarray(
                    rays[start:stop, :, list(combo), :]
                )).to(device=device, dtype=torch.float32)
                target = torch.from_numpy(np.asarray(targets[start:stop])).to(
                    device=device, dtype=torch.float32
                )
                anchors, confidence = anchors_from_rays(batch_rays)
                current = basis_current(anchors)
                shoulder_mid = basis_shoulder_mid(anchors)
                robust = basis_robust(anchors, confidence)
                target_confidence = torch.ones(
                    len(target), 17, device=device, dtype=torch.float32
                )
                reference = basis_robust(target, target_confidence)
                axis, geo = angular_errors(current, reference)
                current_axes.append(axis.cpu().numpy())
                current_geo.append(geo.cpu().numpy())
                axis, geo = angular_errors(shoulder_mid, reference)
                shoulder_mid_axes.append(axis.cpu().numpy())
                shoulder_mid_geo.append(geo.cpu().numpy())
                axis, geo = angular_errors(robust, reference)
                robust_axes.append(axis.cpu().numpy())
                robust_geo.append(geo.cpu().numpy())
        current_axes_np = np.concatenate(current_axes)
        shoulder_mid_axes_np = np.concatenate(shoulder_mid_axes)
        robust_axes_np = np.concatenate(robust_axes)
        rows[f"V{cardinality}"] = {
            "current_shoulder_neck": {
                "x": summarize(current_axes_np[:, 0]),
                "y": summarize(current_axes_np[:, 1]),
                "z": summarize(current_axes_np[:, 2]),
                "geodesic": summarize(np.concatenate(current_geo)),
            },
            "shoulder_mid_pelvis": {
                "x": summarize(shoulder_mid_axes_np[:, 0]),
                "y": summarize(shoulder_mid_axes_np[:, 1]),
                "z": summarize(shoulder_mid_axes_np[:, 2]),
                "geodesic": summarize(np.concatenate(shoulder_mid_geo)),
            },
            "robust_shoulder_hip": {
                "x": summarize(robust_axes_np[:, 0]),
                "y": summarize(robust_axes_np[:, 1]),
                "z": summarize(robust_axes_np[:, 2]),
                "geodesic": summarize(np.concatenate(robust_geo)),
            },
        }
    payload = {
        "protocol": "clean H36M S8 only; all V2/V3/V4 camera combinations",
        "subjects": subjects,
        "cache": str(cache_path),
        "groups": int(len(rays)),
        "reference": "GT shoulder+hip robust torso frame",
        "results": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
