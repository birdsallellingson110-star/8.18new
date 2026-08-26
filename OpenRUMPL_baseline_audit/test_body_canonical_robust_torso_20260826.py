#!/usr/bin/env python3
"""Regression and SE(3)-equivariance checks for robust torso canonicalization."""

import torch
import torch.nn.functional as F

from models.multiview_rumpl import equivariant_body_canonicalize_rays


def established_path(rays, regularization=1e-2, confidence_epsilon=0.05):
    direction = F.normalize(rays[..., :3], dim=-1, eps=1e-7)
    point = rays[..., 3:6]
    confidence = rays[..., 6:7].clamp(0, 1) + confidence_epsilon
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    weighted_projection = confidence.unsqueeze(-1) * projection
    lhs = weighted_projection.sum(dim=2)
    rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
    centroid = (confidence * point).sum(dim=2) / confidence.sum(dim=2)
    pelvis = torch.linalg.solve(
        lhs[:, 0] + regularization * eye,
        rhs[:, 0] + regularization * centroid[:, 0].unsqueeze(-1),
    ).squeeze(-1)
    anchors = torch.linalg.solve(
        lhs + regularization * eye,
        rhs + regularization * pelvis[:, None, :, None],
    ).squeeze(-1)
    origin = anchors[:, 0]
    x_axis = F.normalize(anchors[:, 14] - anchors[:, 11], dim=-1, eps=1e-7)
    up_hint = anchors[:, 8] - origin
    y_axis = F.normalize(
        up_hint - (up_hint * x_axis).sum(-1, keepdim=True) * x_axis,
        dim=-1,
        eps=1e-7,
    )
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-7)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-7)
    basis = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    canonical = rays.clone()
    canonical[..., :3] = torch.einsum('b...i,bij->b...j', rays[..., :3], basis)
    canonical[..., 3:6] = torch.einsum(
        'b...i,bij->b...j', rays[..., 3:6] - origin[:, None, None, :], basis
    )
    return canonical, origin, basis


def make_rays(dtype=torch.float64):
    torch.manual_seed(260826)
    anchors = torch.randn(5, 17, 3, dtype=dtype)
    anchors[:, 0] = torch.tensor([0.0, 0.0, 0.0], dtype=dtype)
    anchors[:, 1] = torch.tensor([-0.15, 0.0, 0.0], dtype=dtype)
    anchors[:, 4] = torch.tensor([0.15, 0.0, 0.0], dtype=dtype)
    anchors[:, 8] = torch.tensor([0.0, 0.55, 0.0], dtype=dtype)
    anchors[:, 11] = torch.tensor([-0.25, 0.48, 0.0], dtype=dtype)
    anchors[:, 14] = torch.tensor([0.25, 0.48, 0.0], dtype=dtype)
    cameras = torch.tensor(
        [[3.0, 1.4, 0.0], [0.0, 1.6, 3.5], [-3.2, 1.3, 0.4], [0.3, 1.8, -3.3]],
        dtype=dtype,
    )
    points = cameras[None, None].expand(5, 17, 4, 3).clone()
    direction = F.normalize(anchors[:, :, None] - points, dim=-1)
    confidence = 0.2 + 0.8 * torch.rand(5, 17, 4, 1, dtype=dtype)
    return torch.cat((direction, points, confidence), dim=-1)


def main():
    rays = make_rays()
    expected = established_path(rays)
    actual = equivariant_body_canonicalize_rays(
        rays, regularization=1e-2, pelvis_prior=True, robust_torso=False
    )
    default_diffs = [float((a - b).abs().max()) for a, b in zip(actual, expected)]
    assert default_diffs == [0.0, 0.0, 0.0], default_diffs

    angle = torch.tensor(0.73, dtype=rays.dtype)
    axis = F.normalize(torch.tensor([0.4, -0.2, 0.7], dtype=rays.dtype), dim=0)
    skew = torch.tensor(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=rays.dtype,
    )
    rotation = (
        torch.eye(3, dtype=rays.dtype) * torch.cos(angle)
        + (1 - torch.cos(angle)) * axis[:, None] * axis[None, :]
        + torch.sin(angle) * skew
    )
    translation = torch.tensor([1.7, -0.8, 2.2], dtype=rays.dtype)
    transformed = rays.clone()
    transformed[..., :3] = rays[..., :3] @ rotation.T
    transformed[..., 3:6] = rays[..., 3:6] @ rotation.T + translation
    canonical, origin, basis = equivariant_body_canonicalize_rays(
        rays, regularization=1e-2, pelvis_prior=True, robust_torso=True
    )
    canonical_t, origin_t, basis_t = equivariant_body_canonicalize_rays(
        transformed, regularization=1e-2, pelvis_prior=True, robust_torso=True
    )
    equivariance_diffs = {
        "canonical": float((canonical_t - canonical).abs().max()),
        "origin": float((origin_t - (origin @ rotation.T + translation)).abs().max()),
        "basis": float((basis_t - torch.einsum('ij,bjk->bik', rotation, basis)).abs().max()),
    }
    assert max(equivariance_diffs.values()) < 1e-10, equivariance_diffs
    print({"default_exact_max_abs": default_diffs, "se3_max_abs": equivariance_diffs})


if __name__ == "__main__":
    main()
