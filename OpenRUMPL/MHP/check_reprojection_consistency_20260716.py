#!/usr/bin/env python3
import sys

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

sys.path.insert(0, "/home/lixiaob/cjy/OpenRUMPL/RUMPL/run")
import _init_paths  # noqa: F401,E402
import dataset  # noqa: E402
from core.config import config, update_config  # noqa: E402


def project_world(points_3d, joints_2ds):
    """Project (B,J,3) with packed [xy,K5,Rt12,conf] camera data."""
    camera = joints_2ds[..., 2:19]
    intrinsics = camera[..., :5]
    rt = camera[..., 5:].reshape(*camera.shape[:-1], 3, 4)
    points_h = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)
    points_cam = torch.einsum("bjvrc,bjc->bjvr", rt, points_h)
    z = points_cam[..., 2]
    z_safe = z.clamp_min(1e-6)
    xz = points_cam[..., 0] / z_safe
    yz = points_cam[..., 1] / z_safe
    fx, fy, skew, cx, cy = intrinsics.unbind(dim=-1)
    projected = torch.stack([fx * xz + skew * yz + cx, fy * yz + cy], dim=-1)
    return projected, z


def point_to_ray_distance(points_3d, rays):
    direction = F.normalize(rays[..., :3], dim=-1, eps=1e-8)
    point = rays[..., 3:6]
    offset = points_3d[:, :, None, :] - point
    perpendicular = offset - (offset * direction).sum(dim=-1, keepdim=True) * direction
    return torch.linalg.vector_norm(perpendicular, dim=-1)


def main():
    update_config(
        "/home/lixiaob/cjy/OpenRUMPL/RUMPL/configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"
    )
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval("dataset." + config.DATASET.TRAIN_DATASET)(
        config,
        config.DATASET.TRAIN_SUBSET,
        True,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=False, num_workers=0
    )
    weighted_sum = 0.0
    robust_sum = 0.0
    weight_sum = 0.0
    ray_weighted_sum = 0.0
    ray_weight_sum = 0.0
    valid_depth = 0
    total_depth = 0
    per_batch = []
    for batch_index, (_, _, target, rays, _, joints_2ds) in enumerate(loader):
        projected, depth = project_world(target, joints_2ds)
        observed = joints_2ds[..., :2]
        confidence = joints_2ds[..., 19].clamp(0, 1)
        error = torch.linalg.vector_norm(projected - observed, dim=-1)
        robust = F.smooth_l1_loss(
            projected, observed, reduction="none", beta=0.01
        ).sum(dim=-1)
        finite = torch.isfinite(error) & torch.isfinite(depth) & (depth > 1e-6)
        weights = confidence * finite
        weighted_sum += (error * weights).sum().item()
        robust_sum += (robust * weights).sum().item()
        weight_sum += weights.sum().item()
        ray_distance = point_to_ray_distance(target, rays)
        ray_confidence = rays[..., 6].clamp(0, 1)
        ray_valid = torch.isfinite(ray_distance)
        ray_weights = ray_confidence * ray_valid
        ray_weighted_sum += (ray_distance * ray_weights).sum().item()
        ray_weight_sum += ray_weights.sum().item()
        valid_depth += finite.sum().item()
        total_depth += finite.numel()
        per_batch.append((error * weights).sum().item() / max(weights.sum().item(), 1.0))
        if batch_index == 7:
            break
    print(f"samples={(batch_index + 1) * 16}")
    print(f"confidence_weighted_l2={weighted_sum / weight_sum:.8f}")
    print(f"confidence_weighted_smooth_l1_beta01={robust_sum / weight_sum:.8f}")
    print(f"batch_min={min(per_batch):.8f} batch_max={max(per_batch):.8f}")
    print(f"positive_finite_depth={valid_depth / total_depth:.8f}")
    print(f"confidence_weighted_gt_point_to_ray={ray_weighted_sum / ray_weight_sum:.8f}")


if __name__ == "__main__":
    main()
