"""Paper-aligned ray augmentations for Geometry-Biased Transformer training.

The GBT paper specifies three geometric training operations but does not
publish the sampling ranges used for its virtual cameras or planar centering
noise.  This module therefore keeps every unpublished range explicit at the
call site while implementing the stated geometry exactly:

* sample sequence-level virtual camera centres and trace rays through GT 3-D
  joints;
* centre the complete sequence at a noisy floor projection of a randomly
  selected neck joint;
* rotate the complete sequence around the vertical (H36M Z) axis.

Rays use the RUMPL layout ``[..., direction(3), point_on_ray(3), confidence]``.
For the H36M configuration used here, ``point_on_ray`` is the camera centre.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GBTSceneTransform:
    """Rigid world-to-centred transform sampled once per sequence."""

    center: torch.Tensor
    rotation: torch.Tensor
    reference_frame: torch.Tensor


def _validate_temporal_inputs(rays: torch.Tensor, target: torch.Tensor) -> None:
    if rays.ndim != 5 or rays.shape[-1] < 7:
        raise ValueError(f"expected rays (B,T,J,V,>=7), got {tuple(rays.shape)}")
    if target.ndim != 4 or target.shape[-1] != 3:
        raise ValueError(f"expected target (B,T,J,3), got {tuple(target.shape)}")
    if rays.shape[:3] != target.shape[:3]:
        raise ValueError("rays and target must share batch/time/joint dimensions")


def append_synthetic_camera_rays(
    rays: torch.Tensor,
    target: torch.Tensor,
    num_synthetic_views: int,
    radius_min_m: float,
    radius_max_m: float,
    height_min_m: float,
    height_max_m: float,
    neck_index: int = 8,
) -> torch.Tensor:
    """Append fixed-per-sequence virtual cameras with perfect GT rays.

    Camera azimuth and radius are sampled around the floor projection of the
    reference subject location.  No camera orientation or intrinsic matrix is
    required because the paper augmentation is defined directly as rays
    passing through the sampled camera positions and GT joints.
    """

    _validate_temporal_inputs(rays, target)
    if num_synthetic_views < 0:
        raise ValueError("num_synthetic_views must be non-negative")
    if num_synthetic_views == 0:
        return rays
    if not 0.0 < radius_min_m <= radius_max_m:
        raise ValueError("invalid synthetic-camera radius range")
    if not 0.0 <= height_min_m <= height_max_m:
        raise ValueError("invalid synthetic-camera height range")
    if not 0 <= neck_index < target.shape[2]:
        raise ValueError("neck_index is outside the target skeleton")

    batch, time, joints, _, channels = rays.shape
    device, dtype = target.device, target.dtype
    # A camera is fixed for all frames in a sequence.  Use the sequence mean
    # neck position only as the centre of the sampling cylinder.
    subject_floor = target[:, :, neck_index].mean(dim=1)
    subject_floor = subject_floor.clone()
    subject_floor[:, 2] = 0.0

    azimuth = torch.rand(batch, num_synthetic_views, device=device, dtype=dtype)
    azimuth = azimuth * (2.0 * math.pi)
    radius = torch.rand(batch, num_synthetic_views, device=device, dtype=dtype)
    radius = radius_min_m + radius * (radius_max_m - radius_min_m)
    height = torch.rand(batch, num_synthetic_views, device=device, dtype=dtype)
    height = height_min_m + height * (height_max_m - height_min_m)
    offsets = torch.stack(
        [radius * azimuth.cos(), radius * azimuth.sin(), height], dim=-1
    )
    camera = subject_floor[:, None, :] + offsets

    camera_expanded = camera[:, None, None, :, :].expand(
        batch, time, joints, num_synthetic_views, 3
    )
    direction = F.normalize(
        target[:, :, :, None, :] - camera_expanded, dim=-1, eps=1e-7
    )
    synthetic = rays.new_zeros(
        batch, time, joints, num_synthetic_views, channels
    )
    synthetic[..., :3] = direction
    synthetic[..., 3:6] = camera_expanded
    synthetic[..., 6] = 1.0
    return torch.cat([rays, synthetic], dim=3)


def sample_scene_transform(
    target: torch.Tensor,
    planar_noise_std_m: float,
    neck_index: int = 8,
) -> GBTSceneTransform:
    """Sample GBT centering and a random Z-axis rotation per sequence."""

    if target.ndim != 4 or target.shape[-1] != 3:
        raise ValueError(f"expected target (B,T,J,3), got {tuple(target.shape)}")
    if planar_noise_std_m < 0:
        raise ValueError("planar_noise_std_m must be non-negative")
    if not 0 <= neck_index < target.shape[2]:
        raise ValueError("neck_index is outside the target skeleton")

    batch, time = target.shape[:2]
    device, dtype = target.device, target.dtype
    reference_frame = torch.randint(time, (batch,), device=device)
    rows = torch.arange(batch, device=device)
    center = target[rows, reference_frame, neck_index].clone()
    center[:, 2] = 0.0
    if planar_noise_std_m > 0:
        center[:, :2] += torch.randn(
            batch, 2, device=device, dtype=dtype
        ) * planar_noise_std_m

    angle = (
        torch.rand(batch, device=device, dtype=dtype) * 2.0 - 1.0
    ) * math.pi
    cosine, sine = angle.cos(), angle.sin()
    rotation = target.new_zeros(batch, 3, 3)
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 1] = -sine
    rotation[:, 1, 0] = sine
    rotation[:, 1, 1] = cosine
    rotation[:, 2, 2] = 1.0
    return GBTSceneTransform(center, rotation, reference_frame)


def _rotate(vector: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Apply batched column-vector rotations to tensors with leading B."""

    return torch.einsum("b...c,bdc->b...d", vector, rotation)


def apply_scene_transform(
    rays: torch.Tensor,
    target: torch.Tensor,
    transform: GBTSceneTransform,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform ray directions, ray points and GT poses consistently."""

    _validate_temporal_inputs(rays, target)
    if transform.center.shape != (rays.shape[0], 3):
        raise ValueError("scene-transform center has the wrong shape")
    if transform.rotation.shape != (rays.shape[0], 3, 3):
        raise ValueError("scene-transform rotation has the wrong shape")

    transformed_rays = rays.clone()
    transformed_rays[..., :3] = _rotate(
        rays[..., :3], transform.rotation
    )
    transformed_rays[..., 3:6] = _rotate(
        rays[..., 3:6] - transform.center[:, None, None, None, :],
        transform.rotation,
    )
    transformed_target = _rotate(
        target - transform.center[:, None, None, :], transform.rotation
    )
    return transformed_rays, transformed_target


def invert_scene_transform(
    points: torch.Tensor,
    transform: GBTSceneTransform,
) -> torch.Tensor:
    """Map predicted centred coordinates back to the original world frame."""

    if points.shape[0] != transform.center.shape[0] or points.shape[-1] != 3:
        raise ValueError("points and scene transform are incompatible")
    restored = torch.einsum(
        "b...c,bcd->b...d", points, transform.rotation
    )
    expand = [points.shape[0]] + [1] * (points.ndim - 2) + [3]
    return restored + transform.center.view(*expand)


def apply_gbt_training_augmentation(
    rays: torch.Tensor,
    target: torch.Tensor,
    *,
    num_synthetic_views: int,
    radius_min_m: float,
    radius_max_m: float,
    height_min_m: float,
    height_max_m: float,
    planar_noise_std_m: float,
    neck_index: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, GBTSceneTransform]:
    """Apply the complete paper-stated geometric training augmentation."""

    rays = append_synthetic_camera_rays(
        rays,
        target,
        num_synthetic_views=num_synthetic_views,
        radius_min_m=radius_min_m,
        radius_max_m=radius_max_m,
        height_min_m=height_min_m,
        height_max_m=height_max_m,
        neck_index=neck_index,
    )
    transform = sample_scene_transform(
        target, planar_noise_std_m=planar_noise_std_m, neck_index=neck_index
    )
    rays, target = apply_scene_transform(rays, target, transform)
    return rays, target, transform
