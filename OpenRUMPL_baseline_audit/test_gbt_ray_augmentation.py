import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))

from dataset.gbt_ray_augmentation import (
    apply_gbt_training_augmentation,
    invert_scene_transform,
)


def make_sequence():
    torch.manual_seed(4)
    target = torch.randn(2, 5, 17, 3)
    target[..., 2] += 1.2
    camera = torch.tensor(
        [[[-4.0, 0.0, 1.5], [4.0, 0.0, 1.5]]]
    ).expand(2, -1, -1)
    camera = camera[:, None, None].expand(2, 5, 17, 2, 3)
    direction = torch.nn.functional.normalize(
        target[..., None, :] - camera, dim=-1
    )
    confidence = torch.ones(2, 5, 17, 2, 1)
    rays = torch.cat([direction, camera, confidence], dim=-1)
    return rays, target


def test_synthetic_rays_and_scene_transform_are_geometrically_consistent():
    rays, target = make_sequence()
    torch.manual_seed(8)
    augmented_rays, augmented_target, transform = apply_gbt_training_augmentation(
        rays,
        target,
        num_synthetic_views=2,
        radius_min_m=3.0,
        radius_max_m=6.0,
        height_min_m=1.0,
        height_max_m=2.5,
        planar_noise_std_m=0.0,
    )
    assert augmented_rays.shape == (2, 5, 17, 4, 7)
    assert torch.all(augmented_rays[..., 2:, 6] == 1)

    # Every synthetic line must pass exactly through its transformed GT joint.
    point = augmented_rays[..., 2:, 3:6]
    direction = augmented_rays[..., 2:, :3]
    displacement = augmented_target[..., None, :] - point
    orthogonal = torch.cross(displacement, direction, dim=-1)
    assert orthogonal.norm(dim=-1).max() < 2e-5

    # Synthetic cameras are fixed for the complete temporal sequence.
    assert torch.allclose(point[:, :1, :1], point[:, :, :, :], atol=1e-6)

    restored = invert_scene_transform(augmented_target, transform)
    assert torch.allclose(restored, target, atol=2e-6)

    # With zero planar noise the selected neck lies on the transformed Z axis.
    rows = torch.arange(target.shape[0])
    selected_neck = augmented_target[rows, transform.reference_frame, 8]
    assert selected_neck[:, :2].abs().max() < 2e-6


def test_zero_synthetic_views_preserves_view_count():
    rays, target = make_sequence()
    torch.manual_seed(9)
    augmented_rays, augmented_target, transform = apply_gbt_training_augmentation(
        rays,
        target,
        num_synthetic_views=0,
        radius_min_m=3.0,
        radius_max_m=6.0,
        height_min_m=1.0,
        height_max_m=2.5,
        planar_noise_std_m=0.25,
    )
    assert augmented_rays.shape == rays.shape
    assert torch.allclose(
        invert_scene_transform(augmented_target, transform), target, atol=2e-6
    )


def main():
    test_synthetic_rays_and_scene_transform_are_geometrically_consistent()
    test_zero_synthetic_views_preserves_view_count()
    print("GBT ray augmentation tests passed")


if __name__ == "__main__":
    main()
