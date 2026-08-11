#!/usr/bin/env python3
"""Tests for subject- and joint-anchor ray coordinate frames."""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from models.multiview_rumpl import center_ray_points_on_anchor


def test_subject_center_uses_joint_zero_for_every_joint():
    points = torch.zeros(1, 2, 3, 3)
    anchors = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    centered = center_ray_points_on_anchor(points, anchors, per_joint=False)
    expected = -anchors[:, :1, None, :].expand_as(points)
    assert torch.allclose(centered, expected)


def test_joint_center_is_translation_invariant():
    points = torch.randn(2, 4, 3, 3)
    anchors = torch.randn(2, 4, 3)
    translation = torch.tensor([2.0, -3.0, 5.0])
    original = center_ray_points_on_anchor(points, anchors, per_joint=True)
    translated = center_ray_points_on_anchor(
        points + translation,
        anchors + translation,
        per_joint=True,
    )
    assert torch.allclose(original, translated, atol=1e-6)


if __name__ == "__main__":
    test_subject_center_uses_joint_zero_for_every_joint()
    test_joint_center_is_translation_invariant()
    print("test_anchor_centered_ray_points: PASS")
