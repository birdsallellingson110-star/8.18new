#!/usr/bin/env python3
"""Tests for analytic ray-normal-matrix uncertainty features."""

import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from models.multiview_rumpl import ray_normal_matrix_features


def make_pair(angle_degrees):
    angle = torch.tensor(angle_degrees * torch.pi / 180.0)
    first = torch.tensor([0.0, 0.0, 1.0])
    second = torch.stack((torch.sin(angle), torch.tensor(0.0), torch.cos(angle)))
    return torch.stack((first, second)).reshape(1, 1, 2, 3)


def test_parallel_pair_has_smaller_minimum_eigen_fraction():
    weak = ray_normal_matrix_features(make_pair(8.0))
    strong = ray_normal_matrix_features(make_pair(45.0))
    assert weak.shape == (1, 1, 4)
    assert weak[..., 0].item() < strong[..., 0].item()


def test_features_are_rotation_invariant_and_confidence_sensitive():
    rays = make_pair(35.0)
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = rays @ rotation.T
    confidence = torch.tensor([[[[0.8], [0.4]]]])
    original = ray_normal_matrix_features(rays, confidence)
    transformed = ray_normal_matrix_features(rotated, confidence)
    assert torch.allclose(original, transformed, atol=1e-6)
    assert torch.allclose(original[..., 3], torch.tensor([[0.6]]))


if __name__ == "__main__":
    test_parallel_pair_has_smaller_minimum_eigen_fraction()
    test_features_are_rotation_invariant_and_confidence_sensitive()
    print("test_geometry_uncertainty_token: PASS")
