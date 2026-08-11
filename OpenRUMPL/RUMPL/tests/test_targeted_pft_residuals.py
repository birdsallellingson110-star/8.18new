#!/usr/bin/env python3
"""Identity and shape checks for H81-H83 targeted PFT additions."""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from models.multiview_rumpl import (
    ZeroInitJointSpecificHead,
    ZeroInitSkeletonGraphResidual,
    build_h36m17_adjacency,
)


def test_h36m_graph_is_symmetric_and_has_all_joints():
    adjacency = build_h36m17_adjacency()
    assert adjacency.shape == (17, 17)
    assert torch.allclose(adjacency, adjacency.T)
    assert torch.all(adjacency.diag() > 0)


def test_optional_residual_modules_start_as_exact_identity():
    x = torch.randn(3, 17, 32)
    graph = ZeroInitSkeletonGraphResidual(32)
    head = ZeroInitJointSpecificHead(17, 32)
    assert torch.equal(graph(x), x)
    assert torch.equal(head(x), torch.zeros(3, 17, 3))


if __name__ == "__main__":
    test_h36m_graph_is_symmetric_and_has_all_joints()
    test_optional_residual_modules_start_as_exact_identity()
    print("test_targeted_pft_residuals: PASS")
