"""Checks for the targeted full-skeleton VFT reliability bias."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from models.multiview_rumpl import (  # noqa: E402
    JointViewConditionalResidual,
    JointViewFeatureResidual,
    SkeletonViewReliabilityBias,
    ZeroInitGeometryConditional3DResidual,
    fusion_token_source_attention_bias,
    geometry_joint_view_reliability_logits,
    geometry_view_reliability_logits,
    joint_fusion_token_source_attention_bias,
)


def test_centered_logits_support_variable_view_counts() -> None:
    torch.manual_seed(11)
    module = SkeletonViewReliabilityBias(num_joints=17, dim=32)
    for num_views in (2, 3, 4):
        logits = module(torch.randn(2, 17, num_views, 32))
        assert logits.shape == (2, num_views)
        torch.testing.assert_close(
            logits.sum(dim=1), torch.zeros(2), atol=1e-6, rtol=0
        )


def test_logits_are_view_permutation_equivariant() -> None:
    torch.manual_seed(13)
    module = SkeletonViewReliabilityBias(num_joints=17, dim=16)
    x = torch.randn(2, 17, 4, 16)
    permutation = torch.tensor([2, 0, 3, 1])
    expected = module(x)[:, permutation]
    actual = module(x[:, :, permutation])
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_bias_only_modifies_fusion_query_source_keys() -> None:
    logits = torch.tensor([[1.0, -0.25, -0.75]])
    bias = fusion_token_source_attention_bias(logits, num_joints=2, gate=torch.tensor(2.0))
    assert bias.shape == (2, 4, 4)
    torch.testing.assert_close(bias[:, 0, 1:], 2.0 * logits.expand(2, -1))
    torch.testing.assert_close(bias[:, 0, 0], torch.zeros(2))
    torch.testing.assert_close(bias[:, 1:, :], torch.zeros(2, 3, 4))


def test_zero_gate_is_exact_identity_and_gate_receives_gradient() -> None:
    torch.manual_seed(17)
    module = SkeletonViewReliabilityBias(num_joints=17, dim=16)
    logits = module(torch.randn(2, 17, 4, 16))
    gate = torch.nn.Parameter(torch.zeros(()))
    zero_bias = fusion_token_source_attention_bias(logits, 17, gate)
    torch.testing.assert_close(zero_bias, torch.zeros_like(zero_bias), atol=0, rtol=0)
    target = torch.randn_like(zero_bias)
    (zero_bias * target).sum().backward()
    assert gate.grad is not None and torch.isfinite(gate.grad)


def test_confidence_view_statistic_is_centered_and_variable_view() -> None:
    confidence = torch.tensor(
        [[[[0.2], [0.8], [0.5]], [[0.4], [0.6], [0.4]]]]
    )
    logits = confidence.clamp(0, 1).mean(dim=1).squeeze(-1)
    logits = logits - logits.mean(dim=1, keepdim=True)
    torch.testing.assert_close(
        logits, torch.tensor([[-0.18333334, 0.21666667, -0.03333333]])
    )
    assert logits.shape == (1, 3)


def test_geometry_view_logits_downweight_inconsistent_view() -> None:
    direction = torch.tensor(
        [[[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]]]
    )
    point = torch.tensor(
        [[[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]]]
    )
    logits = geometry_view_reliability_logits(direction, point)
    assert logits.shape == (1, 3)
    # The first two lines are mutually parallel and disagree with the third;
    # the third is the geometrically inconsistent source in this setup.
    assert logits[0, 2] < logits[0, 0]
    torch.testing.assert_close(logits.sum(dim=1), torch.zeros(1), atol=1e-6, rtol=0)


def test_joint_bias_preserves_joint_specific_logits() -> None:
    logits = torch.tensor(
        [[[0.2, 0.5, 0.3], [0.8, 0.1, 0.4]]]
    )
    logits = logits - logits.mean(dim=-1, keepdim=True)
    bias = joint_fusion_token_source_attention_bias(logits, torch.tensor(2.0))
    assert bias.shape == (2, 4, 4)
    torch.testing.assert_close(bias[:, 0, 1:], 2.0 * logits.reshape(2, 3))
    torch.testing.assert_close(bias[:, 1:, :], torch.zeros(2, 3, 4))


def test_joint_geometry_logits_has_per_joint_shape() -> None:
    direction = torch.tensor(
        [[[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]]]
    )
    point = torch.tensor(
        [[[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ], [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
        ]]]
    )
    logits = geometry_joint_view_reliability_logits(direction, point)
    assert logits.shape == (1, 2, 3)
    torch.testing.assert_close(
        logits.sum(dim=-1), torch.zeros(1, 2), atol=1e-6, rtol=0
    )


def test_joint_feature_residual_is_zero_initialized_and_trainable() -> None:
    torch.manual_seed(19)
    module = JointViewFeatureResidual(num_joints=17, dim=8)
    tokens = torch.randn(2, 17, 3, 8)
    scalar = torch.randn(2, 17, 3)
    torch.testing.assert_close(module(tokens, scalar), tokens, atol=0, rtol=0)
    loss = module(tokens, scalar).square().mean()
    loss.backward()
    assert module.weight.grad is not None
    assert torch.isfinite(module.weight.grad).all()


def test_joint_conditional_residual_is_exact_identity_and_trainable() -> None:
    torch.manual_seed(23)
    module = JointViewConditionalResidual(num_joints=17, dim=8, hidden_dim=4)
    tokens = torch.randn(2, 17, 3, 8)
    scalar = torch.randn(2, 17, 3)
    torch.testing.assert_close(module(tokens, scalar), tokens, atol=0, rtol=0)
    loss = module(tokens, scalar).square().mean()
    loss.backward()
    assert module.output.weight.grad is not None
    assert torch.isfinite(module.output.weight.grad).all()
    assert module.input.weight.grad is not None
    # The first step is intentionally gated by the zero output projection;
    # later optimization steps become active once the output learns.
    torch.testing.assert_close(
        module.output.weight.detach(),
        torch.zeros_like(module.output.weight),
        atol=0,
        rtol=0,
    )


def test_geometry_conditioned_output_residual_is_zero_initialized() -> None:
    torch.manual_seed(29)
    module = ZeroInitGeometryConditional3DResidual(
        num_joints=17, dim=8, condition_dim=4, hidden_dim=4
    )
    features = torch.randn(2, 17, 8)
    condition = torch.randn(2, 17, 4)
    correction = module(features, condition)
    torch.testing.assert_close(correction, torch.zeros_like(correction), atol=0, rtol=0)
    correction.square().mean().backward()
    assert module.output.weight.grad is not None
    assert torch.isfinite(module.output.weight.grad).all()


if __name__ == "__main__":
    test_centered_logits_support_variable_view_counts()
    test_logits_are_view_permutation_equivariant()
    test_bias_only_modifies_fusion_query_source_keys()
    test_zero_gate_is_exact_identity_and_gate_receives_gradient()
    test_confidence_view_statistic_is_centered_and_variable_view()
    test_geometry_view_logits_downweight_inconsistent_view()
    test_joint_bias_preserves_joint_specific_logits()
    test_joint_geometry_logits_has_per_joint_shape()
    test_joint_feature_residual_is_zero_initialized_and_trainable()
    test_joint_conditional_residual_is_exact_identity_and_trainable()
    test_geometry_conditioned_output_residual_is_zero_initialized()
    print("Skeleton view reliability tests passed")
