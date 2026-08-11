"""Focused checks for the paper-faithful learnable GBT attention path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from models.multiview_rumpl import (  # noqa: E402
    Attention,
    geometry_distance_with_fusion_token,
    pairwise_ray_distance,
)


def test_pairwise_ray_distance() -> None:
    # x-axis through origin, parallel x-axis at y=1, and y-axis at z=2.
    direction = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    point = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]])
    distance = pairwise_ray_distance(direction, point)
    torch.testing.assert_close(distance[0, 0, 1], torch.tensor(1.0))
    torch.testing.assert_close(distance[0, 0, 2], torch.tensor(2.0))
    torch.testing.assert_close(distance, distance.transpose(1, 2))
    torch.testing.assert_close(distance.diagonal(dim1=1, dim2=2), torch.zeros(1, 3))


def test_relative_view_fusion_is_identity_initialized_and_view_equivariant() -> None:
    from models.multiview_rumpl import RelativeViewFusion

    torch.manual_seed(7)
    module = RelativeViewFusion(num_joints=17, dim=32)
    x = torch.randn(2, 17, 4, 32)
    torch.testing.assert_close(module(x), x, rtol=0, atol=0)

    with torch.no_grad():
        module.gate.fill_(1.0)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    output = module(x)
    permuted_output = module(x[:, :, permutation])[:, :, inverse]
    torch.testing.assert_close(output, permuted_output, rtol=1e-5, atol=1e-6)
    output.square().mean().backward()
    assert module.gate.grad is not None
    assert torch.isfinite(module.gate.grad).all()


def test_fusion_token_receives_per_view_geometry_inconsistency() -> None:
    direction = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    point = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]])
    distance = geometry_distance_with_fusion_token(direction, point, direct_fusion=True)
    view_distance = pairwise_ray_distance(direction, point)
    torch.testing.assert_close(distance[:, 1:, 1:], view_distance)
    torch.testing.assert_close(distance[:, 0, 1:], view_distance.sum(dim=-1) / 2)
    torch.testing.assert_close(distance[:, 1:, 0], torch.zeros(1, 3))


def test_confidence_bias_direction_and_gradient() -> None:
    attention = Attention(
        dim=4, num_heads=1, qkv_bias=False, learnable_conf_bias=True, conf_bias_init=1.0
    )
    torch.nn.init.zeros_(attention.qkv.weight)
    os.environ["GBT_SAVE_ATTN"] = "1"
    x = torch.zeros(1, 3, 4)
    key_confidence = torch.tensor([0.0, 0.2, 0.9])
    confidence = key_confidence[None, None, :].expand(1, 3, 3)
    attention(x, conf_bias=confidence).sum().backward()
    weights = attention.last_attn[0, 0, 0]
    assert weights[2] > weights[1] > weights[0]
    assert attention.gbt_conf_scale.grad is not None


def test_geometry_bias_direction_and_gradient() -> None:
    attention = Attention(
        dim=4, num_heads=1, qkv_bias=False, learnable_geom_bias=True, geom_bias_init=1.0
    )
    torch.nn.init.zeros_(attention.qkv.weight)
    os.environ["GBT_SAVE_ATTN"] = "1"
    x = torch.zeros(1, 3, 4)
    distance = torch.tensor([[[0.0, 0.1, 2.0], [0.1, 0.0, 1.0], [2.0, 1.0, 0.0]]])
    attention(x, geom_distance=distance).sum().backward()
    weights = attention.last_attn[0, 0, 0]
    assert weights[0] > weights[1] > weights[2]
    assert attention.gbt_geom_scale.grad is not None


def test_rumpl_uses_one_bias_model_for_all_view_counts() -> None:
    from core.config import config, update_config
    from models.multiview_rumpl import get_multiview_rumpl_net

    os.environ.update(
        {
            "GBT_LEARNABLE_BIAS": "1",
            "GBT_USE_CONF_BIAS": "1",
            "GBT_USE_GEOM_BIAS": "1",
            "GBT_CONF_INIT": "0.1",
            "GBT_GEOM_INIT": "0.1",
            "GBT_CONF_BIAS": "0",
            "GBT_GEOM_BIAS": "0",
            "GBT_VIEW_AWARE": "0",
        }
    )
    update_config(str(ROOT / "configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"))
    model = get_multiview_rumpl_net(config, is_train=True)
    conf_parameters = [
        block.attn.gbt_conf_scale for block in model.features.blocks_view_fusion
    ]
    geom_parameters = [
        block.attn.gbt_geom_scale for block in model.features.blocks_view_fusion
    ]
    assert len(conf_parameters) == config.NETWORK.TRANSFORMER_DEPTH
    assert len(geom_parameters) == config.NETWORK.TRANSFORMER_DEPTH
    outputs = []
    for num_views in (2, 3, 4):
        direction = torch.randn(1, 17, num_views, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        point = torch.randn(1, 17, num_views, 3)
        confidence = torch.rand(1, 17, num_views, 1)
        rays = torch.cat([direction, point, confidence], dim=-1)
        output = model(rays, is_training=False)
        assert output.shape == (1, 17, 3)
        outputs.append(output)
    sum(output.square().mean() for output in outputs).backward()
    assert all(parameter.grad is not None for parameter in conf_parameters)
    assert all(parameter.grad is not None for parameter in geom_parameters)
    assert all(torch.isfinite(parameter.grad) for parameter in conf_parameters)
    assert all(torch.isfinite(parameter.grad) for parameter in geom_parameters)


def test_fixed_two_view_training_keeps_five_view_inference_capacity() -> None:
    from core.config import config, update_config
    from models.multiview_rumpl import get_multiview_rumpl_net

    os.environ.update(
        {
            "TRAIN_FIXED_NUM_VIEWS": "2",
            "GBT_LEARNABLE_BIAS": "1",
            "GBT_USE_CONF_BIAS": "1",
            "GBT_USE_GEOM_BIAS": "1",
            "GBT_FUSION_GEOM": "0",
        }
    )
    update_config(str(ROOT / "configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"))
    model = get_multiview_rumpl_net(config, is_train=True)
    direction = torch.randn(1, 17, 5, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    point = torch.randn(1, 17, 5, 3)
    confidence = torch.rand(1, 17, 5, 1)
    rays = torch.cat([direction, point, confidence], dim=-1)
    assert model(rays, is_training=True).shape == (1, 17, 3)
    assert model(rays, is_training=False).shape == (1, 17, 3)
    os.environ.pop("TRAIN_FIXED_NUM_VIEWS", None)


def test_token_dropout_supports_training_and_multiview_inference() -> None:
    from core.config import config, update_config
    from models.multiview_rumpl import get_multiview_rumpl_net

    os.environ.update(
        {
            "TRAIN_FIXED_NUM_VIEWS": "2",
            "GBT_TOKEN_DROPOUT": "0.2",
            "GBT_TOKEN_DROPOUT_EPOCHS": "5",
            "GBT_LEARNABLE_BIAS": "1",
            "GBT_USE_CONF_BIAS": "1",
            "GBT_USE_GEOM_BIAS": "1",
            "GBT_FUSION_GEOM": "0",
        }
    )
    update_config(str(ROOT / "configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"))
    model = get_multiview_rumpl_net(config, is_train=True)
    direction = torch.randn(2, 17, 5, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    point = torch.randn(2, 17, 5, 3)
    confidence = torch.rand(2, 17, 5, 1)
    rays = torch.cat([direction, point, confidence], dim=-1)
    train_output = model(rays, is_training=True, epoch=0)
    post_schedule_output = model(rays, is_training=True, epoch=5)
    inference_output = model(rays, is_training=False)
    assert train_output.shape == (2, 17, 3)
    assert post_schedule_output.shape == (2, 17, 3)
    assert inference_output.shape == (2, 17, 3)
    assert torch.isfinite(train_output).all()
    assert torch.isfinite(post_schedule_output).all()
    assert torch.isfinite(inference_output).all()
    os.environ.pop("TRAIN_FIXED_NUM_VIEWS", None)
    os.environ.pop("GBT_TOKEN_DROPOUT", None)
    os.environ.pop("GBT_TOKEN_DROPOUT_EPOCHS", None)


def test_global_joint_view_refinement_supports_variable_view_counts() -> None:
    from core.config import config, update_config
    from models.multiview_rumpl import get_multiview_rumpl_net

    os.environ.update(
        {
            "TRAIN_FIXED_NUM_VIEWS": "2",
            "GBT_TOKEN_DROPOUT": "0",
            "GBT_GLOBAL_JV_DEPTH": "1",
            "GBT_GLOBAL_JV_BIASED": "1",
            "GBT_GLOBAL_JV_GATED": "1",
            "GBT_LEARNABLE_BIAS": "1",
            "GBT_USE_CONF_BIAS": "1",
            "GBT_USE_GEOM_BIAS": "1",
            "GBT_FUSION_GEOM": "0",
        }
    )
    update_config(str(ROOT / "configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"))
    model = get_multiview_rumpl_net(config, is_train=True)
    torch.testing.assert_close(model.features.global_jv_gate, torch.zeros(1))
    outputs = []
    for num_views, is_training in ((2, True), (5, False)):
        direction = torch.randn(1, 17, num_views, 3)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        point = torch.randn(1, 17, num_views, 3)
        confidence = torch.rand(1, 17, num_views, 1)
        rays = torch.cat([direction, point, confidence], dim=-1)
        output = model(rays, is_training=is_training, epoch=0)
        assert output.shape == (1, 17, 3)
        assert torch.isfinite(output).all()
        outputs.append(output)
    sum(output.square().mean() for output in outputs).backward()
    global_parameters = list(model.features.blocks_global_joint_view.parameters())
    assert global_parameters
    assert all(parameter.grad is not None for parameter in global_parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in global_parameters)
    assert model.features.global_jv_gate.grad is not None
    assert torch.isfinite(model.features.global_jv_gate.grad).all()
    os.environ.pop("TRAIN_FIXED_NUM_VIEWS", None)
    os.environ.pop("GBT_GLOBAL_JV_DEPTH", None)
    os.environ.pop("GBT_GLOBAL_JV_BIASED", None)
    os.environ.pop("GBT_GLOBAL_JV_GATED", None)


if __name__ == "__main__":
    test_pairwise_ray_distance()
    test_fusion_token_receives_per_view_geometry_inconsistency()
    test_confidence_bias_direction_and_gradient()
    test_geometry_bias_direction_and_gradient()
    test_rumpl_uses_one_bias_model_for_all_view_counts()
    test_fixed_two_view_training_keeps_five_view_inference_capacity()
    test_token_dropout_supports_training_and_multiview_inference()
    test_global_joint_view_refinement_supports_variable_view_counts()
    print("GBT focused tests passed")
