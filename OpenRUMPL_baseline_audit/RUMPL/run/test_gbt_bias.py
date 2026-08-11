import importlib.util
import os
from pathlib import Path

import torch

from core.config import config, update_config


ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
OLD_MODEL = Path("/mnt/data/cjyoutput/baseline_reaudit_20260722/snapshot/R5_workers16_fix_scheduler_exact_seed0_20260722/multiview_rumpl.py")
NEW_MODEL = ROOT / "RUMPL/lib/models/multiview_rumpl.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(module):
    return module.MultiView_RUMPL(
        num_joints=config.NETWORK.NUM_JOINTS,
        embed_dim_ratio=config.NETWORK.DIM,
        depth=config.NETWORK.TRANSFORMER_DEPTH,
        num_heads=config.NETWORK.TRANSFORMER_HEADS,
        drop_rate=config.NETWORK.POSEFORMER_DROP_RATE,
        attn_drop_rate=config.NETWORK.POSEFORMER_ATTN_DROP_RATE,
        drop_path_rate=config.NETWORK.POSEFORMER_DROP_PATH_RATE,
        num_views=config.DATASET.N_VIEWS_TRAIN_TEST_ALL,
        linear_weighted_mean=config.NETWORK.POSEFORMER_LINEAR_WEIGHTED_MEAN,
        hidden_dim=config.NETWORK.POSEFORMER_OUTPUT_HEAD_HIDDEN_DIM,
        cfg=config,
    )


def test_ray_distance(module):
    direction = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    point = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    assert torch.equal(module.pairwise_ray_distance(direction, point), torch.zeros(1, 2, 2))

    direction = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    point = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    expected = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
    torch.testing.assert_close(module.pairwise_ray_distance(direction, point), expected)

    direction = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    point = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]])
    target = torch.tensor([[[0.0, 0.5, 0.0]]])
    expected = torch.tensor([[[0.5, 1.5]]])
    torch.testing.assert_close(module.point_to_ray_distance(direction, point, target), expected)
    distance = torch.tensor(
        [[[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]]]
    )
    normalized = module.normalize_pairwise_distance(distance)
    # Median of the positive entries is 4, while structural zeros stay zero.
    torch.testing.assert_close(normalized, distance / 4.0)


def test_oracle_reliability(module):
    direction = torch.tensor([[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]])
    point = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]])
    target = torch.tensor([[[0.0, 0.5, 0.0]]])
    penalty = module.oracle_reliability_with_fusion_token(
        direction, point, target, temperature=0.5
    )
    assert penalty.shape == (1, 3, 3)
    torch.testing.assert_close(penalty[:, 0, 1:].mean(dim=-1), torch.zeros(1))
    assert penalty[0, 0, 1] < penalty[0, 0, 2]
    assert torch.count_nonzero(penalty[:, 1:, :]) == 0

    attention = module.Attention(dim=4, num_heads=1, qkv_bias=False)
    os.environ["GBT_SAVE_ATTN"] = "1"
    attention(torch.zeros(1, 3, 4), reliability_penalty=penalty)
    assert attention.last_attn[0, 0, 0, 1] > attention.last_attn[0, 0, 0, 2]


def test_attention_bias(module):
    attention = module.Attention(
        dim=4, num_heads=1, qkv_bias=False,
        learnable_conf_bias=True, learnable_geom_bias=True,
        conf_bias_init=1.0, geom_bias_init=1.0,
    )
    os.environ["GBT_SAVE_ATTN"] = "1"
    x = torch.zeros(1, 2, 4)
    conf = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    geom = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
    attention(x, conf_bias=conf, geom_distance=torch.zeros_like(geom))
    assert torch.all(attention.last_attn[..., 1] > attention.last_attn[..., 0])
    attention(x, conf_bias=torch.zeros_like(conf), geom_distance=geom)
    assert torch.all(attention.last_attn[..., 1] < attention.last_attn[..., 0])
    loss = attention(x, conf_bias=conf, geom_distance=geom).square().sum()
    loss.backward()
    assert attention.gbt_conf_scale.grad is not None
    assert attention.gbt_geom_scale.grad is not None


def test_baseline_identity(old_module, new_module):
    for name in (
        "GBT_LEARNABLE_BIAS", "GBT_USE_CONF_BIAS", "GBT_USE_GEOM_BIAS",
        "GBT_CONF_INIT", "GBT_GEOM_INIT", "GBT_FUSION_GEOM", "GBT_SAVE_ATTN",
        "GBT_ORACLE_RELIABILITY", "GBT_ORACLE_TEMPERATURE",
        "GBT_COARSE_RELIABILITY",
        "GBT_LEARNED_RELIABILITY", "GBT_RELIABILITY_TEMPERATURE",
        "GBT_RELIABILITY_GATE_INIT", "GBT_RELIABILITY_AUX_WEIGHT",
        "GBT_RELIABILITY_PAIR_GATE", "GBT_RELIABILITY_RANK_WEIGHT",
        "RUMPL_GLOBAL_JOINT_VIEW_FUSION", "RUMPL_GLOBAL_JOINT_VIEW_DEPTH",
        "RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS",
        "RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS",
        "RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT",
        "RUMPL_GLOBAL_JOINT_VIEW_COUNT_GATE",
        "RUMPL_GLOBAL_JOINT_VIEW_GATE_MAX_INIT",
    ):
        os.environ.pop(name, None)
    torch.manual_seed(7)
    old_model = make_model(old_module).eval()
    torch.manual_seed(7)
    new_model = make_model(new_module).eval()
    assert old_model.state_dict().keys() == new_model.state_dict().keys()
    new_model.load_state_dict(old_model.state_dict(), strict=True)
    x = torch.randn(2, config.NETWORK.NUM_JOINTS, 5, 7)
    with torch.no_grad():
        old_output = old_model(x, is_training=False)
        new_output = new_model(x, is_training=False)
    torch.testing.assert_close(old_output, new_output, rtol=0, atol=0)


def test_biased_model_forward(new_module):
    os.environ["GBT_LEARNABLE_BIAS"] = "1"
    os.environ["GBT_USE_CONF_BIAS"] = "1"
    os.environ["GBT_USE_GEOM_BIAS"] = "1"
    os.environ["GBT_CONF_INIT"] = "0.1"
    os.environ["GBT_GEOM_INIT"] = "1.0"
    os.environ["GBT_FUSION_GEOM"] = "0"
    model = make_model(new_module).eval()
    scale_parameters = [name for name, _ in model.named_parameters() if "gbt_" in name]
    assert len(scale_parameters) == 2 * config.NETWORK.TRANSFORMER_DEPTH
    x = torch.randn(2, config.NETWORK.NUM_JOINTS, 5, 7)
    x[..., 6] = torch.rand_like(x[..., 6])
    with torch.no_grad():
        output = model(x, is_training=False)
    assert output.shape == (2, config.NETWORK.NUM_JOINTS, 3)
    assert torch.isfinite(output).all()


def test_oracle_model_forward(new_module):
    os.environ["GBT_LEARNABLE_BIAS"] = "0"
    os.environ["GBT_ORACLE_RELIABILITY"] = "1"
    os.environ["GBT_ORACLE_TEMPERATURE"] = "0.02"
    model = make_model(new_module).eval()
    x = torch.randn(2, config.NETWORK.NUM_JOINTS, 5, 7)
    x[..., 6] = torch.rand_like(x[..., 6])
    target = torch.randn(2, config.NETWORK.NUM_JOINTS, 3)
    with torch.no_grad():
        coarse = model(x, is_training=False, disable_reliability=True)
        output = model(x, is_training=False, oracle_target=target)
    assert output.shape == (2, config.NETWORK.NUM_JOINTS, 3)
    assert coarse.shape == output.shape
    assert torch.isfinite(output).all()


def test_learned_reliability_model(new_module):
    for name in (
        "GBT_LEARNABLE_BIAS", "GBT_ORACLE_RELIABILITY", "GBT_LEARNED_RELIABILITY"
    ):
        os.environ.pop(name, None)
    torch.manual_seed(23)
    baseline = make_model(new_module).eval()

    os.environ["GBT_LEARNED_RELIABILITY"] = "1"
    os.environ["GBT_RELIABILITY_TEMPERATURE"] = "0.2"
    os.environ["GBT_RELIABILITY_GATE_INIT"] = "0.05"
    torch.manual_seed(23)
    learned = make_model(new_module).eval()
    baseline_state = baseline.state_dict()
    learned_state = learned.state_dict()
    for name, value in baseline_state.items():
        torch.testing.assert_close(value, learned_state[name], rtol=0, atol=0)

    x = torch.randn(2, config.NETWORK.NUM_JOINTS, 5, 7)
    x[..., 6] = torch.rand_like(x[..., 6])
    target = torch.randn(2, config.NETWORK.NUM_JOINTS, 3)
    with torch.no_grad():
        baseline_output = baseline(x, is_training=False)
        learned_output = learned(x, is_training=False)
    torch.testing.assert_close(baseline_output, learned_output, rtol=0, atol=0)
    learned.train()
    train_output, aux_loss = learned(x, is_training=True, reliability_target=target)
    assert train_output.shape == baseline_output.shape
    assert aux_loss.ndim == 0 and torch.isfinite(aux_loss)
    (train_output.square().mean() + aux_loss).backward()
    assert learned.reliability_predictor[-1].weight.grad is not None
    assert learned.gbt_reliability_gate_logit.grad is not None

    os.environ["GBT_RELIABILITY_PAIR_GATE"] = "1"
    os.environ["GBT_RELIABILITY_RANK_WEIGHT"] = "1.0"
    torch.manual_seed(23)
    pair_gated = make_model(new_module).train()
    pair_output, pair_aux = pair_gated(
        x, is_training=True, reliability_target=target
    )
    (pair_output.square().mean() + pair_aux).backward()
    assert pair_gated.reliability_pair_gate[-1].weight.grad is not None
    assert torch.isfinite(pair_output).all() and torch.isfinite(pair_aux)


def test_global_joint_view_model(new_module):
    for name in (
        "GBT_LEARNABLE_BIAS", "GBT_ORACLE_RELIABILITY",
        "GBT_LEARNED_RELIABILITY",
    ):
        os.environ.pop(name, None)
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_FUSION"] = "1"
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_DEPTH"] = "2"
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS"] = "1"
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS"] = "1"
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_GEOM_NORM"] = "1"
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_GATE_INIT"] = "0.05"
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_COUNT_GATE"] = "1"
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_GATE_MAX_INIT"] = "0.12"
    model = make_model(new_module).train()
    x = torch.randn(2, config.NETWORK.NUM_JOINTS, 5, 7)
    x[..., 6] = torch.rand_like(x[..., 6])
    output = model(x, is_training=True)
    assert output.shape == (2, config.NETWORK.NUM_JOINTS, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.global_joint_embedding.grad is not None
    assert model.global_joint_view_gate_logit.grad is not None
    assert model.global_joint_view_gate_slope.grad is not None
    low_gate = torch.sigmoid(model.global_joint_view_gate_logit)
    high_gate = torch.sigmoid(
        model.global_joint_view_gate_logit + model.global_joint_view_gate_slope
    )
    torch.testing.assert_close(low_gate, torch.tensor(0.05), atol=1e-6, rtol=0)
    torch.testing.assert_close(high_gate, torch.tensor(0.12), atol=1e-6, rtol=0)
    assert model.global_joint_view_blocks[0].attn.gbt_conf_scale.grad is not None
    assert model.global_joint_view_blocks[0].attn.gbt_geom_scale.grad is not None


def test_singleframe_gbt_model(new_module):
    os.environ["RUMPL_GLOBAL_JOINT_VIEW_FUSION"] = "0"
    os.environ["RUMPL_SINGLEFRAME_GBT"] = "1"
    os.environ["RUMPL_SF_GBT_ENCODER_DEPTH"] = "3"
    os.environ["RUMPL_SF_GBT_DECODER_DEPTH"] = "2"
    os.environ["RUMPL_SF_GBT_PFT_DEPTH"] = "4"
    os.environ["RUMPL_SF_GBT_CONF_BIAS"] = "1"
    os.environ["RUMPL_SF_GBT_GEOM_BIAS"] = "1"
    os.environ["RUMPL_SF_GBT_GEOM_NORM"] = "1"
    model = make_model(new_module).train()
    x = torch.randn(2, config.NETWORK.NUM_JOINTS, 5, 7)
    x[..., :3] = torch.nn.functional.normalize(x[..., :3], dim=-1)
    x[..., 6] = torch.rand_like(x[..., 6])
    output = model(x, is_training=True)
    assert output.shape == (2, config.NETWORK.NUM_JOINTS, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.sf_gbt_ray_embed.weight.grad is not None
    assert model.sf_gbt_queries.grad is not None
    assert model.sf_gbt_encoder[0].attn.gbt_conf_scale.grad is not None
    assert model.sf_gbt_encoder[0].attn.gbt_geom_scale.grad is not None
    model.eval()
    permutation = torch.tensor([4, 1, 3, 0, 2])
    with torch.no_grad():
        original = model(x, is_training=False)
        permuted = model(x[:, :, permutation], is_training=False)
    torch.testing.assert_close(original, permuted, atol=2e-5, rtol=0)


def main():
    update_config(str(CFG))
    old_module = load_module("rumpl_pre_gbt", OLD_MODEL)
    new_module = load_module("rumpl_with_gbt", NEW_MODEL)
    test_ray_distance(new_module)
    test_oracle_reliability(new_module)
    test_attention_bias(new_module)
    test_baseline_identity(old_module, new_module)
    test_biased_model_forward(new_module)
    test_oracle_model_forward(new_module)
    test_learned_reliability_model(new_module)
    test_global_joint_view_model(new_module)
    test_singleframe_gbt_model(new_module)
    print(
        "GBT tests passed: geometry, reliability, global joint-view, "
        "single-frame query decoder, baseline identity"
    )


if __name__ == "__main__":
    main()
