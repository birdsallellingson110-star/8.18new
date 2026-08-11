#!/usr/bin/env python
"""Identity, gradient, and permutation checks for the two separate branches."""

import os

import torch

from core.config import config, update_config


CFG = (
    "configs/cmu_panoptic/rumpl_amass/"
    "crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_"
    "RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
)
CKPT = (
    "/mnt/data/cjyoutput/baseline_reaudit_20260722/output/"
    "multiview_amass_rumpl/multiview_rumpl_999/"
    "R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/"
    "model_best.pth.tar"
)
FEATURE_FLAGS = (
    "RUMPL_GATED_JOINT_ADAPTER",
    "RUMPL_JOINT_ADAPTER_DIRECT_READOUT",
    "RUMPL_JOINT_ADAPTER_COUNT_LOOKUP",
    "RUMPL_GLOBAL_JOINT_VIEW_FUSION",
    "RUMPL_GLOBAL_JOINT_VIEW_RESIDUAL",
    "RUMPL_GLOBAL_JOINT_VIEW_PLUCKER",
)


def build(mode):
    for name in FEATURE_FLAGS:
        os.environ[name] = "0"
    os.environ["RUMPL_ALT_JOINT_VIEW"] = "0"
    os.environ["RUMPL_VFT_DEPTH"] = "12"
    os.environ["RUMPL_PFT_DEPTH"] = "12"
    if mode == "direct":
        os.environ["RUMPL_GATED_JOINT_ADAPTER"] = "1"
        os.environ["RUMPL_JOINT_ADAPTER_DIRECT_READOUT"] = "1"
        os.environ["RUMPL_JOINT_ADAPTER_INDICES"] = "11"
        os.environ["RUMPL_JOINT_ADAPTER_VIEW_POWER"] = "0"
    elif mode == "global":
        os.environ["RUMPL_GLOBAL_JOINT_VIEW_FUSION"] = "1"
        os.environ["RUMPL_GLOBAL_JOINT_VIEW_RESIDUAL"] = "1"
        os.environ["RUMPL_GLOBAL_JOINT_VIEW_PLUCKER"] = "1"
        os.environ["RUMPL_GLOBAL_JOINT_VIEW_DEPTH"] = "3"
        os.environ["RUMPL_GLOBAL_JOINT_VIEW_CONF_BIAS"] = "1"
        os.environ["RUMPL_GLOBAL_JOINT_VIEW_GEOM_BIAS"] = "1"
        os.environ["RUMPL_GLOBAL_JOINT_VIEW_GEOM_NORM"] = "1"

    from models.multiview_rumpl import MultiView_RUMPL_G

    model = MultiView_RUMPL_G(config).cpu()
    state = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if mode == "base":
        assert not missing
    elif mode == "direct":
        assert missing and all("joint_adapter" in key for key in missing)
    else:
        assert missing and all("global_joint" in key for key in missing)
    assert not unexpected
    return model


def main():
    torch.manual_seed(41)
    update_config(CFG)
    baseline = build("base").eval()
    direct = build("direct").eval()
    global_branch = build("global").eval()
    inputs = torch.randn(2, 17, 5, 7)
    inputs[..., :3] = torch.nn.functional.normalize(inputs[..., :3], dim=-1)
    inputs[..., 6] = torch.rand(2, 17, 5)

    with torch.no_grad():
        reference = baseline(inputs, is_training=False)
        direct_output = direct(inputs, is_training=False)
        global_output = global_branch(inputs, is_training=False)
        permutation = torch.tensor([4, 1, 3, 0, 2])
        global_permuted = global_branch(
            inputs[:, :, permutation], is_training=False
        )

    direct_error = (reference - direct_output).abs().max().item()
    global_error = (reference - global_output).abs().max().item()
    permutation_error = (global_output - global_permuted).abs().max().item()
    assert direct_error < 1e-6, direct_error
    assert global_error < 1e-6, global_error
    assert permutation_error < 2e-5, permutation_error

    for model, fragment in (
        (direct, "joint_adapter"),
        (global_branch, "global_joint"),
    ):
        for name, parameter in model.named_parameters():
            parameter.requires_grad = fragment in name
        model.train()
        loss = model(inputs, is_training=True).square().mean()
        loss.backward()
        nonzero = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        ]
        assert nonzero, (fragment, nonzero)

    print(
        "TRANSFORMER_SEPARATE_V2_SMOKE_OK",
        f"direct_identity={direct_error:.3e}",
        f"global_identity={global_error:.3e}",
        f"global_permutation={permutation_error:.3e}",
    )


if __name__ == "__main__":
    main()
