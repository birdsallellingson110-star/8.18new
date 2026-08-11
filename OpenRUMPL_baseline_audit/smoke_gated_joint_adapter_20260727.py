#!/usr/bin/env python
"""Verify R5 equivalence, gradients and view permutation invariance."""

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


def build(enabled):
    os.environ["RUMPL_GATED_JOINT_ADAPTER"] = "1" if enabled else "0"
    os.environ["RUMPL_ALT_JOINT_VIEW"] = "0"
    os.environ["RUMPL_VFT_DEPTH"] = "12"
    os.environ["RUMPL_PFT_DEPTH"] = "12"
    os.environ["RUMPL_FIX_PFT_LAST_BLOCK"] = "0"
    from models.multiview_rumpl import MultiView_RUMPL_G

    model = MultiView_RUMPL_G(config).cpu()
    state = torch.load(CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if enabled:
        assert missing and all("joint_adapter" in key for key in missing)
    else:
        assert not missing
    assert not unexpected
    return model


def main():
    torch.manual_seed(11)
    update_config(CFG)
    baseline = build(False).eval()
    adapter = build(True).eval()
    inputs = torch.randn(2, 17, 5, 7)
    inputs[..., :3] = torch.nn.functional.normalize(inputs[..., :3], dim=-1)
    inputs[..., 6] = torch.rand(2, 17, 5)

    with torch.no_grad():
        base_prediction = baseline(inputs, is_training=False)
        adapter_prediction = adapter(inputs, is_training=False)
        permutation = torch.tensor([4, 1, 3, 0, 2])
        permuted_prediction = adapter(
            inputs[:, :, permutation], is_training=False
        )
    equivalence = (base_prediction - adapter_prediction).abs().max().item()
    permutation_error = (
        adapter_prediction - permuted_prediction
    ).abs().max().item()
    assert equivalence < 1e-6, equivalence
    assert permutation_error < 1e-5, permutation_error

    for name, parameter in adapter.named_parameters():
        parameter.requires_grad = "joint_adapter" in name
    adapter.train()
    loss = adapter(inputs, is_training=True).square().mean()
    loss.backward()
    nonzero_gradients = sum(
        int(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
        )
        for parameter in adapter.parameters()
        if parameter.requires_grad
    )
    assert nonzero_gradients > 0
    print(
        "GATED_ADAPTER_SMOKE_OK",
        f"r5_equivalence={equivalence:.3e}",
        f"permutation={permutation_error:.3e}",
        f"nonzero_gradients={nonzero_gradients}",
    )


if __name__ == "__main__":
    main()
