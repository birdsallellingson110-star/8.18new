#!/usr/bin/env python
"""CPU smoke test for the alternating joint-view RUMPL path."""

import os

import torch


os.environ.update(
    {
        "RUMPL_ALT_JOINT_VIEW": "1",
        "RUMPL_ALT_JOINT_VIEW_DEPTH": "2",
        "RUMPL_PFT_DEPTH": "2",
        "RUMPL_MULTI_HYP": "1",
        "RUMPL_FIX_PFT_LAST_BLOCK": "1",
        "GBT_LEARNABLE_BIAS": "0",
        "GBT_LEARNED_RELIABILITY": "0",
        "RUMPL_GLOBAL_JOINT_VIEW_FUSION": "0",
        "RUMPL_KPA": "0",
        "RUMPL_POSE_CODEBOOK": "0",
        "RUMPL_RAY_DEPTH_AUX": "0",
        "RUMPL_TRI_ANCHOR": "0",
    }
)

from core.config import config, update_config
from models.multiview_rumpl import MultiView_RUMPL_G


CFG = (
    "configs/cmu_panoptic/rumpl_amass/"
    "crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_"
    "RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
)


def main():
    torch.manual_seed(7)
    update_config(CFG)
    model = MultiView_RUMPL_G(config).cpu()
    model.eval()

    batch, joints, views = 2, 17, 5
    inputs = torch.randn(batch, joints, views, 7)
    inputs[..., 0:3] = torch.nn.functional.normalize(inputs[..., 0:3], dim=-1)
    inputs[..., 6] = torch.rand(batch, joints, views)

    with torch.no_grad():
        prediction = model(inputs, is_training=False)
        permutation = torch.tensor([3, 0, 4, 1, 2])
        permuted_prediction = model(
            inputs[:, :, permutation, :], is_training=False
        )

    assert prediction.shape == (batch, joints, 3)
    assert torch.isfinite(prediction).all()
    permutation_error = (prediction - permuted_prediction).abs().max().item()
    assert permutation_error < 1e-5, permutation_error

    model.train()
    train_prediction = model(inputs, is_training=True)
    loss = train_prediction.square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "ALT_JV_SMOKE_OK",
        f"shape={tuple(prediction.shape)}",
        f"permutation_max_abs={permutation_error:.3e}",
        f"parameters={parameter_count / 1e6:.3f}M",
    )


if __name__ == "__main__":
    main()
