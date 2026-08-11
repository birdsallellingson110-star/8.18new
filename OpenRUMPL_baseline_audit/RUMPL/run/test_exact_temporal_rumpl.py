import os

import torch

from models.exact_temporal_rumpl import ExactTemporalRUMPL


ROOT = "/home/lixiaob/cjy/OpenRUMPL_baseline_audit"
CONFIG = ROOT + "/RUMPL/configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
CHECKPOINT = "/mnt/data/cjyoutput/baseline_reaudit_20260722/output/multiview_amass_rumpl/multiview_rumpl_999/R5_workers16_fix_scheduler_exact_seed0_20260722_2026-07-22_23-01-15/model_best.pth.tar"


def main():
    os.environ["RUMPL_FIX_PFT_LAST_BLOCK"] = "0"
    os.environ["GBT_LEARNABLE_BIAS"] = "0"
    torch.manual_seed(11)
    model = ExactTemporalRUMPL(CONFIG, CHECKPOINT).eval()
    assert model.dim == 256
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.temporal.parameters())

    batch_size, num_joints, num_views, num_frames = 2, 17, 2, 9
    rays = torch.randn(batch_size, num_joints, num_views, num_frames, 6)
    confidence = torch.rand(batch_size, num_joints, num_views, num_frames, 1)
    delta_t = torch.arange(-4, 5).float()[None].expand(batch_size, -1) / 30.0
    with torch.no_grad():
        temporal_output = model(rays, confidence, delta_t)
        baseline_output = model(rays, confidence, delta_t, no_temporal=True)
        center_input = torch.cat(
            [rays[:, :, :, 4], confidence[:, :, :, 4]], dim=-1
        )
        direct_output = model.backbone(center_input, is_training=False)
    torch.testing.assert_close(temporal_output, baseline_output, rtol=0, atol=0)
    # Batched nine-frame VFT changes GEMM accumulation order by about 1e-6.
    torch.testing.assert_close(baseline_output, direct_output, rtol=0, atol=2e-6)
    assert temporal_output.shape == (batch_size, num_joints, 3)

    model.train()
    assert not model.backbone.training
    prediction = model(rays, confidence, delta_t)
    prediction.square().mean().backward()
    assert model.temporal.output.weight.grad is not None
    assert torch.isfinite(model.temporal.output.weight.grad).all()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())

    motion_model = ExactTemporalRUMPL(
        CONFIG, CHECKPOINT, motion_only=True
    ).eval()
    static_rays = rays[:, :, :, 4:5].expand_as(rays).clone()
    static_confidence = confidence[:, :, :, 4:5].expand_as(confidence).clone()
    with torch.no_grad():
        static_temporal = motion_model(
            static_rays, static_confidence, delta_t
        )
        static_baseline = motion_model(
            static_rays, static_confidence, delta_t, no_temporal=True
        )
    torch.testing.assert_close(static_temporal, static_baseline, rtol=0, atol=0)
    print("Exact temporal RUMPL tests passed: R5 identity, center-frame identity, gradients")


if __name__ == "__main__":
    main()
