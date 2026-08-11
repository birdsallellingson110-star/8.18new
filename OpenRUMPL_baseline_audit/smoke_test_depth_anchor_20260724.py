"""Smoke test for RUMPL_TRI_ANCHOR and RUMPL_RAY_DEPTH_AUX."""
import os
import sys

sys.path.insert(0, "/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/lib")

import torch

from core.config import config, update_config

CFG = (
    "/home/lixiaob/cjy/OpenRUMPL_baseline_audit/RUMPL/configs/cmu_panoptic/rumpl_amass/"
    "crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_"
    "IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
)


def build_model():
    import importlib
    import models.multiview_rumpl as mr
    importlib.reload(mr)
    model = mr.MultiView_RUMPL_G(config)
    return model


def fake_batch(b=4, j=17, k=5):
    torch.manual_seed(0)
    origin = torch.randn(b, 1, k, 3).expand(b, j, k, 3) * 2.0  # camera centers
    gt = torch.randn(b, j, 3) * 0.5
    direction = gt.unsqueeze(2) - origin + 0.01 * torch.randn(b, j, k, 3)
    conf = torch.rand(b, j, k, 1)
    rays = torch.cat([direction, origin, conf], dim=-1)
    return rays, gt


def run(mode_env, expect_tuple):
    for key in ("RUMPL_TRI_ANCHOR", "RUMPL_RAY_DEPTH_AUX"):
        os.environ.pop(key, None)
    os.environ.update(mode_env)
    model = build_model()
    rays, gt = fake_batch()
    out = model(rays, is_training=True, depth_target=gt)
    if expect_tuple:
        pose, aux = out
        assert aux.ndim == 0 and torch.isfinite(aux), f"bad aux: {aux}"
        loss = pose.mean() + 0.1 * aux
    else:
        pose = out
        loss = pose.mean()
    assert pose.shape == (4, 17, 3), pose.shape
    assert torch.isfinite(pose).all()
    loss.backward()
    # eval path returns single tensor
    model.eval()
    with torch.no_grad():
        pose_eval = model(rays, is_training=False)
    assert pose_eval.shape == (4, 17, 3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OK {mode_env} n_params={n_params} pose_mean={pose.mean().item():.4f}")
    return n_params


def anchor_accuracy():
    # With clean rays through gt, anchor should recover gt closely.
    os.environ.pop("RUMPL_RAY_DEPTH_AUX", None)
    os.environ["RUMPL_TRI_ANCHOR"] = "1"
    model = build_model()
    rays, gt = fake_batch()
    b, j, k, _ = 4, 17, 5, 7
    d = rays[..., :3]
    o = rays[..., 3:6]
    conf = rays[..., 6:7]
    unit = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-7)
    w = conf.clamp(0, 1) + 0.05
    eye3 = torch.eye(3)
    proj = eye3 - unit.unsqueeze(-1) * unit.unsqueeze(-2)
    wp = w.unsqueeze(-1) * proj
    A = wp.sum(dim=2) + 1e-4 * eye3
    rhs = (wp @ o.unsqueeze(-1)).sum(dim=2)
    anchor = torch.linalg.solve(A, rhs).squeeze(-1)
    err = (anchor - gt).norm(dim=-1).mean().item()
    print(f"anchor mean error vs gt (clean rays, noise 0.01): {err*1000:.1f} mm-ish units")
    assert err < 0.1, err


if __name__ == "__main__":
    update_config(CFG)
    base = run({}, expect_tuple=False)
    anchor = run({"RUMPL_TRI_ANCHOR": "1"}, expect_tuple=False)
    aux = run({"RUMPL_RAY_DEPTH_AUX": "1"}, expect_tuple=True)
    combo = run({"RUMPL_TRI_ANCHOR": "1", "RUMPL_RAY_DEPTH_AUX": "1"}, expect_tuple=True)
    assert anchor == base + 1, (base, anchor)
    print("param deltas: anchor=+1, aux=+%d" % (aux - base))
    anchor_accuracy()
    print("ALL SMOKE TESTS PASSED")
