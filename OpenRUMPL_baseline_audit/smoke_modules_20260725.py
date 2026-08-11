#!/usr/bin/env python
import os
import torch

os.environ["RUMPL_FIX_SCHEDULER_ORDER"] = "1"
os.environ["GBT_LEARNABLE_BIAS"] = "0"

from core.config import config, update_config
from models.multiview_rumpl import MultiView_RUMPL_G

update_config(
    "configs/cmu_panoptic/rumpl_amass/crf_4925_random_mmpose_hrnet_ConfConcat_2viewsV3V6_Seed0_RaySineEncNo_IntersectM_Miss20_ZrTknsNo_FuserRays_RNV5.yaml"
)


def smoke(name, env):
    for k in ["RUMPL_KPA", "RUMPL_MULTI_HYP", "RUMPL_POSE_CODEBOOK", "RUMPL_CONF_FILM"]:
        os.environ.pop(k, None)
    os.environ.update(env)
    m = MultiView_RUMPL_G(config).cuda().train()
    b, j, v = 2, 17, 5
    x = torch.randn(b, j, v, 7, device="cuda")
    x[..., -1] = torch.rand(b, j, v, device="cuda")
    tgt = torch.randn(b, j, 3, device="cuda")
    if env.get("RUMPL_POSE_CODEBOOK") == "1":
        out, aux = m(x, is_training=True, codebook_target=tgt)
        assert out.shape == (b, j, 3) and torch.isfinite(aux).all()
        print(name, "ok aux", float(aux))
    else:
        out = m(x, is_training=True)
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape == (b, j, 3)
        print(name, "ok", round(sum(p.numel() for p in m.parameters()) / 1e6, 3), "M")
    del m
    torch.cuda.empty_cache()


if __name__ == "__main__":
    smoke("baseline", {"RUMPL_KPA": "0", "RUMPL_MULTI_HYP": "1", "RUMPL_POSE_CODEBOOK": "0"})
    smoke("kpa", {"RUMPL_KPA": "1", "RUMPL_MULTI_HYP": "1", "RUMPL_POSE_CODEBOOK": "0"})
    smoke("mh3", {"RUMPL_KPA": "0", "RUMPL_MULTI_HYP": "3", "RUMPL_POSE_CODEBOOK": "0"})
    smoke("d3pct", {"RUMPL_KPA": "0", "RUMPL_MULTI_HYP": "1", "RUMPL_POSE_CODEBOOK": "1"})
    print("ALL SMOKE OK")
