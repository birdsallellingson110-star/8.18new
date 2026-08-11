"""Forward/backward smoke checks for the H41-H45 controlled variants."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from core.config import config, update_config  # noqa: E402
from models.multiview_rumpl import get_multiview_rumpl_net  # noqa: E402


CFG = Path(
    "/mnt/data/cjyoutput/open_source_fusion_audit_20260731/"
    "H35_a1d_h21_refined_rumpl_tri_anchor.yaml"
)


def run_variant(relative: int, rate: float, diagonal: int) -> None:
    os.environ.update(
        {
            "RUMPL_RELATIVE_VIEW_FUSION": str(relative),
            "VFT_FULL_RANDOM_MASK": str(rate),
            "VFT_MASK_DIAGONAL": str(diagonal),
            "VFT_MASK_MIN_VIEWS": "2",
            "RUMPL_RANDOM_VIEW_SUBSET": "1",
            "TRAIN_FIXED_NUM_VIEWS": "2",
            "TRAIN_FIXED_NUM_VIEWS_EPOCHS": "8",
            "RUMPL_TRI_ANCHOR": "1",
            "GBT_LEARNABLE_BIAS": "0",
            "GBT_GLOBAL_JV_DEPTH": "0",
            "RUMPL_GBT_SET_DECODER": "0",
        }
    )
    torch.manual_seed(123)
    model = get_multiview_rumpl_net(config, is_train=True)
    model.train()
    direction = torch.randn(2, 17, 4, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    point = torch.randn(2, 17, 4, 3)
    confidence = torch.rand(2, 17, 4, 1)
    rays = torch.cat((direction, point, confidence), dim=-1)
    output = model(rays, is_training=True, epoch=0)
    assert output.shape == (2, 17, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if relative:
        gate = model.features.Relative_view_fusion.gate
        assert gate.grad is not None and torch.isfinite(gate.grad).all()
    del model, rays, output
    gc.collect()


if __name__ == "__main__":
    update_config(str(CFG))
    run_variant(relative=1, rate=0.0, diagonal=1)
    run_variant(relative=0, rate=0.4, diagonal=0)
    run_variant(relative=0, rate=0.5, diagonal=1)
    print("paper mask/fusion integration tests passed")
