#!/usr/bin/env python3
"""Shape, gradient and camera-consistency checks for H40's temporal model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


REPO = Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL")
sys.path.insert(0, str(REPO / "lib"))

from models.temporal_gbt_rumpl import TemporalJointViewRUMPL


class FakeBackbone(nn.Module):
    def __init__(self, joints=17, dim=24, views=4):
        super().__init__()
        from models.multiview_rumpl import Block

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, joints, dim))
        self.encoding_to_embedding = nn.Linear(6, dim // 2)
        self.confidence_to_embedding = nn.Linear(1, dim // 2)
        self.concat_direction_and_intersection_first = True
        self.not_use_intersection_features = False
        self.concat_confidence = True
        self.input_harmonic_l = 0
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.add_view_enc = False
        self.View_enc_learned = nn.Parameter(torch.zeros(1, views + 1, dim))
        self.pos_drop = nn.Identity()
        self.blocks_view_fusion = nn.ModuleList([Block(dim, 4, qkv_bias=True)])
        self.View_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([Block(dim, 4, qkv_bias=True)])
        self.Spatial_norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        self.tri_anchor = True
        self.tri_anchor_reg = 1e-4
        self.tri_anchor_conf_eps = 0.05
        self.tri_anchor_gate = nn.Parameter(torch.tensor(1.0))


def run_case(biased):
    torch.manual_seed(7)
    model = TemporalJointViewRUMPL(
        FakeBackbone(), depth=1, num_heads=4, biased=biased, token_dropout=0.2
    )
    model.train()
    rays = torch.randn(2, 3, 17, 4, 7)
    rays[..., 6] = torch.sigmoid(rays[..., 6])
    output, views = model(rays, num_views=2)
    assert output.shape == (2, 3, 17, 3)
    assert views.shape == (2, 2)
    # A sequence uses exactly one camera subset across all frames by design;
    # view indices have no temporal dimension.
    assert views.ndim == 2
    assert model.global_gate.item() == 0.0
    loss = output.square().mean()
    loss.backward()
    assert model.blocks[0].attn.qkv.weight.grad is not None
    assert torch.isfinite(model.blocks[0].attn.qkv.weight.grad).all()


def main():
    run_case(False)
    run_case(True)
    print("TemporalJointViewRUMPL PASS unbiased+biased shapes/gradients")


if __name__ == "__main__":
    main()
