#!/usr/bin/env python3
"""Unit tests for the opt-in single-pass PFT ablation."""

import pathlib
import sys

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from models.multiview_rumpl import apply_pose_fusion_blocks


class AddAndCount(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.value


def test_public_repeat_last_path():
    blocks = nn.ModuleList([AddAndCount(1.0), AddAndCount(10.0)])
    result = apply_pose_fusion_blocks(torch.zeros(1), blocks, repeat_last=True)
    assert result.item() == 21.0
    assert [block.calls for block in blocks] == [1, 2]


def test_single_pass_ablation():
    blocks = nn.ModuleList([AddAndCount(1.0), AddAndCount(10.0)])
    result = apply_pose_fusion_blocks(torch.zeros(1), blocks, repeat_last=False)
    assert result.item() == 11.0
    assert [block.calls for block in blocks] == [1, 1]


if __name__ == "__main__":
    test_public_repeat_last_path()
    test_single_pass_ablation()
    print("test_pft_repeat_last: PASS")
