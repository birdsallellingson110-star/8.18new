"""Deterministic invariants for MTF/Masked-Gifformer view-mask ablations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from models.multiview_rumpl import build_view_attention_mask  # noqa: E402


def test_zero_rate_masks_nothing() -> None:
    mask = build_view_attention_mask(8, 4, 0.0, torch.device("cpu"))
    assert not mask.any()


def test_fully_random_rate_one_masks_diagonal_but_keeps_fusion_observation() -> None:
    torch.manual_seed(7)
    mask = build_view_attention_mask(
        16, 4, 1.0, torch.device("cpu"), mask_diagonal=True
    )
    diagonal = mask[:, 1:, 1:].diagonal(dim1=1, dim2=2)
    assert diagonal.all()
    assert (~mask[:, 0, 1:]).sum(dim=1).eq(1).all()
    assert not mask[:, :, 0].any()


def test_mtf_style_rate_one_protects_view_self_edges() -> None:
    torch.manual_seed(7)
    mask = build_view_attention_mask(
        16, 4, 1.0, torch.device("cpu"), mask_diagonal=False
    )
    diagonal = mask[:, 1:, 1:].diagonal(dim1=1, dim2=2)
    assert not diagonal.any()
    off_diagonal = mask[:, 1:, 1:].clone()
    indices = torch.arange(4)
    off_diagonal[:, indices, indices] = False
    assert off_diagonal.sum(dim=(1, 2)).eq(12).all()
    assert (~mask[:, 0, 1:]).sum(dim=1).eq(1).all()


if __name__ == "__main__":
    test_zero_rate_masks_nothing()
    test_fully_random_rate_one_masks_diagonal_but_keeps_fusion_observation()
    test_mtf_style_rate_one_protects_view_self_edges()
    print("view attention mask tests passed")
