#!/usr/bin/env python3
"""Shape, identity initialization and backward smoke test."""

import torch

from sparse_epipolar_candidate_transformer import (
    SparseEpipolarCandidateTransformer,
    candidate_loss,
)


def main() -> None:
    torch.manual_seed(7)
    for n_views in (2, 3, 4):
        batch, topk = 3, 8
        scores = torch.rand(batch, n_views, topk).clamp_min(0.01)
        centers = torch.randn(batch, n_views, 3)
        directions = torch.nn.functional.normalize(
            torch.randn(batch, n_views, topk, 3), dim=-1
        )
        joints = torch.tensor([1, 9, 16])
        view_mask = torch.ones(batch, n_views, dtype=torch.bool)
        view_mask[-1, -1] = False
        model = SparseEpipolarCandidateTransformer()
        logits = model(scores, centers, directions, joints, view_mask)
        expected = scores.log().masked_fill(
            ~view_mask[:, :, None], -1e4
        )
        torch.testing.assert_close(logits, expected)
        targets = torch.zeros(batch, n_views, dtype=torch.long)
        loss = candidate_loss(logits, targets, view_mask)
        loss.backward()
        assert model.residual_gate.grad is not None
        assert torch.isfinite(model.residual_gate.grad)
        print(
            f"V{n_views}: identity max error="
            f"{(logits - expected).abs().max().item():.3g}, "
            f"loss={loss.item():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
