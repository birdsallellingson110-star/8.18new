#!/usr/bin/env python3
"""Shape, identity, permutation, and gradient tests for H21."""

import torch

from iterative_pose_query_refiner import IterativePoseQueryRefiner


def main() -> None:
    torch.manual_seed(0)
    model = IterativePoseQueryRefiner()
    for n_views in (2, 3, 4):
        heatmaps = torch.rand(n_views, 13, 24, 18)
        query = torch.rand(n_views, 13, 2)
        query[..., 0] *= 17
        query[..., 1] *= 23
        detector = query + torch.randn_like(query)
        confidence = torch.rand(n_views, 13)
        joint_ids = torch.tensor(
            [11, 13, 15, 12, 14, 16, 0, 5, 7, 9, 6, 8, 10]
        )
        output, auxiliary = model(
            heatmaps, query, detector, confidence, joint_ids
        )
        identity_error = float((output - detector).abs().max())
        assert identity_error == 0.0
        permutation = torch.randperm(n_views)
        permuted, _ = model(
            heatmaps[permutation],
            query[permutation],
            detector[permutation],
            confidence[permutation],
            joint_ids,
        )
        permutation_error = float(
            (permuted - output[permutation]).abs().max()
        )
        assert permutation_error < 1e-5
        loss = (output - query).square().mean()
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(model.output[-1].weight.grad.abs().sum())
        assert gradient > 0
        print(
            f"V{n_views}: identity={identity_error:.1e}, "
            f"permutation={permutation_error:.1e}, "
            f"head_gradient={gradient:.3e}, "
            f"patch={tuple(auxiliary['query_patches'].shape)}"
        )


if __name__ == "__main__":
    main()
