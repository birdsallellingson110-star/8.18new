#!/usr/bin/env python3
"""Identity, variable-view and backward checks for dense residual fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from dense_geometry_residual_fusion import DenseGeometryResidualFusion


def main() -> None:
    torch.manual_seed(7)
    model = DenseGeometryResidualFusion()
    for n_views in (2, 3, 4):
        heatmaps = torch.rand(n_views, 17, 24, 18)
        heatmaps = heatmaps / heatmaps.flatten(-2).amax(
            -1, keepdim=True
        )[..., None]
        support = torch.rand(n_views, n_views, 17, 24, 18)
        logits, auxiliary = model(heatmaps, support)
        expected = torch.log(heatmaps + 1e-4)
        identity_error = float((logits - expected).abs().max())
        if identity_error != 0.0:
            raise AssertionError(
                f"V{n_views}: initialization is not exact identity: "
                f"{identity_error}"
            )
        target = torch.randint(0, 24 * 18, (n_views, 17))
        loss = functional.cross_entropy(
            logits.flatten(0, 1).flatten(-2),
            target.flatten(),
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        gate_gradient = float(
            model.global_geometry_strength.grad.abs()
        )
        last_gradient = float(
            model.spatial_residual[-1].weight.grad.abs().sum()
        )
        if gate_gradient == 0.0 or last_gradient == 0.0:
            raise AssertionError(
                f"V{n_views}: blocked gradient gate={gate_gradient}, "
                f"residual={last_gradient}"
            )
        print(
            {
                "views": n_views,
                "identity_max_abs": identity_error,
                "gate_gradient": gate_gradient,
                "residual_last_gradient": last_gradient,
                "geometry_weight_shape": tuple(
                    auxiliary["geometry_weight"].shape
                ),
            }
        )


if __name__ == "__main__":
    main()
