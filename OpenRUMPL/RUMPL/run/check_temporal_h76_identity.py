#!/usr/bin/env python3
"""Verify that a zero-gated temporal wrapper exactly preserves H76.

This is a mandatory precondition for temporal experiments: the same rays and
camera subset are evaluated by the original frame-wise H76 forward and by the
new temporal wrapper.  No dataset is required, so the check isolates model
plumbing from preprocessing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))

from core.config import config, update_config
from models.multiview_rumpl import get_multiview_rumpl_net
from models.temporal_gbt_rumpl import TemporalJointViewRUMPL


def clean_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--time", type=int, default=3)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--fusion-mode",
        choices=(
            "global-residual", "query-residual", "pre-vft-temporal", "mixste-ttb",
            "mixste-ttb-residual",
            "mixste-alternating",
            "mixste-pose-residual",
        ),
        default="global-residual",
    )
    parser.add_argument("--check-gradient", action="store_true")
    parser.add_argument(
        "--backbone-flavor", choices=("h76", "h81"), default="h76"
    )
    args = parser.parse_args()

    os.environ["RUMPL_BACKBONE_FLAVOR"] = args.backbone_flavor
    os.environ["RUMPL_TRI_ANCHOR"] = "1"
    os.environ["RUMPL_TRI_ANCHOR_REG"] = "1e-4"
    os.environ["RUMPL_TRI_ANCHOR_CONF_EPS"] = "0.05"
    os.environ["RUMPL_ANCHOR_CENTERED_RAYS"] = "1"
    os.environ["RUMPL_ANCHOR_CENTER_PER_JOINT"] = "0"
    os.environ["RUMPL_INPUT_PLUCKER"] = "1"
    os.environ["RUMPL_INPUT_HARMONIC_L"] = "0"
    if args.backbone_flavor == "h81":
        os.environ["RUMPL_PER_JOINT_RESIDUAL_GATE"] = "1"
        for name in (
            "GBT_GLOBAL_JV_DEPTH",
            "GBT_LEARNABLE_BIAS",
            "RUMPL_GBT_SET_DECODER",
            "RUMPL_RELATIVE_VIEW_FUSION",
            "RUMPL_GEOMETRY_UNCERTAINTY_TOKEN",
            "RUMPL_POST_PFT_GRAPH_RESIDUAL",
            "RUMPL_JOINT_SPECIFIC_HEAD",
        ):
            os.environ[name] = "0"
    else:
        for name in (
            "GBT_GLOBAL_JV_DEPTH",
            "GBT_LEARNABLE_BIAS",
            "RUMPL_GBT_SET_DECODER",
            "RUMPL_RELATIVE_VIEW_FUSION",
            "RUMPL_GEOMETRY_UNCERTAINTY_TOKEN",
            "RUMPL_PER_JOINT_RESIDUAL_GATE",
            "RUMPL_POST_PFT_GRAPH_RESIDUAL",
            "RUMPL_JOINT_SPECIFIC_HEAD",
        ):
            os.environ[name] = "0"

    update_config(args.cfg)
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = args.views
    config.DATASET.TRAIN_ON_ALL_CAMERAS = True
    config.DATASET.TEST_ON_ALL_CAMERAS = True
    device = torch.device(args.device)
    base = get_multiview_rumpl_net(config, is_train=False)
    state = clean_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    from utils.rumpl_checkpoint_adapt import merge_pretrained_into_model_state

    merged, _ = merge_pretrained_into_model_state(
        base.state_dict(), state, strict_shapes=False
    )
    base.load_state_dict(merged, strict=False)
    temporal = TemporalJointViewRUMPL(
        base, depth=args.depth, num_heads=8, biased=False, token_dropout=0.0,
        residual_gate=True,
        fusion_mode=args.fusion_mode,
        temporal_length=args.time,
    ).to(device).eval()
    if hasattr(temporal, "global_gate"):
        temporal.global_gate.data.zero_()

    generator = torch.Generator(device=device).manual_seed(76003)
    shape = (args.batch_size, args.time, 17, args.views)
    direction = torch.randn(*shape, 3, generator=generator, device=device)
    direction = torch.nn.functional.normalize(direction, dim=-1)
    # Camera-like origins are shared across joints/time within each view, then
    # offset along the ray; any point on the same line is a valid representation.
    origins = torch.randn(
        args.batch_size, 1, 1, args.views, 3,
        generator=generator, device=device,
    )
    depth = 2.0 + torch.rand(*shape, 1, generator=generator, device=device)
    point = origins + depth * direction
    confidence = 0.2 + 0.8 * torch.rand(
        *shape, 1, generator=generator, device=device
    )
    rays = torch.cat((direction, point, confidence), dim=-1)

    with torch.no_grad():
        flat = rays.reshape(args.batch_size * args.time, 17, args.views, 7)
        expected = base(flat, is_training=False).reshape(
            args.batch_size, args.time, 17, 3
        )
        actual, _ = temporal(rays)
    difference = (actual - expected).abs()
    maximum = float(difference.max())
    mean = float(difference.mean())
    print(f"max_abs_diff={maximum:.9e}")
    print(f"mean_abs_diff={mean:.9e}")
    if maximum > 1e-4:
        raise SystemExit(
            f"FAIL: temporal gate=0 does not preserve frame-wise backbone (max={maximum})"
        )
    print(f"PASS: temporal gate=0 preserves frame-wise {args.backbone_flavor.upper()}")
    if args.check_gradient:
        temporal.train()
        temporal.backbone.eval()
        for parameter in temporal.backbone.parameters():
            parameter.requires_grad_(False)
        prediction, _ = temporal(rays)
        prediction.square().mean().backward()
        if args.fusion_mode == "query-residual":
            gradient = temporal.query_residual_head.weight.grad
            magnitude = 0.0 if gradient is None else float(gradient.abs().sum())
            print(f"query_head_grad_l1={magnitude:.9e}")
            if not magnitude > 0:
                raise SystemExit("FAIL: query residual head has no gradient")
            print("PASS: query residual branch receives gradient")
        elif args.fusion_mode in (
            "mixste-ttb", "mixste-ttb-residual", "mixste-alternating"
        ):
            # The zero output projections make the wrapper an exact H76
            # identity at step 0, but the TTB must still receive a gradient
            # through the residual projection so training can leave that
            # identity when temporal evidence is useful.
            gradient = temporal.mixste_ttb[0].attn.proj.weight.grad
            magnitude = 0.0 if gradient is None else float(gradient.abs().sum())
            print(f"{args.fusion_mode}_proj_grad_l1={magnitude:.9e}")
            if not magnitude > 0:
                raise SystemExit("FAIL: MixSTE TTB branch has no gradient")
            print("PASS: MixSTE TTB branch receives gradient")
        elif args.fusion_mode == "mixste-pose-residual":
            gradient = temporal.mixste_pose_head[-1].weight.grad
            magnitude = 0.0 if gradient is None else float(gradient.abs().sum())
            print(f"mixste_pose_head_grad_l1={magnitude:.9e}")
            if not magnitude > 0:
                raise SystemExit("FAIL: MixSTE pose residual head has no gradient")
            print("PASS: MixSTE pose residual branch receives gradient")


if __name__ == "__main__":
    main()
