#!/usr/bin/env python3
"""Train the H40 retained-RUMPL joint-view-time model on real H36M.

This entry point is step-based so a paired unbiased/biased run can use the
same initialization, sample order, effective batch size, warmup and cosine
schedule.  It supervises all nine output frames with MSE, as in GBT.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))

import dataset
from core.config import config, update_config
from dataset.temporal_h36m_rumpl import (
    TemporalH36MRUMPL,
    collate_temporal_h36m,
)
from models.multiview_rumpl import get_multiview_rumpl_net
from models.temporal_gbt_rumpl import TemporalJointViewRUMPL


# Exact H36M joint weighting used by the official MixSTE training script
# (MixSTE/run.py, commit 72a36be).  The local RUMPL target order is the same
# 17-joint H36M order.  Keep this separate from the ordinary RUMPL MPJPE loss
# so an ablation can identify whether any change comes from the architecture or
# only from MixSTE's objective.
MIXSTE_H36M_WEIGHTS = (
    1.0, 1.0, 2.5, 2.5, 1.0, 2.5, 2.5, 1.0, 1.0,
    1.0, 1.5, 1.5, 4.0, 4.0, 1.5, 4.0, 4.0,
)


def mixste_original_loss(prediction, target):
    """Return the loss used in the official MixSTE H36M trainer.

    MixSTE combines weighted MPJPE with its published temporal-consistency
    term: ``0.5 * mean(weight * velocity^2) + 2 * velocity_error``.  We keep
    the expression verbatim (including the squared predicted velocity term)
    for a faithful loss ablation; the caller reports ordinary MPJPE
    independently, so this objective cannot be mistaken for the metric.
    Inputs are in metres and have shape ``(B,T,J,3)``.
    """

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "MixSTE loss expects matching (B,T,J,3) tensors, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[-2] != len(MIXSTE_H36M_WEIGHTS):
        raise ValueError(
            f"MixSTE H36M weights cover 17 joints, got {prediction.shape[-2]}"
        )
    weights = prediction.new_tensor(MIXSTE_H36M_WEIGHTS).view(1, 1, -1)
    position_error = (prediction - target).norm(dim=-1)
    loss_3d_pos = (weights * position_error).mean()
    if prediction.shape[1] < 2:
        # The official loss is defined for a sequence.  Keeping the position
        # term valid for T=1 makes the helper safe for smoke tests while all
        # actual MixSTE runs use T>=9.
        return loss_3d_pos
    dif_pred = prediction[:, 1:] - prediction[:, :-1]
    dif_target = target[:, 1:] - target[:, :-1]
    velocity_square = (
        dif_pred.square() * weights[:, :, :, None]
    ).mean()
    velocity_error = (dif_pred - dif_target).norm(dim=-1).mean()
    loss_diff = 0.5 * velocity_square + 2.0 * velocity_error
    return loss_3d_pos + loss_diff


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-mmpose-type")
    parser.add_argument("--train-dataset-name")
    parser.add_argument("--biased", action="store_true")
    parser.add_argument("--window-length", type=int, default=9)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--token-dropout", type=float, default=0.2)
    parser.add_argument("--residual-scale", type=float, default=0.1)
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
    parser.add_argument("--optimizer-steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--effective-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache-frame-rays",
        action="store_true",
        help="precompute each synchronized frame's RUMPL rays once before window sampling",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=0,
        help="thread workers used only while building --cache-frame-rays",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--loss-type", choices=("mse", "mpjpe", "smooth-l1"), default="mse",
        help="3-D supervision; mpjpe matches the retained RUMPL baseline",
    )
    parser.add_argument(
        "--loss-frame", choices=("all", "latest"), default="all",
        help="supervise every output or only the causal window's latest frame",
    )
    parser.add_argument(
        "--loss-profile", choices=("rumpl", "mixste-original"), default="rumpl",
        help=(
            "rumpl uses --loss-type; mixste-original uses the official "
            "MixSTE weighted-MPJPE plus temporal-consistency objective"
        ),
    )
    parser.add_argument(
        "--disable-missing-keypoints", action="store_true",
        help="disable the legacy 20%% random keypoint removal in real-H36M training",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16"
    )
    parser.add_argument("--no-residual-gate", action="store_true")
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="also optimize the retained H76 backbone (default: temporal branch only)",
    )
    parser.add_argument(
        "--backbone-lr-multiplier",
        type=float,
        default=1.0,
        help="learning-rate multiplier for H76 parameters when unfrozen",
    )
    parser.add_argument(
        "--backbone-train-scope",
        choices=("all", "head", "pft-head", "vft", "vft-pft-head"),
        default="all",
        help="subset of retained RUMPL optimized when --unfreeze-backbone is set",
    )
    parser.add_argument(
        "--backbone-eval-mode", action="store_true",
        help="keep retained RUMPL dropout/drop-path disabled while allowing gradients",
    )
    parser.add_argument(
        "--backbone-flavor",
        choices=("h76", "h81"),
        default="h76",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def lr_multiplier(step, warmup_steps, total_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_train_dataset(args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base = eval("dataset." + config.DATASET.TRAIN_DATASET)(
        config,
        config.DATASET.TRAIN_SUBSET,
        True,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    temporal = TemporalH36MRUMPL(
        base,
        window_length=args.window_length,
        frame_stride=args.frame_stride,
        window_step=1,
        cache_frames=args.cache_frame_rays,
        cache_workers=args.cache_workers,
    )
    if args.max_windows > 0:
        temporal = Subset(temporal, range(min(args.max_windows, len(temporal))))
    return temporal


def save_checkpoint(path, model, optimizer, scheduler, scaler, args, step):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "optimizer_step": step,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.optimizer_steps < 1:
        raise ValueError("optimizer-steps must be positive")
    if args.micro_batch_size < 1 or args.effective_batch_size < args.micro_batch_size:
        raise ValueError("invalid micro/effective batch sizes")
    if args.loss_profile == "mixste-original" and args.loss_frame != "all":
        raise ValueError("MixSTE original loss requires --loss-frame all")
    accumulation = math.ceil(args.effective_batch_size / args.micro_batch_size)
    actual_effective_batch = accumulation * args.micro_batch_size

    seed_everything(args.seed)
    update_config(args.cfg)
    os.environ["RUMPL_BACKBONE_FLAVOR"] = args.backbone_flavor
    if args.train_mmpose_type:
        config.DATASET.TRAIN_MMPOSE_TYPE = args.train_mmpose_type
        config.DATASET.USE_MMPOSE_TRAIN = True
    if args.train_dataset_name:
        config.DATASET.TRAIN_H36M_DATASET_NAME = args.train_dataset_name
    n_views_all = os.environ.get("RUMPL_N_VIEWS_TRAIN_TEST_ALL", "").strip()
    if n_views_all:
        config.DATASET.N_VIEWS_TRAIN_TEST_ALL = int(n_views_all)
        config.DATASET.TRAIN_ON_ALL_CAMERAS = True
        config.DATASET.TEST_ON_ALL_CAMERAS = True
    if args.disable_missing_keypoints:
        config.DATASET.APPLY_NOISE_MISSING = False
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_args.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    # Retain the established H76 function exactly.  Earlier H40 code forced
    # Pluecker off here, so loading an H76 checkpoint did not reproduce H76.
    os.environ["RUMPL_TRI_ANCHOR"] = "1"
    os.environ.setdefault("RUMPL_TRI_ANCHOR_REG", "1e-4")
    os.environ.setdefault("RUMPL_TRI_ANCHOR_CONF_EPS", "0.05")
    os.environ["RUMPL_ANCHOR_CENTERED_RAYS"] = "1"
    os.environ["RUMPL_ANCHOR_CENTER_PER_JOINT"] = "0"
    os.environ["RUMPL_INPUT_PLUCKER"] = "1"
    os.environ["RUMPL_INPUT_HARMONIC_L"] = "0"
    if os.environ.get("RUMPL_BACKBONE_FLAVOR", "h76").lower() == "h81":
        os.environ.setdefault("RUMPL_PER_JOINT_RESIDUAL_GATE", "1")
        for name in (
            "GBT_GLOBAL_JV_DEPTH",
            "GBT_LEARNABLE_BIAS",
            "RUMPL_GBT_SET_DECODER",
            "RUMPL_RELATIVE_VIEW_FUSION",
            "RUMPL_GEOMETRY_UNCERTAINTY_TOKEN",
            "RUMPL_POST_PFT_GRAPH_RESIDUAL",
            "RUMPL_JOINT_SPECIFIC_HEAD",
        ):
            os.environ.setdefault(name, "0")
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

    device = torch.device(args.device)
    base_model = get_multiview_rumpl_net(config, is_train=False)
    base_state = clean_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    from utils.rumpl_checkpoint_adapt import merge_pretrained_into_model_state

    merged, skipped = merge_pretrained_into_model_state(
        base_model.state_dict(), base_state, strict_shapes=False
    )
    if skipped:
        print(f"[H40] base load skipped keys: {skipped[:20]}", flush=True)
    incompatible = base_model.load_state_dict(merged, strict=False)
    if incompatible.missing_keys:
        critical = [
            k for k in incompatible.missing_keys
            if not k.startswith("features.weighted_mean")
        ]
        if critical:
            raise RuntimeError(
                f"H0 checkpoint missing critical keys: {critical[:10]}"
            )
    # Reset before constructing the new branch so T0/T1 share every common
    # initialization draw.  The biased variant only adds its two scale scalars.
    torch.manual_seed(args.seed + 1000)
    model = TemporalJointViewRUMPL(
        base_model,
        depth=args.depth,
        num_heads=args.heads,
        biased=args.biased,
        token_dropout=args.token_dropout,
        residual_gate=not args.no_residual_gate,
        residual_scale=args.residual_scale,
        fusion_mode=args.fusion_mode,
        temporal_length=args.window_length,
    ).to(device)
    if not args.unfreeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    elif args.backbone_train_scope != "all":
        scope_prefixes = {
            "head": ("head.",),
            "pft-head": ("Spatial_pos_embed", "blocks.", "Spatial_norm.", "head."),
            "vft": (
                "fusion_token", "blocks_view_fusion.", "View_norm.",
                "encoding_to_embedding.", "confidence_to_embedding.",
            ),
            "vft-pft-head": (
                "fusion_token", "blocks_view_fusion.", "View_norm.",
                "Spatial_pos_embed", "blocks.", "Spatial_norm.", "head.",
            ),
        }[args.backbone_train_scope]
        for name, parameter in model.backbone.named_parameters():
            parameter.requires_grad_(name.startswith(scope_prefixes))

    train_dataset = build_train_dataset(args)
    generator = torch.Generator().manual_seed(args.seed)
    # Keep camera-subset draws identical across architecture/dropout arms.
    # Dropout consumes the model RNG, so sharing it with view sampling would
    # silently confound a dropout ablation with different camera pairs.
    view_generator = torch.Generator(device=device).manual_seed(args.seed + 2000)
    loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_temporal_h36m,
        generator=generator,
    )
    if args.unfreeze_backbone:
        if not 0.0 < args.backbone_lr_multiplier <= 1.0:
            raise ValueError("backbone-lr-multiplier must be in (0, 1]")
        backbone_parameters = [
            parameter for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ]
        backbone_ids = {id(parameter) for parameter in backbone_parameters}
        temporal_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_ids
        ]
        optimized_parameters = [
            {"params": temporal_parameters, "lr": args.lr},
            {
                "params": backbone_parameters,
                "lr": args.lr * args.backbone_lr_multiplier,
            },
        ]
    else:
        optimized_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
    optimizer = torch.optim.Adam(
        optimized_parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(step, args.warmup_steps, args.optimizer_steps),
    )
    amp_enabled = args.amp_dtype != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    # BF16 has FP32-like exponent range and does not require loss scaling.
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled and args.amp_dtype == "fp16"
    )
    start_step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_step = int(resume["optimizer_step"])

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"[H40] biased={int(args.biased)} windows={len(train_dataset)} "
        f"T={args.window_length} V={args.num_views} depth={args.depth} "
        f"dropout={args.token_dropout} residual_gate={int(not args.no_residual_gate)} "
        f"frozen_backbone={int(not args.unfreeze_backbone)} "
        f"backbone_lr_multiplier={args.backbone_lr_multiplier:g} "
        f"backbone_scope={args.backbone_train_scope} "
        f"backbone_eval_mode={int(args.backbone_eval_mode)} "
        f"missing_keypoints={int(not args.disable_missing_keypoints)} "
        f"loss={args.loss_profile}:{args.loss_type}/{args.loss_frame} "
        f"fusion_mode={args.fusion_mode} "
        f"residual_scale={args.residual_scale:g} "
        f"amp={args.amp_dtype} trainable={trainable / 1e6:.2f}M",
        flush=True,
    )
    print(
        f"[H40] micro_batch={args.micro_batch_size} accumulation={accumulation} "
        f"effective_batch={actual_effective_batch} steps={args.optimizer_steps}",
        flush=True,
    )

    model.train()
    if not args.unfreeze_backbone or args.backbone_eval_mode:
        # Frozen RUMPL must not inject dropout/stochastic-depth noise while the
        # temporal residual learns.  The new temporal blocks remain in train mode.
        model.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    running_loss = 0.0
    running_mpjpe = 0.0
    running_micro = 0
    wall_start = time.time()
    for optimizer_step in range(start_step, args.optimizer_steps):
        for micro_step in range(accumulation):
            try:
                _, _, target, rays, _, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                _, _, target, rays, _, _ = next(iterator)
            rays = rays.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                prediction, _ = model(
                    rays,
                    num_views=args.num_views,
                    view_generator=view_generator,
                )
                loss_prediction = prediction
                loss_target = target
                if args.loss_profile == "mixste-original":
                    loss = mixste_original_loss(prediction, target)
                else:
                    if args.loss_frame == "latest":
                        loss_prediction = loss_prediction[:, -1]
                        loss_target = loss_target[:, -1]
                    if args.loss_type == "mse":
                        loss = torch.nn.functional.mse_loss(
                            loss_prediction, loss_target
                        )
                    elif args.loss_type == "mpjpe":
                        loss = (loss_prediction - loss_target).norm(dim=-1).mean()
                    else:
                        loss = torch.nn.functional.smooth_l1_loss(
                            loss_prediction, loss_target, beta=0.01
                        )
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at optimizer step {optimizer_step}, micro {micro_step}"
                )
            scaler.scale(scaled_loss).backward()
            with torch.no_grad():
                mpjpe_mm = (
                    (loss_prediction - loss_target).norm(dim=-1).mean() * 1000.0
                )
            running_loss += float(loss.detach())
            running_mpjpe += float(mpjpe_mm.detach())
            running_micro += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        completed = optimizer_step + 1

        if completed % args.log_every == 0 or completed == 1:
            elapsed = time.time() - wall_start
            print(
                f"[H40] step={completed}/{args.optimizer_steps} "
                f"mse={running_loss / running_micro:.8f} "
                f"train_mpjpe_mm={running_mpjpe / running_micro:.3f} "
                f"lr={optimizer.param_groups[0]['lr']:.8g} "
                f"gate={float(model.global_gate.detach()) if hasattr(model, 'global_gate') else 1.0:.6f} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )
            running_loss = running_mpjpe = 0.0
            running_micro = 0
            wall_start = time.time()
        if completed % args.save_every == 0 or completed == args.optimizer_steps:
            save_checkpoint(
                output_dir / f"checkpoint_step_{completed:07d}.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                completed,
            )


if __name__ == "__main__":
    main()
