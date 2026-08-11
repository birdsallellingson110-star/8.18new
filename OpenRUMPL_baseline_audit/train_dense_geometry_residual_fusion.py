#!/usr/bin/env python3
"""Train dense geometry residual fusion on real Human3.6M heatmaps."""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from dense_geometry_residual_fusion import DenseGeometryResidualFusion
from eval_h36m_dense_epipolar_heatmaps import (
    DenseHeatmapStore,
    epipolar_support,
)
from eval_h36m_sparse_epipolar_topk import (
    COCO_TO_H36M_DIRECT,
    build_four_view_groups,
    camera_parameters,
)


DIRECT_COCO = np.asarray(
    list(COCO_TO_H36M_DIRECT.keys()), dtype=np.int64
)
DIRECT_H36M = np.asarray(
    list(COCO_TO_H36M_DIRECT.values()), dtype=np.int64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--depth-min-m", type=float, default=1.0)
    parser.add_argument("--depth-max-m", type=float, default=10.0)
    parser.add_argument("--depth-samples", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-case-scale", type=float, default=8.0)
    parser.add_argument("--maximum-hard-weight", type=float, default=5.0)
    parser.add_argument(
        "--view-probabilities",
        type=float,
        nargs=3,
        metavar=("P_V2", "P_V3", "P_V4"),
        default=[1.0, 1.0, 1.0],
        help="Relative sampling probabilities for V2, V3 and V4.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def image_to_heatmap(
    image_xy: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    size = np.asarray([width, height], dtype=np.float32)
    return (image_xy - center[:, None] + 0.5 * scale[:, None]) / (
        scale[:, None]
    ) * size


def bilinear_coordinate_nll(
    logits: torch.Tensor,
    target_xy: torch.Tensor,
    hard_case_scale: float,
    maximum_hard_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Bilinear spatial NLL, weighted toward detector failure cases."""
    n_views, n_joints, height, width = logits.shape
    x = target_xy[..., 0]
    y = target_xy[..., 1]
    valid = (
        (x >= 0)
        & (x <= width - 1)
        & (y >= 0)
        & (y <= height - 1)
    )
    x_safe = x.clamp(0, width - 1)
    y_safe = y.clamp(0, height - 1)
    x0 = torch.floor(x_safe).long()
    y0 = torch.floor(y_safe).long()
    x1 = (x0 + 1).clamp(max=width - 1)
    y1 = (y0 + 1).clamp(max=height - 1)
    wx = x_safe - x0.float()
    wy = y_safe - y0.float()
    indices = torch.stack(
        (
            y0 * width + x0,
            y0 * width + x1,
            y1 * width + x0,
            y1 * width + x1,
        ),
        dim=-1,
    )
    weights = torch.stack(
        (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        ),
        dim=-1,
    )
    log_probability = torch.log_softmax(logits.flatten(-2), dim=-1)
    target_log_probability = (
        log_probability.gather(-1, indices) * weights
    ).sum(-1)

    peak_index = logits.detach().flatten(-2).argmax(-1)
    baseline_peak = torch.stack(
        (
            torch.remainder(peak_index, width),
            torch.div(peak_index, width, rounding_mode="floor"),
        ),
        dim=-1,
    )
    baseline_distance = torch.linalg.norm(
        baseline_peak - target_xy, dim=-1
    )
    hard_weight = 1.0 + torch.clamp(
        baseline_distance / hard_case_scale,
        max=maximum_hard_weight - 1.0,
    )
    effective = valid.float() * hard_weight
    loss = -(target_log_probability * effective).sum() / effective.sum(
    ).clamp_min(1.0)
    metrics = {
        "loss": float(loss.detach()),
        "valid_joints": float(valid.sum()),
        "mean_target_distance_hm": float(
            baseline_distance[valid].mean() if valid.any() else 0.0
        ),
        "mean_hard_weight": float(
            hard_weight[valid].mean() if valid.any() else 0.0
        ),
    }
    return loss, metrics


def checkpoint_payload(
    model: DenseGeometryResidualFusion,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> dict:
    return {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "direct_coco_joints": DIRECT_COCO.tolist(),
        "direct_h36m_joints": DIRECT_H36M.tolist(),
    }


def atomic_torch_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    if min(args.view_probabilities) < 0 or sum(args.view_probabilities) <= 0:
        raise ValueError("--view-probabilities must be non-negative and nonzero")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"

    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    store = DenseHeatmapStore(args.dense_shards)
    groups = [
        group
        for group in build_four_view_groups(records)
        if all(index in store for index in group)
    ]
    if not groups:
        raise RuntimeError("dense shards contain no complete training groups")
    model = DenseGeometryResidualFusion().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    first_step = 1
    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        first_step = int(payload["step"]) + 1

    depths = torch.linspace(
        args.depth_min_m,
        args.depth_max_m,
        args.depth_samples,
        device=device,
    )
    joint_ids = torch.as_tensor(
        DIRECT_COCO, dtype=torch.long, device=device
    )
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log_handle:
        for step in range(first_step, args.steps + 1):
            group = random.choice(groups)
            n_views = random.choices(
                (2, 3, 4), weights=args.view_probabilities, k=1
            )[0]
            selected_cameras = random.choice(
                list(itertools.combinations(range(4), n_views))
            )
            indices = [group[index] for index in selected_cameras]
            group_records = [records[index] for index in indices]
            data = store.get(indices)
            heatmaps = torch.as_tensor(
                data["heatmaps"][:, DIRECT_COCO],
                dtype=torch.float32,
                device=device,
            ).clamp_min_(0.0)
            maximum = heatmaps.flatten(-2).amax(
                -1, keepdim=True
            )[..., None]
            heatmaps = heatmaps / maximum.clamp_min(1e-6)
            camera_data = [
                camera_parameters(record) for record in group_records
            ]
            intrinsics = [item[0] for item in camera_data]
            rotations = [item[1] for item in camera_data]
            centers = np.stack([item[2] for item in camera_data])
            with torch.no_grad():
                support = epipolar_support(
                    heatmaps,
                    intrinsics,
                    rotations,
                    centers,
                    data["input_center"],
                    data["input_scale"],
                    depths,
                )

            logits, auxiliary = model(
                heatmaps, support, joint_ids=joint_ids
            )
            _, _, height, width = logits.shape
            target_image = np.stack(
                [
                    np.asarray(record["joints_2d"], dtype=np.float32)[
                        DIRECT_H36M
                    ]
                    for record in group_records
                ]
            )
            target_heatmap = image_to_heatmap(
                target_image,
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )
            target = torch.as_tensor(
                target_heatmap, dtype=torch.float32, device=device
            )
            loss, metrics = bilinear_coordinate_nll(
                logits,
                target,
                args.hard_case_scale,
                args.maximum_hard_weight,
            )
            # Keep the identity residual small unless supervision supports a
            # change; this is especially important for already-good V3/V4.
            residual_penalty = auxiliary["spatial_residual"].square().mean()
            loss = loss + 1e-4 * residual_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()

            if step % args.log_every == 0 or step == first_step:
                row = {
                    "step": step,
                    "views": n_views,
                    **metrics,
                    "total_loss": float(loss.detach()),
                    "mean_geometry_weight": float(
                        auxiliary["geometry_weight"].detach().mean()
                    ),
                    "global_geometry_strength": float(
                        model.global_geometry_strength.detach()
                    ),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.time() - started,
                }
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
            if step % args.save_every == 0:
                atomic_torch_save(
                    checkpoint_payload(model, optimizer, step, args),
                    output_dir / f"checkpoint_step{step:06d}.pth",
                )

    atomic_torch_save(
        checkpoint_payload(model, optimizer, args.steps, args),
        output_dir / "final.pth",
    )


if __name__ == "__main__":
    main()
