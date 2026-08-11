#!/usr/bin/env python3
"""Train the H21 HeatFormer-inspired keypoint query refiner."""

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
import torch.nn.functional as functional

from eval_h36m_dense_epipolar_heatmaps import DenseHeatmapStore
from eval_h36m_sparse_epipolar_topk import (
    COCO_TO_H36M_DIRECT,
    build_four_view_groups,
)
from iterative_pose_query_refiner import IterativePoseQueryRefiner
from pose_query_geometry import (
    image_to_heatmap,
    project_world_points,
    triangulate_points,
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
    parser.add_argument(
        "--seed-mmpose-pkl",
        help=(
            "Optional aligned MMPose-format PKL whose joints_2d are used as "
            "the detector seed. This isolates training H21 on the same A1D "
            "input distribution used by H35 export."
        ),
    )
    parser.add_argument("--dense-shards", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--view-probabilities",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
    )
    parser.add_argument("--hard-case-pixels", type=float, default=4.0)
    parser.add_argument("--maximum-hard-weight", type=float, default=5.0)
    parser.add_argument("--delta-penalty", type=float, default=1e-3)
    parser.add_argument("--irls-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def checkpoint_payload(
    model: IterativePoseQueryRefiner,
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


def atomic_save(payload: dict, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    seed_records = None
    if args.seed_mmpose_pkl:
        with open(args.seed_mmpose_pkl, "rb") as handle:
            seed_records = pickle.load(handle)
        if len(seed_records) != len(records):
            raise ValueError("seed MMPose PKL length differs from GT PKL")
        for index, (record, seed_record) in enumerate(zip(records, seed_records)):
            if record["image"] != seed_record["image"]:
                raise ValueError(f"seed MMPose image mismatch at record {index}")
    store = DenseHeatmapStore(args.dense_shards)
    groups = [
        group
        for group in build_four_view_groups(records)
        if all(index in store for index in group)
    ]
    if not groups:
        raise RuntimeError("no complete four-view groups in dense export")

    model = IterativePoseQueryRefiner().to(device)
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
    joint_ids = torch.as_tensor(
        DIRECT_COCO, dtype=torch.long, device=device
    )
    started = time.time()
    log_path = output_dir / "train.jsonl"
    with log_path.open("a", encoding="utf-8") as log_handle:
        for step in range(first_step, args.steps + 1):
            group = random.choice(groups)
            n_views = random.choices(
                (2, 3, 4), weights=args.view_probabilities, k=1
            )[0]
            combination = random.choice(
                list(itertools.combinations(range(4), n_views))
            )
            indices = [group[index] for index in combination]
            group_records = [records[index] for index in indices]
            data = store.get(indices)
            heatmaps_np = data["heatmaps"][:, DIRECT_COCO].astype(
                np.float32
            )
            _, _, height, width = heatmaps_np.shape
            if seed_records is None:
                detector_image = data["decoded_keypoints"][:, DIRECT_COCO].astype(
                    np.float64
                )
                confidence_np = data["decoded_scores"][:, DIRECT_COCO].astype(
                    np.float64
                )
            else:
                selected_seed_records = [seed_records[index] for index in indices]
                detector_image = np.stack(
                    [
                        np.asarray(item["joints_2d"], dtype=np.float64)[DIRECT_H36M]
                        for item in selected_seed_records
                    ]
                )
                confidence_np = np.stack(
                    [
                        np.asarray(item["joints_2d_conf"], dtype=np.float64)
                        .reshape(17)[DIRECT_H36M]
                        for item in selected_seed_records
                    ]
                )
            detector_hm = image_to_heatmap(
                detector_image,
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )
            anchor = triangulate_points(
                group_records,
                detector_image,
                confidence_np,
                args.irls_iterations,
            )
            query_image = project_world_points(group_records, anchor)
            query_hm = image_to_heatmap(
                query_image,
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )
            target_image = np.stack(
                [
                    np.asarray(record["joints_2d"], dtype=np.float64)[
                        DIRECT_H36M
                    ]
                    for record in group_records
                ]
            )
            target_hm = image_to_heatmap(
                target_image,
                data["input_center"],
                data["input_scale"],
                width,
                height,
            )

            heatmaps = torch.as_tensor(
                heatmaps_np, dtype=torch.float32, device=device
            )
            detector = torch.as_tensor(
                detector_hm, dtype=torch.float32, device=device
            )
            query = torch.as_tensor(
                query_hm, dtype=torch.float32, device=device
            )
            confidence = torch.as_tensor(
                confidence_np, dtype=torch.float32, device=device
            )
            target = torch.as_tensor(
                target_hm, dtype=torch.float32, device=device
            )
            prediction, auxiliary = model(
                heatmaps, query, detector, confidence, joint_ids
            )
            valid = (
                (target[..., 0] >= 0)
                & (target[..., 0] <= width - 1)
                & (target[..., 1] >= 0)
                & (target[..., 1] <= height - 1)
            )
            detector_error = torch.linalg.norm(
                detector - target, dim=-1
            )
            hard_weight = 1.0 + torch.clamp(
                detector_error / args.hard_case_pixels,
                max=args.maximum_hard_weight - 1.0,
            )
            coordinate_loss = functional.smooth_l1_loss(
                prediction, target, reduction="none"
            ).mean(dim=-1)
            effective = valid.float() * hard_weight
            coordinate_loss = (
                coordinate_loss * effective
            ).sum() / effective.sum().clamp_min(1.0)
            delta_penalty = auxiliary["delta"].square().mean()
            loss = coordinate_loss + args.delta_penalty * delta_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )
            optimizer.step()

            if step % args.log_every == 0 or step == first_step:
                refined_error = torch.linalg.norm(
                    prediction.detach() - target, dim=-1
                )
                row = {
                    "step": step,
                    "views": n_views,
                    "loss": float(loss.detach()),
                    "coordinate_loss": float(coordinate_loss.detach()),
                    "delta_penalty": float(delta_penalty.detach()),
                    "detector_error_hm": float(
                        detector_error[valid].mean()
                    ),
                    "refined_error_hm": float(
                        refined_error[valid].mean()
                    ),
                    "mean_delta_hm": float(
                        torch.linalg.norm(
                            auxiliary["delta"].detach(), dim=-1
                        ).mean()
                    ),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.time() - started,
                }
                print(json.dumps(row), flush=True)
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
            if step % args.save_every == 0:
                atomic_save(
                    checkpoint_payload(model, optimizer, step, args),
                    output_dir / f"checkpoint_step{step:06d}.pth",
                )
    atomic_save(
        checkpoint_payload(model, optimizer, args.steps, args),
        output_dir / "final.pth",
    )


if __name__ == "__main__":
    main()
