#!/usr/bin/env python3
"""Train sparse epipolar candidate fusion on real H36M subjects.

The frozen HRNet and its top-K modes are inputs; only the small
camera-identity-free candidate Transformer is optimized.  Supervision is the
nearest heatmap candidate to the real H36M 2D annotation.  S9/S11 are not used
here and remain the untouched test set.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from sparse_epipolar_candidate_transformer import (
    SparseEpipolarCandidateTransformer,
    candidate_loss,
)


COCO_JOINTS = np.asarray(
    [11, 13, 15, 12, 14, 16, 0, 5, 7, 9, 6, 8, 10], dtype=np.int64
)
H36M_JOINTS = np.asarray(
    [1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16], dtype=np.int64
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pkl", required=True)
    parser.add_argument("--topk-shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-groups", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--val-stride", type=int, default=20)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--hard-case-weight",
        type=float,
        default=8.0,
        help="Weight when the monocular top-1 is not the GT-nearest mode.",
    )
    return parser.parse_args()


def frame_key(record: dict) -> tuple[int, int, int, int]:
    return (
        int(record["subject"]),
        int(record["action"]),
        int(record["subaction"]),
        int(record["image_id"]),
    )


def build_groups(records: list[dict]) -> np.ndarray:
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    for index, record in enumerate(records):
        key = frame_key(record)
        grouped.setdefault(key, [-1, -1, -1, -1])
        grouped[key][int(record["camera_id"])] = index
    groups = [group for group in grouped.values() if min(group) >= 0]
    return np.asarray(groups, dtype=np.int64)


def camera_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intrinsic_inverse = np.empty((len(records), 3, 3), dtype=np.float32)
    rotations = np.empty((len(records), 3, 3), dtype=np.float32)
    centers = np.empty((len(records), 3), dtype=np.float32)
    for index, record in enumerate(records):
        camera = record["camera"]
        intrinsic = np.asarray(
            [
                [
                    float(np.asarray(camera["fx"]).reshape(-1)[0]),
                    0.0,
                    float(np.asarray(camera["cx"]).reshape(-1)[0]),
                ],
                [
                    0.0,
                    float(np.asarray(camera["fy"]).reshape(-1)[0]),
                    float(np.asarray(camera["cy"]).reshape(-1)[0]),
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        intrinsic_inverse[index] = np.linalg.inv(intrinsic)
        rotations[index] = np.asarray(camera["R"], dtype=np.float32).reshape(3, 3)
        centers[index] = (
            np.asarray(camera["T"], dtype=np.float32).reshape(3) / 1000.0
        )
    return intrinsic_inverse, rotations, centers


def load_candidates(
    paths: list[str], n_records: int
) -> tuple[np.ndarray, np.ndarray]:
    merged_xy = None
    merged_scores = None
    seen = np.zeros(n_records, dtype=bool)
    for path in paths:
        with np.load(path) as shard:
            indices = shard["record_indices"].astype(np.int64)
            xy = shard["candidate_xy"].astype(np.float32)
            scores = shard["candidate_scores"].astype(np.float32)
        if merged_xy is None:
            merged_xy = np.empty((n_records, *xy.shape[1:]), dtype=np.float32)
            merged_scores = np.empty(
                (n_records, *scores.shape[1:]), dtype=np.float32
            )
        if seen[indices].any():
            raise RuntimeError(f"duplicate top-K record in {path}")
        merged_xy[indices] = xy
        merged_scores[indices] = scores
        seen[indices] = True
    if not seen.all():
        raise RuntimeError(
            f"top-K records incomplete: {int(seen.sum())}/{n_records}"
        )
    return merged_xy, merged_scores


def choose_records(
    groups: np.ndarray, n_views: int, rng: np.random.Generator
) -> np.ndarray:
    view_indices = np.stack(
        [rng.choice(4, n_views, replace=False) for _ in range(len(groups))]
    )
    return np.take_along_axis(groups, view_indices, axis=1)


def make_batch(
    record_indices: np.ndarray,
    candidate_xy: np.ndarray,
    candidate_scores: np.ndarray,
    gt_2d: np.ndarray,
    intrinsic_inverse: np.ndarray,
    rotations: np.ndarray,
    camera_centers: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    xy = candidate_xy[record_indices][:, :, COCO_JOINTS]
    scores = candidate_scores[record_indices][:, :, COCO_JOINTS]
    target_xy = gt_2d[record_indices][:, :, H36M_JOINTS]
    # G,V,J,K,* -> G,J,V,K,*
    xy = np.transpose(xy, (0, 2, 1, 3, 4))
    scores = np.transpose(scores, (0, 2, 1, 3))
    target_xy = np.transpose(target_xy, (0, 2, 1, 3))
    squared_error = np.square(xy - target_xy[:, :, :, None]).sum(axis=-1)
    targets = squared_error.argmin(axis=-1)

    xy_tensor = torch.as_tensor(xy, dtype=torch.float32, device=device)
    ones = torch.ones((*xy_tensor.shape[:-1], 1), device=device)
    homogeneous = torch.cat((xy_tensor, ones), dim=-1)
    inv_k = torch.as_tensor(
        intrinsic_inverse[record_indices], dtype=torch.float32, device=device
    )
    rotation = torch.as_tensor(
        rotations[record_indices], dtype=torch.float32, device=device
    )
    # Matrices are G,V,... while candidates are G,J,V,K,... .
    homogeneous_gv = homogeneous.permute(0, 2, 1, 3, 4)
    camera_ray = torch.matmul(
        inv_k[:, :, None, None], homogeneous_gv.unsqueeze(-1)
    ).squeeze(-1)
    world_ray = torch.matmul(
        rotation.transpose(-1, -2)[:, :, None, None],
        camera_ray.unsqueeze(-1),
    ).squeeze(-1)
    world_ray = torch.nn.functional.normalize(world_ray, dim=-1)
    world_ray = world_ray.permute(0, 2, 1, 3, 4)

    n_groups, n_joints, n_views, topk = scores.shape
    centers = torch.as_tensor(
        camera_centers[record_indices], dtype=torch.float32, device=device
    )
    centers = centers[:, None].expand(-1, n_joints, -1, -1)
    joint_ids = torch.as_tensor(H36M_JOINTS, device=device)
    joint_ids = joint_ids[None].expand(n_groups, -1)
    return {
        "scores": torch.as_tensor(scores, dtype=torch.float32, device=device)
        .reshape(n_groups * n_joints, n_views, topk),
        "centers": centers.reshape(n_groups * n_joints, n_views, 3),
        "directions": world_ray.reshape(
            n_groups * n_joints, n_views, topk, 3
        ),
        "joint_ids": joint_ids.reshape(-1),
        "targets": torch.as_tensor(targets, dtype=torch.long, device=device)
        .reshape(n_groups * n_joints, n_views),
        "candidate_xy": xy_tensor.reshape(
            n_groups * n_joints, n_views, topk, 2
        ),
        "target_xy": torch.as_tensor(
            target_xy, dtype=torch.float32, device=device
        ).reshape(n_groups * n_joints, n_views, 2),
    }


@torch.no_grad()
def candidate_metrics(logits: torch.Tensor, batch: dict) -> dict[str, float]:
    selected = logits.argmax(dim=-1)
    baseline = torch.zeros_like(selected)
    gather_index = selected[..., None, None].expand(-1, -1, 1, 2)
    baseline_index = baseline[..., None, None].expand(-1, -1, 1, 2)
    selected_xy = batch["candidate_xy"].gather(2, gather_index).squeeze(2)
    baseline_xy = batch["candidate_xy"].gather(
        2, baseline_index
    ).squeeze(2)
    selected_error = torch.linalg.vector_norm(
        selected_xy - batch["target_xy"], dim=-1
    )
    baseline_error = torch.linalg.vector_norm(
        baseline_xy - batch["target_xy"], dim=-1
    )
    return {
        "selected_correct": float(
            (selected == batch["targets"]).float().sum().item()
        ),
        "baseline_correct": float(
            (baseline == batch["targets"]).float().sum().item()
        ),
        "selected_error_sum": float(selected_error.sum().item()),
        "baseline_error_sum": float(baseline_error.sum().item()),
        "count": float(selected.numel()),
    }


def run_epoch(
    model: SparseEpipolarCandidateTransformer,
    group_indices: np.ndarray,
    groups: np.ndarray,
    arrays: dict,
    device: torch.device,
    batch_groups: int,
    rng: np.random.Generator,
    optimizer: torch.optim.Optimizer | None,
    hard_case_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    order = group_indices.copy()
    if training:
        rng.shuffle(order)
    totals = {
        "loss_sum": 0.0,
        "batches": 0.0,
        "selected_correct": 0.0,
        "baseline_correct": 0.0,
        "selected_error_sum": 0.0,
        "baseline_error_sum": 0.0,
        "count": 0.0,
    }
    for start in range(0, len(order), batch_groups):
        selected_groups = groups[order[start : start + batch_groups]]
        n_views = int(rng.integers(2, 5)) if training else 4
        records = choose_records(selected_groups, n_views, rng)
        batch = make_batch(
            records,
            arrays["candidate_xy"],
            arrays["candidate_scores"],
            arrays["gt_2d"],
            arrays["intrinsic_inverse"],
            arrays["rotations"],
            arrays["camera_centers"],
            device,
        )
        with torch.set_grad_enabled(training):
            logits = model(
                batch["scores"],
                batch["centers"],
                batch["directions"],
                batch["joint_ids"],
            )
            correction_weight = torch.where(
                batch["targets"] != 0,
                torch.as_tensor(hard_case_weight, device=device),
                torch.ones((), device=device),
            )
            loss = candidate_loss(
                logits,
                batch["targets"],
                sample_weight=correction_weight,
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        metrics = candidate_metrics(logits.detach(), batch)
        totals["loss_sum"] += float(loss.item())
        totals["batches"] += 1.0
        for key, value in metrics.items():
            totals[key] += value
    return {
        "loss": totals["loss_sum"] / totals["batches"],
        "candidate_accuracy": totals["selected_correct"] / totals["count"],
        "baseline_candidate_accuracy": totals["baseline_correct"]
        / totals["count"],
        "mean_2d_error_px": totals["selected_error_sum"] / totals["count"],
        "baseline_mean_2d_error_px": totals["baseline_error_sum"]
        / totals["count"],
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading real H36M records: {args.input_pkl}", flush=True)
    with open(args.input_pkl, "rb") as handle:
        records = pickle.load(handle)
    groups = build_groups(records)
    print(
        f"records={len(records)} complete_four_view_groups={len(groups)}",
        flush=True,
    )
    candidate_xy, candidate_scores = load_candidates(
        args.topk_shards, len(records)
    )
    intrinsic_inverse, rotations, camera_centers = camera_arrays(records)
    arrays = {
        "candidate_xy": candidate_xy,
        "candidate_scores": candidate_scores,
        "gt_2d": np.stack(
            [np.asarray(record["joints_2d"], np.float32) for record in records]
        ),
        "intrinsic_inverse": intrinsic_inverse,
        "rotations": rotations,
        "camera_centers": camera_centers,
    }

    all_indices = np.arange(len(groups))
    validation_indices = all_indices[:: args.val_stride]
    train_mask = np.ones(len(groups), dtype=bool)
    train_mask[validation_indices] = False
    train_indices = all_indices[train_mask]
    print(
        f"train_groups={len(train_indices)} "
        f"heldout_train_subject_groups={len(validation_indices)}",
        flush=True,
    )

    model = SparseEpipolarCandidateTransformer(
        dim=args.dim, depth=args.depth, num_heads=args.heads
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best_error = float("inf")
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_indices,
            groups,
            arrays,
            device,
            args.batch_groups,
            rng,
            optimizer,
            args.hard_case_weight,
        )
        validation_metrics = run_epoch(
            model,
            validation_indices,
            groups,
            arrays,
            device,
            args.batch_groups,
            np.random.default_rng(args.seed + epoch),
            None,
            1.0,
        )
        scheduler.step()
        entry = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "residual_gate": float(torch.tanh(model.residual_gate).item()),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
            "coco_joints": COCO_JOINTS.tolist(),
            "h36m_joints": H36M_JOINTS.tolist(),
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pth")
        if validation_metrics["mean_2d_error_px"] < best_error:
            best_error = validation_metrics["mean_2d_error_px"]
            torch.save(checkpoint, output_dir / "checkpoint_best.pth")
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
