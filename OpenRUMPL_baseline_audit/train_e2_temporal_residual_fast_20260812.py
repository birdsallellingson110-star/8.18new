#!/usr/bin/env python3
"""Vectorized batch trainer for the E-2 temporal residual experiments.

It is numerically the same as ``train_e2_temporal_residual_20260812.py`` but
uses paired NumPy indexing on the pre-expanded window memmap.  The previous
DataLoader path spent most of its time constructing 384k individual Python
samples; this path keeps that exact sample order/protocol while batching it in
one C-level gather.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_e2_temporal_residual_20260812 import (
    COMBINATIONS,
    STAGES,
    build_model,
    parse_task_ids,
    stage_action_equal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-window-cache", required=True)
    p.add_argument("--train-index", required=True)
    p.add_argument("--validation-window-cache", required=True)
    p.add_argument("--validation-index", required=True)
    p.add_argument("--architecture", choices=("fixedlag", "mixste"), required=True)
    p.add_argument("--task-ids", default="v3v4")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-validation-windows", type=int, default=0)
    p.add_argument(
        "--holdout-subject", type=int, default=0,
        help="hold out a complete train subject for selection; 0 keeps the legacy modulo-10 control",
    )
    return p.parse_args()


class WindowArrays:
    def __init__(self, cache_dir, index_path, task_ids, max_windows=0):
        cache_dir = Path(cache_dir)
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        if manifest["task_ids"] != task_ids.tolist():
            raise ValueError(
                f"memmap task ids {manifest['task_ids']} != {task_ids.tolist()}"
            )
        self.poses = np.load(cache_dir / "window_poses.npy", mmap_mode="r")
        self.targets = np.load(cache_dir / "window_targets.npy", mmap_mode="r")
        index = np.load(index_path)
        self.windows = index["window_indices"]
        self.actions = index["actions"].astype(np.int16)
        self.subjects = index["subjects"].astype(np.int16)
        self.center_group_indices = index["center_group_indices"].astype(np.int64)
        if max_windows:
            self.windows = self.windows[:max_windows]
            self.actions = self.actions[:max_windows]
            self.subjects = self.subjects[:max_windows]
            self.center_group_indices = self.center_group_indices[:max_windows]
        self.task_ids = task_ids
        self.window_count = len(self.windows)

    def gather(self, sample_indices):
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        task_positions = sample_indices // self.window_count
        window_positions = sample_indices % self.window_count
        # Paired advanced indexing returns B,T,J,3, not a CxB Cartesian product.
        poses = np.asarray(self.poses[task_positions, window_positions], dtype=np.float32)
        targets = np.asarray(self.targets[window_positions], dtype=np.float32)
        task_ids = self.task_ids[task_positions]
        actions = self.actions[window_positions]
        return poses, targets, task_ids, actions


@torch.inference_mode()
def evaluate(model, arrays, indices, device, batch_size):
    model.eval()
    stores = {int(task): {"err": [], "base": [], "actions": []} for task in arrays.task_ids}
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        poses, targets, task_ids, actions = arrays.gather(batch_indices)
        poses_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
        targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
        task_t = torch.from_numpy(task_ids).to(device=device, dtype=torch.long)
        prediction = model(poses_t, task_t)
        err = torch.linalg.vector_norm(prediction - targets_t, dim=-1).cpu().numpy()
        base = np.linalg.norm(
            poses[:, poses.shape[1] // 2] - targets, axis=-1
        )
        for task in arrays.task_ids.tolist():
            mask = task_ids == task
            if np.any(mask):
                stores[int(task)]["err"].append(err[mask])
                stores[int(task)]["base"].append(base[mask])
                stores[int(task)]["actions"].append(actions[mask])
    result = {"tasks": {}, "stages": {}}
    for task in arrays.task_ids.tolist():
        store = stores[int(task)]
        err = np.concatenate(store["err"], axis=0)
        base = np.concatenate(store["base"], axis=0)
        actions = np.concatenate(store["actions"], axis=0)
        result["tasks"][str(COMBINATIONS[task])] = {
            "temporal_action_equal_all17_mm": stage_action_equal(err, actions),
            "center_baseline_action_equal_all17_mm": stage_action_equal(base, actions),
            "frame_weighted_temporal_all17_mm": float(err.mean() * 1000.0),
            "delta_mm": float((err.mean() - base.mean()) * 1000.0),
        }
    active = set(arrays.task_ids.tolist())
    for stage, task_range in STAGES.items():
        vals = [
            result["tasks"][str(COMBINATIONS[i])]
            for i in task_range if i in active
        ]
        if not vals:
            continue
        result["stages"][stage] = {
            "temporal_action_equal_all17_mm": float(np.mean([
                v["temporal_action_equal_all17_mm"] for v in vals
            ])),
            "center_baseline_action_equal_all17_mm": float(np.mean([
                v["center_baseline_action_equal_all17_mm"] for v in vals
            ])),
            "delta_mm": float(np.mean([v["delta_mm"] for v in vals])),
        }
    return result


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    task_ids = parse_task_ids(args.task_ids)
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train = WindowArrays(
        args.train_window_cache, args.train_index, task_ids,
        args.max_train_windows,
    )
    validation = WindowArrays(
        args.validation_window_cache, args.validation_index, task_ids,
        args.max_validation_windows,
    )
    mean = torch.from_numpy(np.asarray(train.targets[:train.window_count]).mean(axis=0)).float()
    std = torch.from_numpy(np.asarray(train.targets[:train.window_count]).std(axis=0)).float()
    model = build_model(
        args.architecture, mean, std, train.poses.shape[2],
        args.hidden_dim, args.layers,
    ).to(device)
    total_windows = train.window_count
    total_samples = total_windows * len(task_ids)
    all_indices = np.arange(total_samples, dtype=np.int64)
    if args.holdout_subject:
        if args.holdout_subject not in set(train.subjects.tolist()):
            raise ValueError(
                f"holdout subject {args.holdout_subject} is absent from training windows"
            )
        holdout_windows = train.subjects == args.holdout_subject
        holdout_protocol = f"leave-subject-{args.holdout_subject}-out"
    else:
        holdout_windows = train.center_group_indices % 10 == 0
        holdout_protocol = "center-group-index-modulo-10"
    sample_windows = all_indices % total_windows
    train_indices = all_indices[~holdout_windows[sample_windows]]
    holdout_indices = all_indices[holdout_windows[sample_windows]]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    checkpoint = out_dir / "model_best.pth.tar"
    best_metric = math.inf
    best_epoch = -1
    history = []
    active_stages = [
        stage for stage, task_range in STAGES.items()
        if any(i in set(task_ids.tolist()) for i in task_range)
    ]
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        order = train_indices.copy()
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            poses, targets, task_batch, _ = train.gather(order[start : start + args.batch_size])
            poses_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
            targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
            task_t = torch.from_numpy(task_batch).to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(poses_t, task_t)
            loss_abs = F.smooth_l1_loss(prediction, targets_t, beta=0.01)
            loss_rel = F.smooth_l1_loss(
                prediction - prediction[:, :1],
                targets_t - targets_t[:, :1], beta=0.01,
            )
            loss = loss_abs + 0.25 * loss_rel
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout = evaluate(model, train, holdout_indices, device, args.batch_size)
        metric = float(np.mean([
            holdout["stages"][stage]["temporal_action_equal_all17_mm"]
            for stage in active_stages
        ]))
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": metric, "holdout": holdout,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "architecture": args.architecture, "task_ids": task_ids.tolist(),
                "window_length": int(train.poses.shape[2]),
                "hidden_dim": args.hidden_dim, "layers": args.layers,
                "epoch": epoch,
            }, checkpoint)
    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    final = evaluate(
        model, validation,
        np.arange(validation.window_count * len(task_ids), dtype=np.int64),
        device, args.batch_size,
    )
    result = {
        "method": (
            "E-2 fixed-lag temporal residual (vectorized batches)"
            if args.architecture == "fixedlag"
            else "E-2 MixSTE-style factorized temporal residual (vectorized batches)"
        ),
        "task_ids": task_ids.tolist(),
        "protocol": {
            "window_length": int(train.poses.shape[2]), "frame_stride": 5,
            "root_protected": True, "center_frame_target": True,
            "train_windows": int(train.window_count),
            "internal_holdout_windows": int(holdout_windows.sum()),
            "holdout_protocol": holdout_protocol,
            "validation_windows": int(validation.window_count),
        },
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": final,
        "args": vars(args),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"S9_S11_final_once": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
