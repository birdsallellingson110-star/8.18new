#!/usr/bin/env python3
"""Train small, identity-safe temporal post-processors on E-2 frame outputs.

This is deliberately a postprocessor: H76 and the E-2 counterfactual utility
head remain frozen.  The center-frame pose is the identity baseline, the root
is copied unchanged, and the temporal block predicts only a bounded
root-relative residual.  Thus a failed temporal experiment cannot silently
rewrite the absolute translation or the single-frame result.

``fixedlag`` is a joint-wise fixed-lag temporal Transformer.  ``mixste`` is a
small MixSTE-style factorized block that alternates spatial (joint-axis) and
temporal attention.  Both are trained on the official train-subject windows
and touch S9/S11 only for the final report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


COMBINATIONS = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3),
)
STAGES = {"V2": range(0, 6), "V3": range(6, 10), "V4": range(10, 11)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-index", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-index", required=True)
    p.add_argument("--architecture", choices=("fixedlag", "mixste"), required=True)
    p.add_argument(
        "--task-ids", default="all",
        help="all, v2, v3, v4, or comma-separated task indices (0..10)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window-length", type=int, default=9)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-validation-windows", type=int, default=0)
    p.add_argument("--train-window-cache", default=None)
    p.add_argument("--validation-window-cache", default=None)
    p.add_argument("--smoke-batches", type=int, default=0)
    return p.parse_args()


def parse_task_ids(spec: str) -> np.ndarray:
    spec = spec.strip().lower()
    if spec == "all":
        values = list(range(11))
    elif spec == "v2":
        values = list(range(0, 6))
    elif spec == "v3":
        values = list(range(6, 10))
    elif spec == "v4":
        values = [10]
    elif spec in ("v3v4", "v3+v4"):
        values = list(range(6, 11))
    else:
        values = [int(item) for item in spec.split(",") if item.strip()]
    if not values or min(values) < 0 or max(values) >= 11:
        raise ValueError(f"invalid --task-ids={spec!r}")
    return np.asarray(sorted(set(values)), dtype=np.int64)


class TemporalTaskDataset(Dataset):
    """Flatten (window, task) pairs without materializing duplicate windows."""

    def __init__(self, cache_path, index_path, task_ids=None, max_windows=0,
                 window_cache=None):
        index = np.load(index_path)
        self.predictions = None
        self.targets = None
        self.window_poses = None
        self.window_targets = None
        if window_cache is None:
            cache = np.load(cache_path)
            self.predictions = cache["predictions"]
            self.targets = cache["targets"]
        self.windows = index["window_indices"].astype(np.int64)
        self.actions = index["actions"].astype(np.int16)
        self.center_group_indices = index["center_group_indices"].astype(np.int64)
        if max_windows:
            self.windows = self.windows[:max_windows]
            self.actions = self.actions[:max_windows]
            self.center_group_indices = self.center_group_indices[:max_windows]
        self.task_ids = np.arange(11, dtype=np.int64) if task_ids is None else np.asarray(task_ids, dtype=np.int64)
        self.window_count = len(self.windows)
        if window_cache is not None:
            window_cache = Path(window_cache)
            self.window_poses = np.load(
                window_cache / "window_poses.npy", mmap_mode="r"
            )
            self.window_targets = np.load(
                window_cache / "window_targets.npy", mmap_mode="r"
            )
            manifest = json.loads(
                (window_cache / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest["task_ids"] != self.task_ids.tolist():
                raise ValueError(
                    f"window cache task ids {manifest['task_ids']} != "
                    f"requested {self.task_ids.tolist()}"
                )
            if manifest["window_count"] < self.window_count:
                raise ValueError("window cache is shorter than temporal index")

    def __len__(self):
        return self.window_count * len(self.task_ids)

    def __getitem__(self, index):
        task_position, window_position = divmod(index, self.window_count)
        task_id = int(self.task_ids[task_position])
        frame_indices = self.windows[window_position]
        if self.window_poses is not None:
            pose = self.window_poses[task_position, window_position]
            target = self.window_targets[window_position]
        else:
            pose = self.predictions[frame_indices, task_id].astype(np.float32, copy=False)
            center = len(frame_indices) // 2
            target = self.targets[frame_indices[center]].astype(np.float32, copy=False)
        return (
            torch.from_numpy(np.asarray(pose)),
            torch.from_numpy(np.asarray(target)),
            torch.tensor(task_id, dtype=torch.long),
            torch.tensor(int(self.actions[window_position]), dtype=torch.long),
            torch.tensor(int(self.center_group_indices[window_position]), dtype=torch.long),
        )


class BaseTemporalResidual(nn.Module):
    def __init__(self, mean, std, window_length, hidden_dim, layers):
        super().__init__()
        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_std", std.clamp_min(1e-5))
        self.window_length = window_length
        self.center = window_length // 2
        self.hidden_dim = hidden_dim
        self.joint_embedding = nn.Parameter(torch.zeros(17, hidden_dim))
        self.task_embedding = nn.Embedding(11, hidden_dim)
        self.time_embedding = nn.Parameter(torch.zeros(window_length, hidden_dim))
        self.input_projection = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.layers = layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 3)
        # Start exactly at the single-frame center pose.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _features(self, poses, task_ids):
        normalized = (poses - self.pose_mean) / self.pose_std
        root = normalized[:, :, :1]
        relative = normalized - root
        features = torch.cat((relative, root.expand(-1, -1, 17, -1)), dim=-1)
        tokens = self.input_projection(features)
        tokens = tokens + self.joint_embedding[None, None]
        tokens = tokens + self.time_embedding[None, :, None]
        tokens = tokens + self.task_embedding(task_ids)[:, None, None]
        return tokens, normalized

    def _residual(self, center_pose, center_normalized, center_features):
        # Head predicts a normalized root-relative displacement.  Tanh keeps
        # an early unstable epoch from making a many-decimeter jump.
        delta_normalized = 0.5 * torch.tanh(self.output(self.output_norm(center_features)))
        delta_normalized = delta_normalized.clone()
        delta_normalized[:, 0] = 0.0  # root translation is never altered
        delta = delta_normalized * self.pose_std
        return center_pose + delta


class FixedLagTemporalResidual(BaseTemporalResidual):
    def __init__(self, mean, std, window_length, hidden_dim, layers):
        super().__init__(mean, std, window_length, hidden_dim, layers)
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])

    def forward(self, poses, task_ids):
        tokens, normalized = self._features(poses, task_ids)
        batch, time, joints, dim = tokens.shape
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch * joints, time, dim)
        for block in self.temporal:
            tokens = block(tokens)
        center = tokens[:, self.center].reshape(batch, joints, dim)
        return self._residual(poses[:, self.center], normalized[:, self.center], center)


class MixSTEFactorizedResidual(BaseTemporalResidual):
    """Small spatial/temporal factorization following MixSTE's design."""

    def __init__(self, mean, std, window_length, hidden_dim, layers):
        super().__init__(mean, std, window_length, hidden_dim, layers)
        self.spatial = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 2,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])

    def forward(self, poses, task_ids):
        tokens, normalized = self._features(poses, task_ids)
        batch, time, joints, dim = tokens.shape
        for spatial, temporal in zip(self.spatial, self.temporal):
            spatial_tokens = tokens.reshape(batch * time, joints, dim)
            spatial_tokens = spatial(spatial_tokens)
            tokens = spatial_tokens.reshape(batch, time, joints, dim)
            temporal_tokens = tokens.permute(0, 2, 1, 3).reshape(
                batch * joints, time, dim
            )
            temporal_tokens = temporal(temporal_tokens)
            tokens = temporal_tokens.reshape(batch, joints, time, dim).permute(
                0, 2, 1, 3
            )
        center = tokens[:, self.center]
        return self._residual(poses[:, self.center], normalized[:, self.center], center)


def build_model(architecture, mean, std, window_length, hidden_dim, layers):
    cls = FixedLagTemporalResidual if architecture == "fixedlag" else MixSTEFactorizedResidual
    return cls(mean, std, window_length, hidden_dim, layers)


def stage_action_equal(error, actions):
    """Mean over actions of all-17 MPJPE, in mm."""
    frame = error.mean(axis=-1) * 1000.0
    return float(np.mean([frame[actions == a].mean() for a in sorted(set(actions))]))


@torch.inference_mode()
def evaluate(model, loader, device, task_ids):
    model.eval()
    task_errors = {task: [] for task in task_ids}
    task_baselines = {task: [] for task in task_ids}
    task_actions = {task: [] for task in task_ids}
    for poses, targets, batch_task_ids, actions, _ in loader:
        poses = poses.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_task_ids = batch_task_ids.to(device, non_blocking=True)
        prediction = model(poses, batch_task_ids)
        error = torch.linalg.vector_norm(prediction - targets, dim=-1).cpu().numpy()
        baseline = torch.linalg.vector_norm(poses[:, poses.shape[1] // 2] - targets, dim=-1).cpu().numpy()
        for task in task_ids:
            mask = batch_task_ids.cpu().numpy() == task
            if np.any(mask):
                task_errors[task].append(error[mask])
                task_baselines[task].append(baseline[mask])
                task_actions[task].append(actions.numpy()[mask])
    result = {"tasks": {}, "stages": {}}
    for task in task_ids:
        err = np.concatenate(task_errors[task], axis=0)
        base = np.concatenate(task_baselines[task], axis=0)
        act = np.concatenate(task_actions[task], axis=0)
        result["tasks"][str(COMBINATIONS[task])] = {
            "temporal_action_equal_all17_mm": stage_action_equal(err, act),
            "center_baseline_action_equal_all17_mm": stage_action_equal(base, act),
            "frame_weighted_temporal_all17_mm": float(err.mean() * 1000.0),
            "delta_mm": float((err.mean() - base.mean()) * 1000.0),
        }
    for stage, indices in STAGES.items():
        values = [
            result["tasks"][str(COMBINATIONS[i])]
            for i in indices if i in task_ids
        ]
        if not values:
            continue
        result["stages"][stage] = {
            "temporal_action_equal_all17_mm": float(np.mean([
                v["temporal_action_equal_all17_mm"] for v in values
            ])),
            "center_baseline_action_equal_all17_mm": float(np.mean([
                v["center_baseline_action_equal_all17_mm"] for v in values
            ])),
            "delta_mm": float(np.mean([v["delta_mm"] for v in values])),
        }
    return result


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    task_ids = parse_task_ids(args.task_ids)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_all = TemporalTaskDataset(
        args.train_cache, args.train_index, task_ids=task_ids,
        max_windows=args.max_train_windows,
        window_cache=args.train_window_cache,
    )
    validation = TemporalTaskDataset(
        args.validation_cache, args.validation_index,
        task_ids=task_ids,
        max_windows=args.max_validation_windows,
        window_cache=args.validation_window_cache,
    )
    # Per-joint train statistics are computed only from center targets; no
    # validation labels enter model construction or checkpoint selection.
    if train_all.window_targets is not None:
        train_targets = np.asarray(train_all.window_targets[:train_all.window_count])
    else:
        train_targets = train_all.targets[
            train_all.windows[:, args.window_length // 2]
        ]
    mean = torch.from_numpy(train_targets.mean(axis=0)).float()
    std = torch.from_numpy(train_targets.std(axis=0)).float()
    model = build_model(
        args.architecture, mean, std, args.window_length,
        args.hidden_dim, args.layers,
    ).to(device)

    # Internal temporal holdout uses complete windows selected by center group,
    # then expands the same window split to every camera task.
    holdout_windows = train_all.center_group_indices % 10 == 0
    train_tasks = TemporalTaskDataset(
        args.train_cache, args.train_index,
        task_ids=task_ids, max_windows=args.max_train_windows,
        window_cache=args.train_window_cache,
    )
    # Dataset indexing is task-major, so select sample indices explicitly.
    total_windows = train_tasks.window_count
    all_sample_indices = np.arange(len(train_tasks), dtype=np.int64)
    sample_window = all_sample_indices % total_windows
    train_sample_indices = all_sample_indices[~holdout_windows[sample_window]]
    holdout_sample_indices = all_sample_indices[holdout_windows[sample_window]]

    class Indexed(Dataset):
        def __init__(self, base, indices): self.base, self.indices = base, indices
        def __len__(self): return len(self.indices)
        def __getitem__(self, i): return self.base[int(self.indices[i])]

    train_loader = DataLoader(
        Indexed(train_tasks, train_sample_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    holdout_loader = DataLoader(
        Indexed(train_tasks, holdout_sample_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    checkpoint = out_dir / "model_best.pth.tar"
    best_metric = math.inf
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_index, (poses, targets, task_ids, _, _) in enumerate(train_loader):
            poses = poses.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            task_ids = task_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(poses, task_ids)
            # Absolute robust loss plus a small root-relative term.  The root
            # is copied by construction, so the second term trains articulation
            # without encouraging global translation drift.
            loss_abs = F.smooth_l1_loss(prediction, targets, beta=0.01)
            pred_rel = prediction - prediction[:, :1]
            target_rel = targets - targets[:, :1]
            loss_rel = F.smooth_l1_loss(pred_rel, target_rel, beta=0.01)
            loss = loss_abs + 0.25 * loss_rel
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        holdout = evaluate(model, holdout_loader, device, task_ids.tolist())
        active_stages = [
            stage for stage, indices in STAGES.items()
            if any(index in set(task_ids.tolist()) for index in indices)
        ]
        metric = float(np.mean([
            holdout["stages"][stage]["temporal_action_equal_all17_mm"]
            for stage in active_stages
        ]))
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": metric,
            "holdout": holdout,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "architecture": args.architecture, "window_length": args.window_length,
                "hidden_dim": args.hidden_dim, "layers": args.layers,
                "epoch": epoch,
            }, checkpoint)

    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    final = None if args.smoke_batches else evaluate(
        model, val_loader, device, task_ids.tolist()
    )
    result = {
        "method": (
            "E-2 fixed-lag temporal residual" if args.architecture == "fixedlag"
            else "E-2 MixSTE-style factorized temporal residual"
        ),
        "paper_basis": "MixSTE (CVPR 2022) factorized spatial/temporal attention; fixed-lag control",
        "protocol": {
            "train_cache": args.train_cache, "validation_cache": args.validation_cache,
            "window_length": args.window_length, "frame_stride": 5,
            "root_protected": True, "center_frame_target": True,
            "train_action_codes": sorted(set(train_tasks.actions.tolist())),
            "train_windows": train_tasks.window_count,
            "internal_holdout_windows": int(holdout_windows.sum()),
            "validation_windows": validation.window_count,
        },
        "architecture": args.architecture,
        "task_ids": task_ids.tolist(),
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": final,
        "args": vars(args),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"S9_S11_final_once": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
