#!/usr/bin/env python3
"""GBT-aligned causal temporal refinement of the frozen E2-C2 pose.

This experiment fixes three limitations of H18:

1. evaluation predicts the latest frame from the current and eight past frames
   instead of predicting the center frame with future context;
2. every frame in the input window is supervised (seq2seq), following GBT and
   MixSTE rather than training only a center-frame residual;
3. spatial and temporal transformer blocks are alternated, as in MixSTE.

Model selection uses S1/S5/S6/S7 -> S8.  Once the epoch count is fixed, the
model is reinitialized and fitted on the official complete H36M training set
S1/S5/S6/S7/S8, then S9/S11 are evaluated exactly once.  The E2-C2 generator,
candidate scorer and input protocol remain frozen.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_e2_clean_temporal_residual_20260818 import (
    ACTION_NAMES,
    JOINT_COUNT,
    TASK_COUNT,
    action_equal,
    build_windows,
    metadata_from_pkl,
    select_rows,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-fused", required=True)
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-fused", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window-length", type=int, default=9)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--relative-scale-m", type=float, default=0.10)
    p.add_argument("--root-scale-m", type=float, default=0.05)
    p.add_argument("--root-mode", choices=("protected", "learned"),
                   default="learned")
    p.add_argument("--temporal-loss-weight", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-holdout-windows", type=int, default=0)
    p.add_argument("--max-validation-windows", type=int, default=0)
    p.add_argument("--preload-fused", action="store_true",
                   help="Load fused pose memmaps into RAM for random temporal windows.")
    p.add_argument("--no-refit-all", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CausalSeq2SeqTemporalModel(nn.Module):
    """Alternating MixSTE-style encoder with identity initialization.

    The encoder itself can attend throughout the supplied window.  The formal
    causal guarantee comes from reporting only the last output token: there is
    no observation later than the evaluated frame in that window.
    """

    def __init__(self, window_length: int, hidden_dim: int, layers: int,
                 relative_scale_m: float, root_scale_m: float,
                 root_mode: str, task_count: int = TASK_COUNT):
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden-dim must be divisible by 8")
        self.window_length = int(window_length)
        self.relative_scale_m = float(relative_scale_m)
        self.root_scale_m = float(root_scale_m)
        self.root_mode = root_mode
        self.input = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.joint_embedding = nn.Parameter(torch.zeros(JOINT_COUNT, hidden_dim))
        self.task_embedding = nn.Embedding(task_count, hidden_dim)
        self.time_embedding = nn.Parameter(torch.zeros(window_length, hidden_dim))
        self.spatial = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])
        self.relative_output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3)
        )
        self.root_output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3)
        )
        for head in (self.relative_output[-1], self.root_output[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.task_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(self, pose: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        # pose: B,T,K,J,3; output: B,T,K,J,3
        if pose.ndim != 5:
            raise ValueError(f"expected B,T,K,J,3, got {tuple(pose.shape)}")
        batch, time, tasks, joints, channels = pose.shape
        if time != self.window_length or joints != JOINT_COUNT or channels != 3:
            raise ValueError("unexpected temporal pose shape")
        root = pose[:, :, :, :1]
        relative = pose - root
        velocity = torch.diff(pose, dim=1, prepend=pose[:, :1])
        acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1])
        features = torch.cat(
            (relative, root.expand(-1, -1, -1, joints, -1),
             velocity, acceleration), dim=-1,
        )
        x = self.input(features)
        x = x + self.joint_embedding[None, None, None]
        x = x + self.task_embedding(task_ids)[:, None, :, None]
        x = x + self.time_embedding[None, :, None, None]

        # MixSTE alternates a per-frame spatial block and a per-joint temporal
        # block at every depth.  H18 applied all spatial blocks first.
        for spatial, temporal in zip(self.spatial, self.temporal):
            x = x.reshape(batch * time * tasks, joints, -1)
            x = spatial(x)
            x = x.reshape(batch, time, tasks, joints, -1)
            x = x.permute(0, 2, 3, 1, 4).reshape(
                batch * tasks * joints, time, -1
            )
            x = temporal(x)
            x = x.reshape(batch, tasks, joints, time, -1).permute(0, 3, 1, 2, 4)

        relative_delta = self.relative_scale_m * torch.tanh(self.relative_output(x))
        relative_delta = relative_delta.clone()
        relative_delta[:, :, :, 0] = 0.0
        if self.root_mode == "learned":
            pooled = x.mean(dim=3)
            root_delta = self.root_scale_m * torch.tanh(self.root_output(pooled))
            root_delta = root_delta[:, :, :, None]
        else:
            root_delta = torch.zeros_like(relative_delta[:, :, :, :1])
        return pose + relative_delta + root_delta


def gather_batch(cache, rows: np.ndarray, fused: np.ndarray,
                 device: torch.device):
    pose = torch.from_numpy(np.asarray(fused[rows])).to(
        device=device, dtype=torch.float32
    )
    target = torch.from_numpy(np.asarray(cache["targets"][rows])).to(
        device=device, dtype=torch.float32
    )
    task_ids = torch.arange(TASK_COUNT, device=device)[None].expand(len(rows), -1)
    return pose, target, task_ids


@torch.inference_mode()
def evaluate(model: nn.Module | None, cache, fused: np.ndarray,
             windows: np.ndarray, device: torch.device,
             batch_size: int) -> dict:
    names = ("absolute", "root", "root_relative")
    base_values = {name: {f"V{k}": [] for k in (2, 3, 4)} for name in names}
    pred_values = {name: {f"V{k}": [] for k in (2, 3, 4)} for name in names}
    action_values = {f"V{k}": [] for k in (2, 3, 4)}
    if model is not None:
        model.eval()
    for start in range(0, len(windows), batch_size):
        rows = windows[start:start + batch_size]
        pose, target_sequence, task_ids = gather_batch(cache, rows, fused, device)
        baseline = pose[:, -1]
        prediction = baseline if model is None else model(pose, task_ids)[:, -1]
        target = target_sequence[:, -1, None]
        base_absolute = torch.linalg.vector_norm(baseline - target, dim=-1)
        pred_absolute = torch.linalg.vector_norm(prediction - target, dim=-1)
        base_root = torch.linalg.vector_norm(
            baseline[:, :, 0] - target[:, :, 0], dim=-1
        )
        pred_root = torch.linalg.vector_norm(
            prediction[:, :, 0] - target[:, :, 0], dim=-1
        )
        base_relative = torch.linalg.vector_norm(
            (baseline - baseline[:, :, :1]) - (target - target[:, :, :1]), dim=-1
        )
        pred_relative = torch.linalg.vector_norm(
            (prediction - prediction[:, :, :1]) - (target - target[:, :, :1]), dim=-1
        )
        base_batch = {
            "absolute": base_absolute, "root": base_root,
            "root_relative": base_relative,
        }
        pred_batch = {
            "absolute": pred_absolute, "root": pred_root,
            "root_relative": pred_relative,
        }
        actions = np.asarray(cache["actions"][rows[:, -1]])
        for task_index, count in enumerate((2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4)):
            stage = f"V{count}"
            for name in names:
                base_values[name][stage].append(
                    base_batch[name][:, task_index].cpu().numpy() * 1000.0
                )
                pred_values[name][stage].append(
                    pred_batch[name][:, task_index].cpu().numpy() * 1000.0
                )
            action_values[stage].append(actions.copy())
    result: dict[str, object] = {}
    stage_metrics = []
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(action_values[stage])
        stage_result = {}
        for name in names:
            base = np.concatenate(base_values[name][stage])
            pred = np.concatenate(pred_values[name][stage])
            base_metric = action_equal(base, actions)
            pred_metric = action_equal(pred, actions)
            stage_result[f"baseline_{name}_mm"] = float(base_metric)
            stage_result[f"temporal_{name}_mm"] = float(pred_metric)
            stage_result[f"delta_{name}_mm"] = float(pred_metric - base_metric)
        result[stage] = stage_result
        stage_metrics.append(stage_result["temporal_absolute_mm"])
    result["mean_v234_mm"] = float(np.mean(stage_metrics))
    result["windows"] = int(len(windows))
    return result


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, cache,
                fused: np.ndarray, windows: np.ndarray, device: torch.device,
                batch_size: int, seed: int, temporal_loss_weight: float) -> dict:
    model.train()
    order = np.random.default_rng(seed).permutation(len(windows))
    losses, coordinates, mpjpes, temporals = [], [], [], []
    for offset in range(0, len(order), batch_size):
        rows = windows[order[offset:offset + batch_size]]
        pose, target, task_ids = gather_batch(cache, rows, fused, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pose, task_ids)
        target_expanded = target[:, :, None].expand_as(prediction)
        coordinate = F.smooth_l1_loss(
            prediction, target_expanded, beta=0.01
        )
        mpjpe = torch.linalg.vector_norm(
            prediction - target_expanded, dim=-1
        ).mean()
        if temporal_loss_weight > 0:
            pred_velocity = torch.diff(prediction, dim=1)
            target_velocity = torch.diff(target_expanded, dim=1)
            temporal = F.smooth_l1_loss(
                pred_velocity, target_velocity, beta=0.005
            )
        else:
            temporal = prediction.new_zeros(())
        loss = coordinate + 0.10 * mpjpe + temporal_loss_weight * temporal
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        coordinates.append(float(coordinate.detach().cpu()))
        mpjpes.append(float(mpjpe.detach().cpu()))
        temporals.append(float(temporal.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "coordinate": float(np.mean(coordinates)),
        "mpjpe_m": float(np.mean(mpjpes)),
        "temporal": float(np.mean(temporals)),
    }


def make_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    return CausalSeq2SeqTemporalModel(
        args.window_length, args.hidden_dim, args.layers,
        args.relative_scale_m, args.root_scale_m, args.root_mode,
    ).to(device)


def make_optimizer(model: nn.Module, args: argparse.Namespace):
    return torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )


def main() -> None:
    args = parse_args()
    if args.window_length < 2:
        raise ValueError("causal experiment requires at least two frames")
    seed_everything(args.seed)
    print("loading E2-C2 caches", flush=True)
    train_cache = np.load(args.train_cache, allow_pickle=False)
    val_cache = np.load(args.validation_cache, allow_pickle=False)
    train_fused = np.load(args.train_fused, mmap_mode="r")
    val_fused = np.load(args.validation_fused, mmap_mode="r")
    if getattr(args, "preload_fused", False):
        print("preloading fused temporal anchors into RAM", flush=True)
        train_fused = np.asarray(train_fused).copy()
        val_fused = np.asarray(val_fused).copy()
    print("loading and validating frame metadata", flush=True)
    train_meta = metadata_from_pkl(args.train_pkl, len(train_cache["targets"]))
    val_meta = metadata_from_pkl(args.validation_pkl, len(val_cache["targets"]))
    for name, cache, meta in (("train", train_cache, train_meta),
                              ("validation", val_cache, val_meta)):
        if not np.array_equal(cache["subjects"], meta["subjects"]):
            raise RuntimeError(f"{name} subject order mismatch")
        if not np.array_equal(cache["actions"], meta["actions"]):
            raise RuntimeError(f"{name} action order mismatch")

    print("building causal windows", flush=True)
    all_train = build_windows(train_meta, args.window_length, args.frame_stride)
    all_val = build_windows(val_meta, args.window_length, args.frame_stride)
    latest = all_train[:, -1]
    selection_train = all_train[
        np.isin(train_meta["subjects"][latest], [1, 5, 6, 7])
    ]
    holdout = all_train[train_meta["subjects"][latest] == 8]
    official_train = all_train[
        np.isin(train_meta["subjects"][latest], [1, 5, 6, 7, 8])
    ]
    selection_train = select_rows(selection_train, args.max_train_windows)
    official_train = select_rows(official_train, args.max_train_windows)
    holdout = select_rows(holdout, args.max_holdout_windows)
    all_val = select_rows(all_val, args.max_validation_windows)
    if not len(selection_train) or not len(holdout) or not len(all_val):
        raise RuntimeError("empty train, holdout, or validation windows")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print("evaluating latest-frame baseline", flush=True)
    baseline_holdout = evaluate(
        None, train_cache, train_fused, holdout, device, args.batch_size
    )
    baseline_validation = evaluate(
        None, val_cache, val_fused, all_val, device, args.batch_size
    )
    manifest = {
        "method": "GBT-aligned causal latest-frame MixSTE seq2seq E2-C2 residual",
        "protocol": {
            "input_frames": args.window_length,
            "train_outputs": "all input frames",
            "evaluation_output": "latest frame only",
            "future_context_at_evaluation": False,
            "selection_subjects": [1, 5, 6, 7],
            "holdout_subjects": [8],
            "official_refit_subjects": [1, 5, 6, 7, 8],
            "test_subjects": [9, 11],
            "spatial_temporal_order": "alternating at every layer",
        },
        "train_cache": str(Path(args.train_cache).resolve()),
        "train_fused": str(Path(args.train_fused).resolve()),
        "validation_cache": str(Path(args.validation_cache).resolve()),
        "validation_fused": str(Path(args.validation_fused).resolve()),
        "selection_train_windows": int(len(selection_train)),
        "holdout_windows": int(len(holdout)),
        "official_train_windows": int(len(official_train)),
        "validation_windows": int(len(all_val)),
        "args": vars(args),
        "baseline_holdout": baseline_holdout,
        "baseline_validation": baseline_validation,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"baseline_holdout": baseline_holdout,
                      "baseline_validation": baseline_validation}, indent=2),
          flush=True)

    model = make_model(args, device)
    optimizer = make_optimizer(model, args)
    best_metric = float(baseline_holdout["mean_v234_mm"])
    best_epoch = -1
    selection_history = []
    for epoch in range(args.epochs):
        train_stats = train_epoch(
            model, optimizer, train_cache, train_fused, selection_train,
            device, args.batch_size, args.seed + epoch,
            args.temporal_loss_weight,
        )
        holdout_result = evaluate(
            model, train_cache, train_fused, holdout, device, args.batch_size
        )
        row = {
            "epoch": epoch, "train": train_stats,
            "holdout_selection_metric_mm": holdout_result["mean_v234_mm"],
            "holdout": holdout_result,
        }
        selection_history.append(row)
        print(json.dumps({"selection": row}), flush=True)
        if holdout_result["mean_v234_mm"] < best_metric:
            best_metric = float(holdout_result["mean_v234_mm"])
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "best_holdout_metric_mm": best_metric,
                        "args": vars(args)}, out / "model_selection_best.pth.tar")

    gate_mm = float(getattr(args, "gate_mm", 0.0))
    holdout_gain_mm = float(
        baseline_holdout["mean_v234_mm"] - best_metric
    )
    gate_passed = best_epoch >= 0 and holdout_gain_mm >= gate_mm
    final_training_epochs = max(best_epoch + 1, 0) if gate_passed else 0
    refit_history = []
    if gate_passed and not args.no_refit_all:
        # Reinitialize so S8 is used only after the training duration has been
        # fixed.  This is the model compared with the official GBT protocol.
        seed_everything(args.seed)
        final_model = make_model(args, device)
        final_optimizer = make_optimizer(final_model, args)
        for epoch in range(final_training_epochs):
            stats = train_epoch(
                final_model, final_optimizer, train_cache, train_fused,
                official_train, device, args.batch_size,
                args.seed + epoch, args.temporal_loss_weight,
            )
            refit_history.append({"epoch": epoch, "train": stats})
            print(json.dumps({"official_refit": refit_history[-1]}), flush=True)
        torch.save({"state_dict": final_model.state_dict(),
                    "epochs": final_training_epochs, "args": vars(args)},
                   out / "model_official_refit.pth.tar")
    elif gate_passed:
        final_model = model
        state = torch.load(out / "model_selection_best.pth.tar",
                           map_location=device, weights_only=False)
        final_model.load_state_dict(state["state_dict"], strict=True)
    else:
        final_model = None

    final_validation = evaluate(
        final_model, val_cache, val_fused, all_val, device, args.batch_size
    )
    result = {
        **manifest,
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "holdout_gain_mm": holdout_gain_mm,
        "gate_mm": gate_mm,
        "gate_passed": gate_passed,
        "selection_history": selection_history,
        "official_refit_epochs": final_training_epochs,
        "official_refit_history": refit_history,
        "S9_S11_final_once": final_validation,
        "gbt_hrnet_target_mm": {"V2": 36.8, "V3": 30.4, "V4": 26.0},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "COMPLETED").write_text("completed\n")
    print(json.dumps({"S9_S11_final_once": final_validation,
                      "best_epoch": best_epoch,
                      "official_refit_epochs": final_training_epochs}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
