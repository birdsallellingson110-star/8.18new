#!/usr/bin/env python3
"""Train T-CVU: temporal residuals on E-2 candidate utility logits.

The model never regresses a new 3-D pose.  It predicts a zero-initialized
correction to the E-2 candidate utility for the target frame, then fuses the
existing RUMPL/E-2 candidates.  This preserves exact E-2 identity at step 0.
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

from train_h76_hypothesis_utility_20260811 import (
    COMBINATIONS,
    TASK_COMBINATIONS,
)
from train_h76_pairwise_set_transformer_20260812 import EXPANDED_COMBINATIONS
from train_h76_counterfactual_delta_20260811 import training_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-index", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-index", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--holdout-subject", type=int, default=8)
    p.add_argument(
        "--expanded-candidates", action="store_true",
        help="Use the 17-candidate H76+pairwise cache instead of the original 11.",
    )
    p.add_argument(
        "--task-stage", choices=("all", "v3", "v4"), default="all",
        help="Train/evaluate only the selected view-count task stage.",
    )
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-validation-windows", type=int, default=0)
    p.add_argument(
        "--aux-file", default="candidate_confidence.npy",
        help="one optional candidate-level quality feature stored in the cache",
    )
    return p.parse_args()


CANDIDATE_COMBINATIONS = COMBINATIONS
ACTIVE_TASK_POSITIONS = list(range(len(TASK_COMBINATIONS)))


def configure_candidate_pool(expanded: bool, task_stage: str = "all"):
    global CANDIDATE_COMBINATIONS, TASK_AVAILABLE, TASK_BASELINE_LOCAL
    global ACTIVE_TASK_POSITIONS
    CANDIDATE_COMBINATIONS = EXPANDED_COMBINATIONS if expanded else COMBINATIONS
    TASK_AVAILABLE = [task_available(task) for task in TASK_COMBINATIONS]
    TASK_BASELINE_LOCAL = [
        available.index(CANDIDATE_COMBINATIONS.index(task))
        for task, available in zip(TASK_COMBINATIONS, TASK_AVAILABLE)
    ]
    if task_stage == "all":
        ACTIVE_TASK_POSITIONS = list(range(len(TASK_COMBINATIONS)))
    elif task_stage == "v3":
        ACTIVE_TASK_POSITIONS = list(range(4))
    elif task_stage == "v4":
        ACTIVE_TASK_POSITIONS = [4]
    else:
        raise ValueError(f"unknown task stage: {task_stage}")


def task_available(task):
    return [
        i for i, combo in enumerate(CANDIDATE_COMBINATIONS)
        if set(combo).issubset(task)
    ]


TASK_AVAILABLE = [task_available(task) for task in TASK_COMBINATIONS]
TASK_BASELINE_LOCAL = [
    available.index(CANDIDATE_COMBINATIONS.index(task))
    for task, available in zip(TASK_COMBINATIONS, TASK_AVAILABLE)
]


class CandidateFrameArrays:
    def __init__(self, cache_dir, index_path, max_windows=0, aux_file=None):
        cache_dir = Path(cache_dir)
        self.candidate_poses = np.load(cache_dir / "candidate_poses.npy", mmap_mode="r")
        confidence_path = None if aux_file in (None, "", "none") else (
            cache_dir / aux_file
        )
        self.candidate_confidence = (
            np.load(confidence_path, mmap_mode="r")
            if confidence_path is not None and confidence_path.exists() else None
        )
        self.utility_delta = np.load(cache_dir / "utility_delta.npy", mmap_mode="r")
        self.targets = np.load(cache_dir / "targets.npy", mmap_mode="r")
        index = np.load(index_path)
        self.windows = index["window_indices"].astype(np.int64)
        self.actions = index["actions"].astype(np.int16)
        self.subjects = index["subjects"].astype(np.int16)
        self.center_group_indices = index["center_group_indices"].astype(np.int64)
        if max_windows:
            self.windows = self.windows[:max_windows]
            self.actions = self.actions[:max_windows]
            self.subjects = self.subjects[:max_windows]
            self.center_group_indices = self.center_group_indices[:max_windows]
        self.window_count = len(self.windows)
        self.window_length = int(self.windows.shape[1])

    def gather(self, sample_indices):
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        local_task_positions = sample_indices // self.window_count
        window_positions = sample_indices % self.window_count
        task_positions = np.asarray(
            [ACTIVE_TASK_POSITIONS[int(i)] for i in local_task_positions],
            dtype=np.int64,
        )
        frame_indices = self.windows[window_positions]
        batch = len(sample_indices)
        time = self.window_length
        candidate_poses = np.asarray(
            self.candidate_poses[frame_indices], dtype=np.float32
        )  # B,T,11,J,3
        candidate_confidence = None
        if self.candidate_confidence is not None:
            candidate_confidence = np.asarray(
                self.candidate_confidence[frame_indices], dtype=np.float32
            )  # B,T,11,J
        utility_all = np.asarray(
            self.utility_delta[frame_indices], dtype=np.float32
        )  # B,T,5,J,11
        time_axis = np.arange(time, dtype=np.int64)[None, :]
        batch_axis = np.arange(batch, dtype=np.int64)[:, None]
        utility = utility_all[batch_axis, time_axis, task_positions[:, None]]
        targets = np.asarray(
            self.targets[frame_indices[:, time // 2]], dtype=np.float32
        )
        return (
            candidate_poses,
            utility,
            targets,
            task_positions.astype(np.int64),
            self.actions[window_positions],
            candidate_confidence,
        )


class TemporalCandidateUtility(nn.Module):
    def __init__(self, window_length, hidden_dim=64, layers=1, heads=8):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.window_length = window_length
        self.center = window_length // 2
        # candidate motion(3), candidate-vs-set displacement(3), displacement
        # norm(1), frozen E-2 utility logit(1), optional mean detector
        # confidence (1).  The confidence channel is the I3 single-variable
        # ablation; zero is retained for backwards-compatible identity tests.
        self.input_projection = nn.Sequential(
            nn.Linear(9, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.time_embedding = nn.Parameter(torch.zeros(window_length, hidden_dim))
        self.joint_embedding = nn.Parameter(torch.zeros(17, hidden_dim))
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=heads,
                dim_feedforward=hidden_dim * 2, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        # Exact E-2 identity at initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)

    def forward(self, candidate_poses, utility_delta, candidate_confidence=None):
        # candidate_poses: B,T,C,J,3; utility_delta: B,T,J,C
        center_pose = candidate_poses[:, self.center]
        motion = (candidate_poses - center_pose[:, None]) / 0.1
        consensus = candidate_poses.mean(dim=2, keepdim=True)
        displacement = (candidate_poses - consensus) / 0.1
        displacement_norm = torch.linalg.vector_norm(
            displacement, dim=-1, keepdim=True
        )
        utility = utility_delta.permute(0, 1, 3, 2).unsqueeze(-1)
        if candidate_confidence is None:
            confidence = torch.zeros_like(displacement_norm)
        else:
            confidence = candidate_confidence.permute(0, 1, 2, 3).unsqueeze(-1)
        features = torch.cat(
            (motion, displacement, displacement_norm, utility, confidence), dim=-1
        )
        batch, time, candidates, joints, _ = features.shape
        tokens = self.input_projection(features)
        tokens = tokens + self.time_embedding[None, :, None, None]
        tokens = tokens + self.joint_embedding[None, None, None]
        tokens = tokens.permute(0, 3, 2, 1, 4).reshape(
            batch * joints * candidates, time, -1
        )
        for block in self.temporal:
            # PyTorch's attention kernel has a grid-size limit on this
            # machine.  A V4-only batch expands to 512*17*17 tokens, which
            # exceeds that limit although the tensor itself is valid.  Chunk
            # in both train and eval; concatenation is mathematically exact
            # because the temporal block does not mix token rows.
            if tokens.shape[0] > 16384:
                tokens = torch.cat(
                    [block(chunk) for chunk in tokens.split(16384, dim=0)],
                    dim=0,
                )
            else:
                tokens = block(tokens)
        center = tokens[:, self.center]
        residual = 0.5 * torch.tanh(
            self.output(self.output_norm(center)).squeeze(-1)
        )
        return residual.reshape(batch, joints, candidates)


def stage_action_equal(values, actions):
    values = np.asarray(values)
    actions = np.asarray(actions)
    return float(np.mean([
        values[actions == action].mean()
        for action in sorted(set(actions.tolist()))
    ]) * 1000.0)


def forward_task(
    model, poses, utility, targets, task_position, candidate_confidence=None
):
    available = TASK_AVAILABLE[task_position]
    baseline_local = TASK_BASELINE_LOCAL[task_position]
    candidates = poses[:, :, available]
    candidate_utility = utility[:, :, :, available]
    candidate_quality = None
    if candidate_confidence is not None:
        candidate_quality = candidate_confidence[:, :, available]
    correction = model(candidates, candidate_utility, candidate_quality)
    center_delta = candidate_utility[:, model.center]
    predicted_delta = center_delta + correction
    predicted_delta = predicted_delta.clone()
    predicted_delta[:, :, baseline_local] = 0.0
    center_candidates = candidates[:, model.center]
    true_error = torch.linalg.vector_norm(
        center_candidates - targets[:, None], dim=-1
    ).permute(0, 2, 1)
    true_delta = true_error - true_error[:, :, baseline_local:baseline_local + 1]
    weights = F.softmax(-predicted_delta, dim=-1)
    fused = torch.einsum("bjc,bcjd->bjd", weights, center_candidates)
    # The comparison baseline is the frozen E-2 soft fusion, not the raw
    # full-view H76 hypothesis.  At zero correction this is an exact
    # identity check for the exported E-2 logits.  The raw full-view
    # hypothesis is still the reference used to construct true_delta, which
    # matches the counterfactual utility supervision used by E-2.
    e2_weights = F.softmax(-center_delta, dim=-1)
    e2_baseline = torch.einsum(
        "bjc,bcjd->bjd", e2_weights, center_candidates
    )
    return predicted_delta, true_delta, fused, e2_baseline


@torch.inference_mode()
def evaluate(model, arrays, indices, device, batch_size):
    model.eval()
    active_stage_names = sorted({"V3" if i < 4 else "V4" for i in ACTIVE_TASK_POSITIONS})
    stores = {
        stage: {"pred": [], "base": [], "actions": []}
        for stage in active_stage_names
    }
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        poses, utility, targets, task_positions, actions, confidence = arrays.gather(
            batch_indices
        )
        poses_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
        utility_t = torch.from_numpy(utility).to(device=device, dtype=torch.float32)
        targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
        confidence_t = None if confidence is None else torch.from_numpy(
            confidence
        ).to(device=device, dtype=torch.float32)
        for task_position in ACTIVE_TASK_POSITIONS:
            mask = task_positions == task_position
            if not np.any(mask):
                continue
            mask_t = torch.from_numpy(mask).to(device=device)
            predicted, true_delta, fused, baseline = forward_task(
                model, poses_t[mask_t], utility_t[mask_t], targets_t[mask_t],
                task_position,
                None if confidence_t is None else confidence_t[mask_t],
            )
            del predicted, true_delta
            pred_error = torch.linalg.vector_norm(
                fused - targets_t[mask_t], dim=-1
            ).cpu().numpy()
            base_error = torch.linalg.vector_norm(
                baseline - targets_t[mask_t], dim=-1
            ).cpu().numpy()
            stage = "V3" if task_position < 4 else "V4"
            stores[stage]["pred"].append(pred_error)
            stores[stage]["base"].append(base_error)
            stores[stage]["actions"].append(actions[mask])
    result = {}
    for stage, store in stores.items():
        pred = np.concatenate(store["pred"], axis=0)
        base = np.concatenate(store["base"], axis=0)
        actions = np.concatenate(store["actions"])
        result[stage] = {
            "temporal_action_equal_all17_mm": stage_action_equal(pred, actions),
            "center_e2_action_equal_all17_mm": stage_action_equal(base, actions),
            "frame_weighted_temporal_all17_mm": float(pred.mean() * 1000.0),
            "frame_weighted_delta_mm": float((pred.mean() - base.mean()) * 1000.0),
        }
    result["mean_active_action_equal_mm"] = float(np.mean([
        result[stage]["temporal_action_equal_all17_mm"]
        for stage in active_stage_names
    ]))
    if set(active_stage_names) == {"V3", "V4"}:
        result["mean_v34_action_equal_mm"] = result["mean_active_action_equal_mm"]
    return result


def main():
    args = parse_args()
    configure_candidate_pool(args.expanded_candidates, args.task_stage)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train = CandidateFrameArrays(
        args.train_cache, args.train_index, args.max_train_windows, args.aux_file
    )
    validation = CandidateFrameArrays(
        args.validation_cache, args.validation_index, args.max_validation_windows,
        args.aux_file,
    )
    if train.window_length not in (3, 5, 9):
        raise ValueError(f"unexpected window length {train.window_length}")
    model = TemporalCandidateUtility(
        train.window_length, args.hidden_dim, args.layers, args.heads
    ).to(device)
    total_samples = train.window_count * len(ACTIVE_TASK_POSITIONS)
    all_indices = np.arange(total_samples, dtype=np.int64)
    sample_windows = all_indices % train.window_count
    if args.holdout_subject:
        holdout_windows = train.subjects == args.holdout_subject
        if not np.any(holdout_windows):
            raise ValueError(f"subject {args.holdout_subject} absent from train")
        holdout_protocol = f"leave-subject-{args.holdout_subject}-out"
    else:
        holdout_windows = train.center_group_indices % 10 == 0
        holdout_protocol = "center-group-index-modulo-10"
    train_indices = all_indices[~holdout_windows[sample_windows]]
    holdout_indices = all_indices[holdout_windows[sample_windows]]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    checkpoint = out_dir / "model_best.pth.tar"
    best_metric = math.inf
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        order = train_indices.copy()
        rng.shuffle(order)
        model.train()
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch_indices = order[start:start + args.batch_size]
            poses, utility, targets, task_positions, _, confidence = train.gather(
                batch_indices
            )
            poses_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
            utility_t = torch.from_numpy(utility).to(device=device, dtype=torch.float32)
            targets_t = torch.from_numpy(targets).to(device=device, dtype=torch.float32)
            confidence_t = None if confidence is None else torch.from_numpy(
                confidence
            ).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            active = 0
            for task_position in ACTIVE_TASK_POSITIONS:
                mask = task_positions == task_position
                if not np.any(mask):
                    continue
                mask_t = torch.from_numpy(mask).to(device=device)
                predicted, true_delta, _, _ = forward_task(
                    model, poses_t[mask_t], utility_t[mask_t], targets_t[mask_t],
                    task_position,
                    None if confidence_t is None else confidence_t[mask_t],
                )
                loss = loss + training_loss(predicted, true_delta, "balanced_rank")
                active += 1
            loss = loss / max(active, 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout = evaluate(model, train, holdout_indices, device, args.batch_size)
        metric = holdout["mean_active_action_equal_mm"]
        record = {
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": float(metric), "holdout": holdout,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = float(metric)
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "window_length": train.window_length,
                "hidden_dim": args.hidden_dim, "layers": args.layers,
                "heads": args.heads, "epoch": epoch,
            }, checkpoint)
    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    final = evaluate(
        model, validation,
        np.arange(validation.window_count * len(ACTIVE_TASK_POSITIONS), dtype=np.int64),
        device, args.batch_size,
    )
    result = {
        "method": "T-CVU temporal residual on frozen E-2 candidate utility",
        "protocol": {
            "window_length": train.window_length, "frame_stride": 5,
            "center_target": True, "identity_initialized": True,
            "task_stage": args.task_stage,
            "direct_3d_regression": False, "aux_file": args.aux_file,
            "aux_feature_present": train.candidate_confidence is not None,
            "holdout_protocol": holdout_protocol,
            "train_windows": train.window_count,
            "internal_holdout_windows": int(holdout_windows.sum()),
            "validation_windows": validation.window_count,
        },
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": final, "args": vars(args),
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"S9_S11_final_once": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
