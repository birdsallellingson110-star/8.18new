#!/usr/bin/env python3
"""Pairwise E2 with the canonical whole-pose score used by GHT.

This is a controlled architecture experiment.  The frozen RUMPL-derived
candidate pool and the E2 per-joint utility remain unchanged.  The added
branch follows the public Generalizable Human Pose Triangulation (CVPR 2022)
ScoreNN preprocessing: coordinate standardization, pelvis centering, shoulder
normal canonicalization, and optional H36M bone-length features.  Its last
layer is zero initialized, so the initial model is exactly pairwise E2.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import ArrayDataset, TASK_COMBINATIONS
from train_h76_pairwise_set_transformer_20260812 import (
    EXPANDED_COMBINATIONS,
    evaluate_expanded,
    load_expanded,
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


# RUMPL's 17-joint order is pelvis, right leg, left leg, trunk/head, then arms.
# This is the same H36M connectivity represented in the official GHT order.
H36M_EDGES = (
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
)
PELVIS = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 14


def canonical_pose_features(
    candidates: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    """Return GHT-style canonical features for B,C,J,3 candidates."""
    normalized = (candidates - mean) / std.clamp_min(1e-6)
    centered = normalized - normalized[:, :, PELVIS:PELVIS + 1]

    # GHT rotates the centered pose so the shoulder/pelvis plane has a fixed
    # orientation.  Use a stable Rodrigues rotation and handle the rare
    # anti-parallel/degenerate normal without NaNs.
    a = centered[:, :, LEFT_SHOULDER]
    b = centered[:, :, RIGHT_SHOULDER]
    c = centered[:, :, PELVIS]
    normal = torch.cross(c - a, b - a, dim=-1)
    normal_norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
    unit_normal = normal / normal_norm.clamp_min(1e-6)
    z = torch.zeros_like(unit_normal)
    z[..., 2] = 1.0
    axis = torch.cross(unit_normal, z, dim=-1)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    cosine = (unit_normal * z).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    safe_axis = axis / axis_norm.clamp_min(1e-6)
    # For parallel normals the identity is correct; for anti-parallel normals
    # a fixed x-axis gives the limiting pi rotation.
    fallback_x = torch.zeros_like(safe_axis)
    fallback_x[..., 0] = 1.0
    anti_parallel = (axis_norm < 1e-5) & (cosine < 0.0)
    safe_axis = torch.where(anti_parallel, fallback_x, safe_axis)
    angle = torch.acos(cosine)
    sin_angle = torch.sin(angle)
    kx, ky, kz = safe_axis.unbind(dim=-1)
    zero = torch.zeros_like(kx)
    skew = torch.stack(
        (zero, -kz, ky, kz, zero, -kx, -ky, kx, zero), dim=-1
    ).reshape(*safe_axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=candidates.device, dtype=candidates.dtype)
    eye = eye.reshape((1, 1, 3, 3)).expand(*safe_axis.shape[:-1], 3, 3)
    rotation = eye + sin_angle[..., None] * skew + (
        1.0 - cosine[..., None]
    ) * (skew @ skew)
    oriented = centered @ rotation.transpose(-1, -2)

    if mode == 0:
        return oriented.flatten(2)
    lengths = torch.stack(
        [torch.linalg.vector_norm(oriented[..., i, :] - oriented[..., j, :], dim=-1)
         for i, j in H36M_EDGES], dim=-1
    )
    if mode == 1:
        return torch.cat((oriented.flatten(2), lengths), dim=-1)
    if mode == 2:
        return lengths
    raise ValueError(f"unknown canonical GHT mode {mode}")


class CanonicalGHTE2Utility(nn.Module):
    """E2 scorer plus a zero-initialized canonical GHT whole-pose score."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, depth: int, mode: int):
        super().__init__()
        self.base = SetTransformerJointUtility(mean, std, depth)
        self.mode = int(mode)
        input_size = {0: 51, 1: 67, 2: 16}[self.mode]
        # Exact public GHT ScoreNN width/depth: 50-50-50-1, ReLU6.
        self.global_head = nn.Sequential(
            nn.Linear(input_size, 50), nn.ReLU6(),
            nn.Linear(50, 50), nn.ReLU6(),
            nn.Linear(50, 50), nn.ReLU6(),
            nn.Linear(50, 1),
        )
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)

    def forward(self, candidates, rays, candidate_masks, task_mask):
        local = self.base(candidates, rays, candidate_masks, task_mask)
        features = canonical_pose_features(
            candidates, self.base.pose_mean, self.base.pose_std, self.mode
        )
        global_score = self.global_head(features).squeeze(-1)
        return local + global_score[:, None, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-depth", type=int, default=2)
    parser.add_argument("--canonical-mode", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--target-temperature-mm", type=float, default=5.0)
    parser.add_argument("--oracle-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def task_loss(model, predictions, targets, rays, phase, temperature,
              target_temperature_m, oracle_weight):
    direct, oracle, ght = [], [], []
    for task in TASK_COMBINATIONS:
        predicted, true_delta, true_error, candidates, _ = predict_delta_expanded(
            model, predictions, targets, rays, task
        )
        direct.append(training_loss(predicted, true_delta, "balanced_rank"))
        target = F.softmax(-true_error / target_temperature_m, dim=-1)
        log_prob = F.log_softmax(-predicted / temperature, dim=-1)
        oracle.append(-(target * log_prob).sum(dim=-1).mean())
        if phase == "ght":
            weights = F.softmax(-predicted / temperature, dim=-1)
            expected = (weights * true_error).sum(dim=-1).mean()
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            ght.append((expected + 0.05 * fused_error) / 0.01)
    loss = torch.stack(direct).mean() + oracle_weight * torch.stack(oracle).mean()
    if ght:
        loss = loss + torch.stack(ght).mean()
    return loss


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_expanded(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if args.smoke_batches:
        train_indices = train_indices[:args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[:args.batch_size]
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = CanonicalGHTE2Utility(
        mean, std, args.attention_depth, args.canonical_mode
    ).to(device)
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model_best.pth.tar"
    phases = ([('direct', 5e-4)] * args.pretrain_epochs
              + [('ght', 1e-4)] * args.finetune_epochs)
    optimizer = None
    previous = None
    best_metric, best_epoch, history = math.inf, -1, []
    for epoch, (phase, lr) in enumerate(phases):
        if phase != previous:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            previous = phase
        model.train()
        losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(
                model, predictions, targets, rays, phase, args.temperature,
                args.target_temperature_mm / 1000.0, args.oracle_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        holdout_result = evaluate_expanded(
            model, holdout_loader, device, args.temperature
        )
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
        record = {"epoch": epoch, "phase": phase,
                  "train_loss": float(np.mean(losses)),
                  "holdout_selection_metric_mm": float(metric),
                  "holdout": holdout_result}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = float(metric), epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth,
                "canonical_mode": args.canonical_mode, "epoch": epoch,
                "phase": phase, "candidate_count": len(EXPANDED_COMBINATIONS),
                "temperature": args.temperature,
            }, checkpoint)
    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate_expanded(model, test_loader, device, args.temperature)
    result = {
        "method": "pairwise E2 + canonical GHT whole-pose score",
        "paper_basis": (
            "Generalizable Human Pose Triangulation (CVPR 2022) official "
            "ScoreNN preprocessing and body_lengths_mode"
        ),
        "canonical_mode": args.canonical_mode,
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": test_result, "args": vars(args),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
