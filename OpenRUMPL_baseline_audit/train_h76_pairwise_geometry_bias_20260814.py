#!/usr/bin/env python3
"""Geometry-biased candidate utility on the frozen pairwise E2 pool.

The public Geometry-Biased Transformer (GBT) uses calibrated ray geometry in
attention rather than relying only on learned view identities.  This control
adds a zero-initialized, joint-conditioned residual head using permutation-
invariant Plucker/ray summaries for each candidate's active views.  The E2
candidate Set Transformer, RUMPL features, candidate pool, and losses are
otherwise unchanged.
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


def masked_stats(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean/min/max over the last dimension for B,C,J,V or B,C,J,V,V."""
    mask = mask.bool()
    expanded = mask[None, :, None]
    count = expanded.sum(dim=-1).clamp_min(1).to(values.dtype)
    mean = (values * expanded).sum(dim=-1) / count
    large = torch.finfo(values.dtype).max
    minimum = values.masked_fill(~expanded, large).min(dim=-1).values
    maximum = values.masked_fill(~expanded, -large).max(dim=-1).values
    present = expanded.any(dim=-1)
    minimum = torch.where(present, minimum, torch.zeros_like(minimum))
    maximum = torch.where(present, maximum, torch.zeros_like(maximum))
    return torch.stack((mean, minimum, maximum), dim=-1)


def geometry_features(
    rays: torch.Tensor, candidate_masks: torch.Tensor, task_mask: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Build candidate geometry summaries with shape B,C,J,F."""
    del task_mask
    direction = F.normalize(rays[..., :3], dim=-1)
    origin = rays[..., 3:6]
    moment = torch.cross(origin, direction, dim=-1)
    batch, joints, views, _ = direction.shape
    count = candidate_masks.sum(dim=-1).clamp_min(1).to(direction.dtype)
    active = candidate_masks[None, :, None, :, None].bool()

    d = direction[:, None].expand(-1, candidate_masks.shape[0], -1, -1, -1)
    o = origin[:, None].expand_as(d)
    m = moment[:, None].expand_as(d)
    mean_d = (d * active).sum(dim=-2) / count[None, :, None, None]
    mean_o = (o * active).sum(dim=-2) / count[None, :, None, None]
    mean_m = (m * active).sum(dim=-2) / count[None, :, None, None]

    # Pairwise GBT Plucker distance and ray-angle summaries.  The diagonal is
    # excluded; every candidate has at least two active views.
    d_i = direction[:, :, :, None, :]
    d_j = direction[:, :, None, :, :]
    o_i = origin[:, :, :, None, :]
    o_j = origin[:, :, None, :, :]
    cross = torch.cross(d_i, d_j, dim=-1)
    cross_norm = torch.linalg.vector_norm(cross, dim=-1)
    baseline = o_j - o_i
    skew = (baseline * cross).sum(dim=-1).abs() / cross_norm.clamp_min(1e-7)
    point_line = torch.linalg.vector_norm(
        torch.cross(baseline, d_i.expand_as(baseline), dim=-1), dim=-1
    )
    ray_distance = torch.where(cross_norm > 1e-5, skew, point_line)
    ray_distance = torch.log1p(ray_distance / 0.05)
    angle = cross_norm.clamp(0.0, 1.0)
    mask_bool = candidate_masks.bool()
    pair_mask = mask_bool[:, :, None] & mask_bool[:, None, :]
    pair_mask = pair_mask & ~torch.eye(views, dtype=torch.bool, device=rays.device)[None]
    # N,J,V,V -> N,C,J,V,V.
    distance = ray_distance[:, None].expand(-1, candidate_masks.shape[0], -1, -1, -1)
    angle = angle[:, None].expand_as(distance)
    distance = distance.reshape(batch, candidate_masks.shape[0], joints, -1)
    angle = angle.reshape(batch, candidate_masks.shape[0], joints, -1)
    pair_mask_flat = pair_mask.reshape(candidate_masks.shape[0], -1)
    pair_summary = torch.cat(
        (masked_stats(distance, pair_mask_flat), masked_stats(angle, pair_mask_flat)), dim=-1
    )

    if mode == "angle_distance":
        # Six explicit geometry statistics, plus the 3 eigen-spectrum values
        # already used by E2.  This isolates geometry beyond raw Plucker means.
        projection = (
            torch.eye(3, device=rays.device, dtype=rays.dtype)
            - direction.unsqueeze(-1) * direction.unsqueeze(-2)
        )
        weight = rays[..., 6:7].clamp(0, 1) + 0.05
        normal = torch.einsum(
            "cv,bjvxy->bcjxy", candidate_masks, weight.unsqueeze(-1) * projection
        )
        eigen = torch.linalg.eigvalsh(normal).clamp_min(1e-7)
        spectrum = torch.log(eigen / eigen.sum(dim=-1, keepdim=True))
        return torch.cat((pair_summary, spectrum), dim=-1)
    if mode != "full":
        raise ValueError(f"unknown geometry mode {mode}")
    # Scale positions/moments to the same numerical range as normalized E2
    # features; no camera ID is used, only calibrated ray geometry.
    pooled = torch.cat((mean_d, mean_o / 3.0, mean_m / 3.0), dim=-1)
    return torch.cat((pooled, pair_summary), dim=-1)


class GeometryBiasE2Utility(nn.Module):
    """E2 plus zero-initialized, joint-conditioned geometry residual."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, depth: int, mode: str):
        super().__init__()
        self.base = SetTransformerJointUtility(mean, std, depth)
        self.mode = mode
        geometry_dim = {"full": 9 + 6, "angle_distance": 6 + 3}[mode]
        self.joint_embedding = nn.Parameter(torch.zeros(17, 16))
        self.geometry_head = nn.Sequential(
            nn.Linear(geometry_dim + 16, 64), nn.ReLU6(),
            nn.Linear(64, 32), nn.ReLU6(), nn.Linear(32, 1),
        )
        nn.init.zeros_(self.geometry_head[-1].weight)
        nn.init.zeros_(self.geometry_head[-1].bias)

    def forward(self, candidates, rays, candidate_masks, task_mask):
        local = self.base(candidates, rays, candidate_masks, task_mask)
        geom = geometry_features(rays, candidate_masks, task_mask, self.mode)
        batch, count, joints, _ = geom.shape
        joint = self.joint_embedding[None, None].expand(batch, count, -1, -1)
        residual = self.geometry_head(torch.cat((geom, joint), dim=-1)).squeeze(-1)
        return local + residual.permute(0, 2, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--attention-depth", type=int, default=2)
    p.add_argument("--geometry-mode", choices=("full", "angle_distance"), default="full")
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.8)
    p.add_argument("--target-temperature-mm", type=float, default=5.0)
    p.add_argument("--oracle-weight", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--smoke-batches", type=int, default=0)
    return p.parse_args()


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
    return loss + (torch.stack(ght).mean() if ght else 0.0)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_expanded(args.train_shards)
    val_npz = np.load(args.validation_cache)
    validation = {key: val_npz[key] for key in val_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if args.smoke_batches:
        train_indices = train_indices[:args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[:args.batch_size]
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = GeometryBiasE2Utility(mean, std, args.attention_depth, args.geometry_mode).to(device)
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
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "model_best.pth.tar"
    phases = ([('direct', 5e-4)] * args.pretrain_epochs
              + [('ght', 1e-4)] * args.finetune_epochs)
    optimizer = None; previous = None; best_metric = math.inf; best_epoch = -1; history = []
    for epoch, (phase, lr) in enumerate(phases):
        if phase != previous:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            previous = phase
        model.train(); losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(
                model, predictions, targets, rays, phase, args.temperature,
                args.target_temperature_mm / 1000.0, args.oracle_weight,
            )
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.item()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches: break
        holdout_result = evaluate_expanded(model, holdout_loader, device, args.temperature)
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
        record = {"epoch": epoch, "phase": phase, "train_loss": float(np.mean(losses)),
                  "holdout_selection_metric_mm": float(metric), "holdout": holdout_result}
        history.append(record); print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = float(metric), epoch
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "attention_depth": args.attention_depth, "geometry_mode": args.geometry_mode,
                        "epoch": epoch, "phase": phase, "candidate_count": len(EXPANDED_COMBINATIONS),
                        "temperature": args.temperature}, checkpoint)
    best = torch.load(checkpoint, map_location=device); model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate_expanded(model, test_loader, device, args.temperature)
    result = {"method": "pairwise E2 + geometry-biased candidate utility residual",
              "paper_basis": "Geometry-Biased Transformer (calibrated ray geometry bias)",
              "geometry_mode": args.geometry_mode, "candidate_count": len(EXPANDED_COMBINATIONS),
              "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
              "history": history, "S9_S11_final_once": test_result, "args": vars(args)}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
