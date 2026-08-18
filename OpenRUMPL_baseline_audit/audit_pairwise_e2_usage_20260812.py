#!/usr/bin/env python3
"""Audit whether pairwise E2 actually uses the new geometry hypotheses."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_hypothesis_utility_20260811 import ArrayDataset, TASK_COMBINATIONS
from train_h76_pairwise_set_transformer_20260812 import (
    EXPANDED_COMBINATIONS,
    load_expanded,
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    train = load_expanded(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SetTransformerJointUtility(mean, std, checkpoint["attention_depth"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )

    stage_stats = {stage: {
        "samples": 0, "pairwise_weight_sum": 0.0,
        "pairwise_top1": 0, "baseline_top1": 0,
        "pairwise_selected_error_mm": [], "baseline_selected_error_mm": [],
        "soft_error_mm": [], "baseline_error_mm": [], "oracle_error_mm": [],
    } for stage in ("V3", "V4")}
    with torch.inference_mode():
        for predictions, targets, rays, _ in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            for task in TASK_COMBINATIONS:
                stage = f"V{len(task)}"
                predicted, _, true_error, candidates, baseline_local = predict_delta_expanded(
                    model, predictions, targets, rays, task
                )
                weights = F.softmax(-predicted, dim=-1)
                available = [
                    i for i, combo in enumerate(EXPANDED_COMBINATIONS)
                    if set(combo).issubset(task)
                ]
                pairwise_local = torch.tensor(
                    [index >= 11 for index in available], device=device
                )
                pairwise_weight = weights[..., pairwise_local].sum(dim=-1)
                top = predicted.argmin(dim=-1)
                top_global = torch.tensor(available, device=device)[top]
                pairwise_top = top_global >= 11
                hard_error = true_error.gather(-1, top[..., None]).squeeze(-1)
                baseline_error = true_error[..., baseline_local]
                soft_pose = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                soft_error = torch.linalg.vector_norm(soft_pose - targets, dim=-1)
                oracle_error = true_error.min(dim=-1).values
                stats = stage_stats[stage]
                stats["samples"] += int(pairwise_weight.numel())
                stats["pairwise_weight_sum"] += float(pairwise_weight.sum().item())
                stats["pairwise_top1"] += int(pairwise_top.sum().item())
                stats["baseline_top1"] += int((top == baseline_local).sum().item())
                stats["pairwise_selected_error_mm"].append(
                    hard_error[pairwise_top].cpu().numpy() * 1000.0
                )
                stats["baseline_selected_error_mm"].append(
                    hard_error[~pairwise_top].cpu().numpy() * 1000.0
                )
                stats["soft_error_mm"].append(soft_error.cpu().numpy() * 1000.0)
                stats["baseline_error_mm"].append(baseline_error.cpu().numpy() * 1000.0)
                stats["oracle_error_mm"].append(oracle_error.cpu().numpy() * 1000.0)

    result = {"checkpoint": args.checkpoint, "candidate_count": len(EXPANDED_COMBINATIONS)}
    for stage, stats in stage_stats.items():
        result[stage] = {
            "samples": stats["samples"],
            "mean_pairwise_soft_weight": stats["pairwise_weight_sum"] / stats["samples"],
            "pairwise_top1_fraction": stats["pairwise_top1"] / stats["samples"],
            "baseline_top1_fraction": stats["baseline_top1"] / stats["samples"],
            "pairwise_top1_error_mm": float(np.concatenate(stats["pairwise_selected_error_mm"]).mean())
            if stats["pairwise_selected_error_mm"] and sum(map(len, stats["pairwise_selected_error_mm"])) else None,
            "non_pairwise_top1_error_mm": float(np.concatenate(stats["baseline_selected_error_mm"]).mean())
            if stats["baseline_selected_error_mm"] and sum(map(len, stats["baseline_selected_error_mm"])) else None,
            "soft_error_mm": float(np.concatenate(stats["soft_error_mm"]).mean()),
            "baseline_error_mm": float(np.concatenate(stats["baseline_error_mm"]).mean()),
            "oracle_error_mm": float(np.concatenate(stats["oracle_error_mm"]).mean()),
        }
    Path(args.output).resolve().write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

