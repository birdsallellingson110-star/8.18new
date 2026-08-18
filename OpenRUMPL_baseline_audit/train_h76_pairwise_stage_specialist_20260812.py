#!/usr/bin/env python3
"""Train a view-count-specialized pairwise E2 utility head.

This is a strict ablation of the shared E2 soft-oracle ranker: candidate
generation, losses, temperature, subject holdout and Set Transformer are
unchanged; only the task set seen by a checkpoint is restricted to V3 or V4.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import ArrayDataset, TASK_COMBINATIONS
from train_h76_pairwise_set_transformer_20260812 import (
    EXPANDED_COMBINATIONS,
    load_expanded,
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("v3", "v4"), required=True)
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--attention-depth", type=int, default=2)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.8)
    p.add_argument("--target-temperature-mm", type=float, default=5.0)
    p.add_argument("--oracle-weight", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-validation-samples", type=int, default=0)
    return p.parse_args()


def stage_tasks(stage):
    return TASK_COMBINATIONS[:4] if stage == "v3" else TASK_COMBINATIONS[4:]


def task_loss(model, predictions, targets, rays, tasks, phase, temperature,
              target_temperature_m, oracle_weight):
    direct, oracle, ght = [], [], []
    for task in tasks:
        # The helper's second return is the exact baseline-relative delta.
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


@torch.inference_mode()
def evaluate_stage(model, loader, device, tasks, temperature, stage_name):
    model.eval()
    values = []
    baselines = []
    actions = []
    for predictions, targets, rays, batch_actions in loader:
        predictions = predictions.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        rays = rays.to(device, non_blocking=True)
        batch_actions = batch_actions.numpy()
        for task in tasks:
            predicted, _, _, candidates, baseline_local = predict_delta_expanded(
                model, predictions, targets, rays, task
            )
            weights = F.softmax(-predicted / temperature, dim=-1)
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            error = torch.linalg.vector_norm(fused - targets, dim=-1)
            baseline = torch.linalg.vector_norm(
                candidates[:, baseline_local] - targets, dim=-1
            )
            values.append(error.cpu().numpy())
            baselines.append(baseline.cpu().numpy())
            actions.append(batch_actions.copy())
    values = np.concatenate(values, axis=0)
    baselines = np.concatenate(baselines, axis=0)
    actions = np.concatenate(actions, axis=0)
    action_equal = lambda x: float(np.mean([
        x[actions == a].mean() for a in sorted(set(actions.tolist()))
    ]) * 1000.0)
    return {
        "stage": stage_name,
        "soft_action_equal_all17_mm": action_equal(values),
        "center_baseline_action_equal_all17_mm": action_equal(baselines),
        "soft_frame_weighted_all17_mm": float(values.mean() * 1000.0),
        "baseline_frame_weighted_all17_mm": float(baselines.mean() * 1000.0),
        "delta_action_equal_mm": float(action_equal(values) - action_equal(baselines)),
    }


def main():
    args = parse_args()
    if args.target_temperature_mm <= 0 or args.oracle_weight < 0:
        raise ValueError("target temperature must be positive and oracle weight non-negative")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    tasks = stage_tasks(args.stage)
    train = load_expanded(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if args.max_train_samples:
        train_indices = train_indices[:args.max_train_samples]
        holdout_indices = holdout_indices[:max(1, args.max_train_samples // 10)]
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = SetTransformerJointUtility(mean, std, args.attention_depth).to(device)
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    validation_indices = np.arange(len(validation["targets"]))
    if args.max_validation_samples:
        validation_indices = validation_indices[:args.max_validation_samples]
    test_loader = DataLoader(
        ArrayDataset(validation, validation_indices),
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
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(
                model, predictions, targets, rays, tasks, phase,
                args.temperature, args.target_temperature_mm / 1000.0,
                args.oracle_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate_stage(
            model, holdout_loader, device, tasks, args.temperature, args.stage
        )
        metric = holdout_result["soft_action_equal_all17_mm"]
        record = {
            "epoch": epoch, "phase": phase, "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": float(metric), "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = float(metric), epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth, "epoch": epoch,
                "phase": phase, "candidate_count": len(EXPANDED_COMBINATIONS),
                "temperature": args.temperature, "specialist_stage": args.stage,
            }, checkpoint)
    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate_stage(
        model, test_loader, device, tasks, args.temperature, args.stage
    )
    result = {
        "method": "E2 pairwise Set Transformer + view-count specialist",
        "stage": args.stage,
        "paper_basis": "view-count-specific ablation of the shared GHT-style utility scorer",
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "candidate_combinations": [list(c) for c in EXPANDED_COMBINATIONS],
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": test_result, "args": vars(args),
    }
    (out / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
