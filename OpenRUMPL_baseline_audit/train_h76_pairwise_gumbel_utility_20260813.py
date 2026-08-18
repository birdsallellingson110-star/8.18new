#!/usr/bin/env python3
"""Pairwise E2 with GHT's Gumbel-Softmax hypothesis exploration.

Only the GHT fine-tuning phase uses a differentiable Gumbel-Softmax sample to
avoid early collapse onto the full-input candidate.  Direct counterfactual and
soft oracle-target losses are unchanged.  Evaluation always uses deterministic
softmax weighted averaging, so the reported S9/S11 path has no sampling noise.
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
    evaluate_expanded,
    load_expanded,
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-depth", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--gumbel-tau", type=float, default=1.0)
    parser.add_argument("--target-temperature-mm", type=float, default=5.0)
    parser.add_argument("--oracle-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def task_loss(model, predictions, targets, rays, phase, temperature,
              gumbel_tau, target_temperature_m, oracle_weight):
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
            # GHT uses Gumbel-Softmax to keep probability mass exploring
            # multiple hypotheses.  This affects training only; evaluation
            # in evaluate_expanded remains deterministic softmax.
            weights = F.gumbel_softmax(
                -predicted / temperature,
                tau=gumbel_tau,
                hard=False,
                dim=-1,
            )
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
    if args.gumbel_tau <= 0 or args.target_temperature_mm <= 0:
        raise ValueError("gumbel tau and target temperature must be positive")
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
        train_indices = train_indices[: args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[: args.batch_size]
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = SetTransformerJointUtility(mean, std, args.attention_depth).to(device)
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "model_best.pth.tar"
    phases = (
        [("direct", 5e-4)] * args.pretrain_epochs
        + [("ght_gumbel", 1e-4)] * args.finetune_epochs
    )
    optimizer = None
    previous_phase = None
    best_metric, best_epoch, history = math.inf, -1, []
    for epoch, (phase, lr) in enumerate(phases):
        if phase != previous_phase:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            previous_phase = phase
        model.train()
        losses = []
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(
                model, predictions, targets, rays, phase, args.temperature,
                args.gumbel_tau, args.target_temperature_mm / 1000.0,
                args.oracle_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
            if args.smoke_batches and batch_index + 1 >= args.smoke_batches:
                break
        holdout_result = evaluate_expanded(model, holdout_loader, device, args.temperature)
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
        record = {
            "epoch": epoch, "phase": phase,
            "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": float(metric),
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = float(metric), epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth, "epoch": epoch,
                "phase": phase, "candidate_count": len(EXPANDED_COMBINATIONS),
                "temperature": args.temperature, "gumbel_tau": args.gumbel_tau,
                "method": "pairwise_e2_gumbel_ght",
            }, checkpoint)

    best = torch.load(checkpoint, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate_expanded(model, test_loader, device, args.temperature)
    result = {
        "method": "pairwise E2 + GHT Gumbel-Softmax exploration in fine-tuning",
        "paper_basis": "Generalizable Human Pose Triangulation (CVPR 2022), Gumbel-Softmax hypothesis scoring",
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": test_result,
        "args": vars(args),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
