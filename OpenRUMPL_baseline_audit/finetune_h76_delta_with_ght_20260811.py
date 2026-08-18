#!/usr/bin/env python3
"""Fine-tune a pretrained counterfactual delta head with GHT task losses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import (
    evaluate,
    predict_delta,
    training_loss,
)
from train_h76_hypothesis_utility_20260811 import (
    ArrayDataset,
    JointUtilityScorer,
    TASK_COMBINATIONS,
    load_arrays,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--variant", choices=("ght", "ght_monotonic"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    initial = torch.load(args.init_checkpoint, map_location="cpu")
    model = JointUtilityScorer(initial["mean"], initial["std"])
    model.load_state_dict(initial["state_dict"], strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(0),
        num_workers=args.workers, pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pth.tar"
    best_metric = math.inf
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            direct_losses = []
            ght_losses = []
            fused_joint_errors = []
            for task_combo in TASK_COMBINATIONS:
                predicted, true_delta, true_error, candidates, _ = predict_delta(
                    model, predictions, targets, rays, task_combo
                )
                direct_losses.append(
                    training_loss(predicted, true_delta, "balanced_rank")
                )
                weights = F.softmax(-predicted, dim=-1)
                expected_error = (weights * true_error).sum(dim=-1).mean()
                fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                fused_error = torch.linalg.vector_norm(
                    fused - targets, dim=-1
                )
                fused_joint_errors.append(fused_error)
                # Official GHT coefficients: expectation=1, estimate=0.05.
                # Divide by 10 mm to share C2's normalized delta scale.
                ght_losses.append(
                    (expected_error + 0.05 * fused_error.mean()) / 0.01
                )
            loss = torch.stack(direct_losses).mean() + torch.stack(ght_losses).mean()
            monotonic = torch.zeros((), device=device)
            if args.variant == "ght_monotonic":
                mean_v3_error = torch.stack(fused_joint_errors[:4]).mean(dim=0)
                v4_error = fused_joint_errors[4]
                monotonic = F.relu(v4_error - mean_v3_error).mean() / 0.01
                loss = loss + monotonic
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        holdout_result = evaluate(model, holdout_loader, device, 1.0)
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "holdout_selection_metric_mm": metric,
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": initial["mean"],
                "std": initial["std"], "variant": args.variant,
                "epoch": epoch,
            }, checkpoint_path)

    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = evaluate(model, test_loader, device, 1.0)
    payload = {
        "variant": args.variant,
        "init_checkpoint": str(Path(args.init_checkpoint).resolve()),
        "best_epoch": best_epoch,
        "loss": (
            "C2b balanced delta + GHT expected risk + 0.05 weighted-estimate"
            + (" + V3-to-V4 monotonic" if args.variant == "ght_monotonic" else "")
        ),
        "history": history,
        "S9_S11_final_once": test_result,
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(test_result, indent=2), flush=True)


if __name__ == "__main__":
    main()
