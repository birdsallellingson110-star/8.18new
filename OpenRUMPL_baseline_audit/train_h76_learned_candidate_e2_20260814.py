#!/usr/bin/env python3
"""Retrain the E2 scorer after adding learned triangulation candidates.

This keeps the E2 Set-Transformer, loss, split, and temperature unchanged. The
only experimental variable is five extra candidates generated from the frozen
ray-only learnable triangulation network (one for each V3/V4 subset).
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

import train_h76_pairwise_set_transformer_20260812 as base
from train_h76_hypothesis_utility_20260811 import ArrayDataset, TASK_COMBINATIONS
from train_h76_pairwise_oracle_rank_20260812 import task_loss
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


LEARNED_COMBINATIONS = tuple(
    combo for combo in base.COMBINATIONS if len(combo) >= 3
)
EXPANDED_COMBINATIONS = base.EXPANDED_COMBINATIONS + LEARNED_COMBINATIONS
# Patch the imported E2 helpers' module-global candidate list. Their code is
# otherwise identical to the formal pairwise E2 implementation.
base.EXPANDED_COMBINATIONS = EXPANDED_COMBINATIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
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
    return p.parse_args()


def load_expanded(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    order = np.argsort(arrays["group_indices"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["group_indices"])) != len(arrays["group_indices"]):
        raise ValueError("duplicate train group indices")
    expected = (len(EXPANDED_COMBINATIONS), 17, 3)
    if arrays["predictions"].shape[1:] != expected:
        raise ValueError(f"bad expanded prediction shape {arrays['predictions'].shape}; expected {expected}")
    return arrays


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_expanded(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    expected = (len(EXPANDED_COMBINATIONS), 17, 3)
    if validation["predictions"].shape[1:] != expected:
        raise ValueError(f"bad validation prediction shape {validation['predictions'].shape}")
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
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
                model, predictions, targets, rays, phase, args.temperature,
                args.target_temperature_mm / 1000.0, args.oracle_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = base.evaluate_expanded(
            model, holdout_loader, device, args.temperature
        )
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
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
                "temperature": args.temperature,
            }, checkpoint)
    best = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = base.evaluate_expanded(model, test_loader, device, args.temperature)
    result = {
        "method": "E2 Set Transformer + learned triangulation candidate extension",
        "paper_basis": [
            "GHT-style multi-hypothesis scoring",
            "Learnable Triangulation (ICCV 2019) reliability candidate",
        ],
        "candidate_count": len(EXPANDED_COMBINATIONS),
        "candidate_combinations": [list(c) for c in EXPANDED_COMBINATIONS],
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": test_result, "args": vars(args),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
