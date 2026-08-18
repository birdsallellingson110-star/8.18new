#!/usr/bin/env python3
"""Evaluate pairwise E2 fusion temperature on holdout and S9/S11."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset, TASK_COMBINATIONS
from train_h76_pairwise_set_transformer_20260812 import (
    load_expanded,
    predict_delta_expanded,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def action_equal(values, actions):
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def evaluate(model, loader, device, temperatures):
    stores = {float(t): {"V3": [], "V4": []} for t in temperatures}
    actions = {"V3": [], "V4": []}
    with torch.inference_mode():
        for predictions, targets, rays, batch_actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            batch_actions = batch_actions.numpy()
            for task in TASK_COMBINATIONS:
                stage = f"V{len(task)}"
                predicted, _, _, candidates, _ = predict_delta_expanded(
                    model, predictions, targets, rays, task
                )
                for temperature in temperatures:
                    weights = F.softmax(-predicted / float(temperature), dim=-1)
                    fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                    stores[float(temperature)][stage].append(
                        torch.linalg.vector_norm(fused - targets, dim=-1).cpu().numpy() * 1000.0
                    )
                actions[stage].append(batch_actions.copy())
    result = {}
    for temperature in temperatures:
        result[str(float(temperature))] = {}
        for stage in ("V3", "V4"):
            values = np.concatenate(stores[float(temperature)][stage])
            stage_actions = np.concatenate(actions[stage])
            result[str(float(temperature))][stage] = {
                "action_equal_all17_mm": action_equal(values, stage_actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
    return result


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    train = load_expanded(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SetTransformerJointUtility(mean, std, checkpoint["attention_depth"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    temperatures = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    result = {
        "checkpoint": args.checkpoint,
        "holdout": evaluate(model, holdout_loader, device, temperatures),
        "S9_S11_final_once": evaluate(model, test_loader, device, temperatures),
        "temperatures": temperatures,
    }
    Path(args.output).resolve().write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

