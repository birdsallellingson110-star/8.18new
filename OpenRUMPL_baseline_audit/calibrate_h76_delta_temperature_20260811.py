#!/usr/bin/env python3
"""Select one C2b soft temperature on train holdout, then test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import evaluate
from train_h76_hypothesis_utility_20260811 import (
    ArrayDataset,
    JointUtilityScorer,
    load_arrays,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model = JointUtilityScorer(checkpoint["mean"], checkpoint["std"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def main():
    args = parse_args()
    if any(value <= 0 for value in args.temperatures):
        raise ValueError("temperatures must be positive")
    device = torch.device(f"cuda:{args.gpu}")
    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout_indices = np.flatnonzero(train["group_indices"] % 10 == 0)
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    models = [load_model(path, device) for path in args.checkpoints]
    sweep = []
    for temperature in args.temperatures:
        per_seed = []
        for model in models:
            result = evaluate(model, holdout_loader, device, temperature)
            metric = 0.5 * (
                result["V3"]["soft"]["action_equal_all17_mm"]
                + result["V4"]["soft"]["action_equal_all17_mm"]
            )
            per_seed.append({"metric_mm": metric, "result": result})
        sweep.append({
            "temperature": temperature,
            "mean_holdout_metric_mm": float(np.mean([
                value["metric_mm"] for value in per_seed
            ])),
            "per_seed": per_seed,
        })
        print(json.dumps(sweep[-1]), flush=True)
    selected = min(sweep, key=lambda value: value["mean_holdout_metric_mm"])
    temperature = selected["temperature"]
    test_results = [
        evaluate(model, test_loader, device, temperature) for model in models
    ]
    aggregate = {}
    for stage in ("V3", "V4"):
        values = np.asarray([
            result[stage]["soft"]["action_equal_all17_mm"]
            for result in test_results
        ])
        aggregate[stage] = {
            "mean_mm": float(values.mean()),
            "std_population_mm": float(values.std()),
            "per_seed_mm": values.tolist(),
        }
    payload = {
        "selection_split": "H36M train subjects, group_index modulo 10 holdout",
        "test_split": "S9/S11 evaluated only for holdout-selected temperature",
        "temperatures": args.temperatures,
        "holdout_sweep": sweep,
        "selected_temperature": temperature,
        "test_per_seed": test_results,
        "test_aggregate": aggregate,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected_temperature": temperature,
        "test_aggregate": aggregate,
    }, indent=2))


if __name__ == "__main__":
    main()
