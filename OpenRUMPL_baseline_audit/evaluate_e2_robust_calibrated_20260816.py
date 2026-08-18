#!/usr/bin/env python3
"""Evaluate the H2 33-candidate scorer with fixed per-cardinality temperatures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_current_e2_robust_20260815 as wrapper
import train_e2_v234_universal_20260812 as trainer
from train_h76_set_transformer_utility_20260811 import ArrayDataset, SetTransformerJointUtility


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--v2-temperature", type=float, default=0.4)
    p.add_argument("--v3-temperature", type=float, default=1.8)
    p.add_argument("--v4-temperature", type=float, default=1.8)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main():
    args = parse_args()
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL + wrapper.ORIGINAL
    device = torch.device(f"cuda:{args.gpu}")
    arrays = trainer.load_arrays([args.cache], 33)
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    temperatures = {"V2": args.v2_temperature, "V3": args.v3_temperature, "V4": args.v4_temperature}
    per_seed = {}
    for seed in (0, 1):
        checkpoint = Path(args.checkpoint_root) / f"seed{seed}" / "model_best.pth.tar"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = SetTransformerJointUtility(
            state["mean"], state["std"], state["attention_depth"],
            stage_heads=state.get("stage_heads", False),
        ).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        result = trainer.evaluate(model, loader, device, temperatures)
        per_seed[str(seed)] = {
            stage: result[stage]["soft"]["action_equal_all17_mm"]
            for stage in ("V2", "V3", "V4")
        }
    means = {stage: float(np.mean([per_seed[str(seed)][stage] for seed in (0, 1)])) for stage in ("V2", "V3", "V4")}
    stds = {stage: float(np.std([per_seed[str(seed)][stage] for seed in (0, 1)])) for stage in ("V2", "V3", "V4")}
    payload = {
        "method": "H2 33-candidate E2 soft with fixed per-cardinality temperature calibration",
        "protocol": "strict flip=false, H36M S9/S11, action-equal All-17 absolute MPJPE",
        "cache": str(Path(args.cache).resolve()),
        "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        "candidate_count": 33,
        "temperatures": temperatures,
        "per_seed_mm": per_seed,
        "mean_mm": means,
        "std_mm": stds,
        "selection": "V2 temperature 0.4 reused from pre-registered holdout calibration; V3/V4 retain GHT T=1.8",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
