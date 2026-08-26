#!/usr/bin/env python3
"""Evaluate the view-snap/bone E2-C2 scorer with the frozen E2-C2 temperatures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from train_h76_set_transformer_utility_20260811 import (
    ArrayDataset,
    SetTransformerJointUtility,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--v2-temperature", type=float, default=0.4)
    parser.add_argument("--v3-temperature", type=float, default=1.8)
    parser.add_argument("--v4-temperature", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    trainer.predict_task = extra.predict_task
    device = torch.device(f"cuda:{args.gpu}")
    arrays = trainer.load_arrays([args.cache], 22)
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    temperature = {
        "V2": args.v2_temperature,
        "V3": args.v3_temperature,
        "V4": args.v4_temperature,
    }
    per_seed = {}
    for seed in (0, 1):
        checkpoint = Path(args.checkpoint_root) / f"seed{seed}" / "model_best.pth.tar"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = SetTransformerJointUtility(
            state["mean"], state["std"], state["attention_depth"],
            stage_heads=state.get("stage_heads", False),
        ).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        result = trainer.evaluate(model, loader, device, temperature)
        per_seed[str(seed)] = {
            stage: {
                mode: result[stage][mode]["action_equal_all17_mm"]
                for mode in ("baseline", "hard", "soft", "oracle")
            }
            for stage in ("V2", "V3", "V4")
        }
    means = {
        stage: {
            mode: float(np.mean([per_seed[str(seed)][stage][mode] for seed in (0, 1)]))
            for mode in ("baseline", "hard", "soft", "oracle")
        }
        for stage in ("V2", "V3", "V4")
    }
    payload = {
        "method": "E2-C2 view-snap+bone-ray soft fusion, E2-C2 calibrated temperatures",
        "protocol": "strict flip=false, H36M S9/S11, action-equal All-17 absolute MPJPE",
        "cache": str(Path(args.cache).resolve()),
        "checkpoint_root": str(Path(args.checkpoint_root).resolve()),
        "temperatures": temperature,
        "per_seed_mm": per_seed,
        "mean_mm": means,
        "baseline_e2_c2_soft_cal": {"V2": 38.700, "V3": 29.486, "V4": 27.274},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
