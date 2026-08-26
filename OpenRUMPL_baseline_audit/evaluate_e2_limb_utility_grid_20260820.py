#!/usr/bin/env python3
"""Calibrate limb-utility fusion temperatures on an internal train holdout."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_limb_utility_20260820 as limb
import train_e2_pose_dsac_20260820 as dsac
import train_e2_v234_universal_20260812 as trainer
from diagnose_e2_structured_candidates_20260820 import train_bone_stats
from train_h76_hypothesis_utility_20260811 import ArrayDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--v2-temperatures", nargs="+", type=float, default=(0.1, 0.2, 0.4, 0.8))
    parser.add_argument("--v34-temperatures", nargs="+", type=float, default=(0.2, 0.4, 0.8, 1.2))
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--no-group-filter", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    trainer.predict_task = extra.predict_task
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved = state["args"]
    e2, coord_mean, coord_std = dsac.load_e2(args.e2_checkpoint, device)
    model = limb.LimbUtility(coord_mean, coord_std).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    arrays = trainer.load_arrays([args.cache], 22)
    if args.no_group_filter:
        selected = np.arange(len(arrays["targets"]))
    else:
        selected = np.flatnonzero(arrays["group_indices"] % 10 == 0)
    if args.max_examples:
        selected = selected[: args.max_examples]
    arrays = {key: value[selected] for key, value in arrays.items()}
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    rows = []
    for v2, v34 in itertools.product(args.v2_temperatures, args.v34_temperatures):
        run_args = SimpleNamespace(**saved)
        run_args.v2_temperature = v2
        run_args.v34_temperature = v34
        result = limb.evaluate(model, e2, loader, device, bone_mean, bone_std, run_args)
        weighted = {
            stage: result[stage]["soft"]["action_equal_all17_mm"]
            for stage in ("V2", "V3", "V4")
        }
        row = {"v2_temperature": v2, "v34_temperature": v34,
               "headline_mm": float(np.mean(list(weighted.values()))),
               "weighted_mm": weighted}
        rows.append(row)
        print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: row["headline_mm"])
    payload = {
        "protocol": "first 2048 examples of train group_index%10==0; S9/S11 untouched",
        "checkpoint": args.checkpoint,
        "rows": rows,
        "best": rows[0],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
