#!/usr/bin/env python3
"""Holdout-only inference grid for a trained PoseDSAC-style E2 scorer."""
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
    parser.add_argument("--hypotheses", nargs="+", type=int, default=(48, 96))
    parser.add_argument("--proposal-temperatures", nargs="+", type=float, default=(0.4, 0.8, 1.2))
    parser.add_argument("--score-temperatures", nargs="+", type=float, default=(0.25, 0.5, 1.0))
    parser.add_argument("--group-modulo", nargs=2, type=int, default=(10, 0))
    parser.add_argument("--no-group-filter", action="store_true")
    parser.add_argument("--max-examples", type=int, default=2048)
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
    scorer = dsac.PoseScoreMLP(
        saved["features"], saved.get("sigmoid_score", False)
    ).to(device)
    scorer.load_state_dict(state["state_dict"], strict=True)
    e2, coord_mean, coord_std = dsac.load_e2(args.e2_checkpoint, device)
    bone_mean, bone_std = train_bone_stats(args.train_cache)
    bone_mean, bone_std = bone_mean.to(device), bone_std.to(device)
    arrays = trainer.load_arrays([args.cache], 22)
    if args.group_modulo and not args.no_group_filter:
        divisor, remainder = args.group_modulo
        keep = arrays["group_indices"] % divisor == remainder
        arrays = {key: value[keep] for key, value in arrays.items()}
    if args.max_examples:
        arrays = {key: value[: args.max_examples] for key, value in arrays.items()}
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    rows = []
    for hypotheses, proposal, score_temperature in itertools.product(
        args.hypotheses, args.proposal_temperatures, args.score_temperatures
    ):
        run_args = SimpleNamespace(**saved)
        run_args.hypotheses = hypotheses
        run_args.proposal_temperature = proposal
        run_args.score_temperature = score_temperature
        result = dsac.evaluate(
            scorer, e2, loader, device, run_args,
            coord_mean, coord_std, bone_mean, bone_std,
        )
        headline = float(np.mean([
            result[stage]["weighted"]["action_equal_all17_mm"]
            for stage in ("V2", "V3", "V4")
        ]))
        row = {
            "hypotheses": hypotheses,
            "proposal_temperature": proposal,
            "score_temperature": score_temperature,
            "headline_mm": headline,
            "weighted_mm": {
                stage: result[stage]["weighted"]["action_equal_all17_mm"]
                for stage in ("V2", "V3", "V4")
            },
            "sampled_oracle_mm": {
                stage: result[stage]["sampled_oracle"]["action_equal_all17_mm"]
                for stage in ("V2", "V3", "V4")
            },
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    rows.sort(key=lambda row: row["headline_mm"])
    payload = {
        "protocol": "train internal group-index holdout only; no S9/S11 tuning",
        "num_examples": len(arrays["targets"]),
        "checkpoint": args.checkpoint,
        "rows": rows,
        "best": rows[0],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
