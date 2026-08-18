#!/usr/bin/env python3
"""Export learned-triangulation candidates and measure oracle complementarity."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, JOINT_NAMES
from train_h76_learnable_triangulation_20260814 import (
    ALL_COMBINATIONS,
    LearnableTriangulation,
)


LEARNED_COMBINATIONS = tuple(combo for combo in ALL_COMBINATIONS if len(combo) >= 3)


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("independent", "cross"), required=True)
    parser.add_argument("--attention-depth", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def export_one(
    input_path: str,
    output_path: Path,
    model: LearnableTriangulation,
    device: torch.device,
    batch_size: int,
) -> dict:
    source = np.load(input_path)
    arrays = {key: source[key] for key in source.files}
    rays = torch.from_numpy(arrays["rays"])
    loader = DataLoader(TensorDataset(rays), batch_size=batch_size, shuffle=False)
    generated = [[] for _ in LEARNED_COMBINATIONS]
    model.eval()
    model.temperature = 1.0
    with torch.inference_mode():
        for (batch_rays,) in loader:
            batch_rays = batch_rays.to(device, non_blocking=True)
            for index, combo in enumerate(LEARNED_COMBINATIONS):
                pose, _ = model(batch_rays, combo)
                generated[index].append(pose.cpu().numpy().astype(np.float32))
    learned = np.stack([np.concatenate(chunks, axis=0) for chunks in generated], axis=1)
    arrays["predictions"] = np.concatenate((arrays["predictions"], learned), axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)

    errors = np.linalg.norm(arrays["predictions"] - arrays["targets"][:, None], axis=-1)
    actions = arrays["actions"]
    summary = {"file": str(output_path), "groups": int(len(actions))}
    for count in (3, 4):
        stage_values = {"existing": [], "expanded": []}
        for combo in itertools.combinations(range(4), count):
            # The first 17 entries are the original 11 combinations followed
            # by six pairwise duplicates.  Reconstruct those indices explicitly.
            expanded_combinations = ALL_COMBINATIONS + tuple(
                itertools.combinations(range(4), 2)
            )
            available_existing = [
                index for index, candidate in enumerate(expanded_combinations)
                if set(candidate).issubset(combo)
            ]
            learned_offset = len(expanded_combinations)
            available_learned = [
                learned_offset + index
                for index, candidate in enumerate(LEARNED_COMBINATIONS)
                if set(candidate).issubset(combo)
            ]
            existing = errors[:, available_existing].min(axis=1) * 1000.0
            expanded = errors[:, available_existing + available_learned].min(axis=1) * 1000.0
            stage_values["existing"].append(existing)
            stage_values["expanded"].append(expanded)
        existing = np.stack(stage_values["existing"], axis=1).reshape(-1)
        expanded = np.stack(stage_values["expanded"], axis=1).reshape(-1)
        # For the cache-level oracle, equal weighting over task combinations is
        # not the final protocol; the per-action values below are the primary
        # comparison and remain comparable with the E2 records.
        summary[f"V{count}"] = {
            "existing_frame_weighted_mm": float(existing.mean()),
            "expanded_frame_weighted_mm": float(expanded.mean()),
            "existing_action_equal_mm": action_equal(
                np.stack(stage_values["existing"], axis=1).mean(axis=1), actions
            ),
            "expanded_action_equal_mm": action_equal(
                np.stack(stage_values["expanded"], axis=1).mean(axis=1), actions
            ),
            "oracle_gain_mm": float(
                np.stack(stage_values["existing"], axis=1).mean()
                - np.stack(stage_values["expanded"], axis=1).mean()
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LearnableTriangulation(args.variant, args.attention_depth).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    output_dir = Path(args.output_dir).resolve()
    summaries = []
    for input_path in args.input_files:
        output_path = output_dir / Path(input_path).name
        summaries.append(export_one(input_path, output_path, model, device, args.batch_size))
        print(json.dumps(summaries[-1]), flush=True)
    result = {
        "method": "learned triangulation candidate extension",
        "checkpoint": args.checkpoint,
        "learned_combinations": [list(combo) for combo in LEARNED_COMBINATIONS],
        "summaries": summaries,
    }
    (output_dir / "export_summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
