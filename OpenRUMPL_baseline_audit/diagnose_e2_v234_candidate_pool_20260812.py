#!/usr/bin/env python3
"""Measure zero-training complementarity of the V234 candidate cache."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from train_h76_hypothesis_utility_20260811 import ACTION_NAMES


ORIGINAL = tuple(
    combo
    for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)
PAIRWISE = tuple(itertools.combinations(range(4), 2))
LEARNED = tuple(combo for combo in ORIGINAL if len(combo) >= 3)
COMBINATIONS = ORIGINAL + PAIRWISE + LEARNED
ALL = COMBINATIONS + ORIGINAL + ORIGINAL
VARIANTS = {
    "existing_22": range(0, 22),
    "existing_plus_confidence": list(range(0, 22)) + list(range(22, 33)),
    "existing_plus_irls": list(range(0, 22)) + list(range(33, 44)),
    "existing_plus_confidence_irls": range(0, 44),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def action_equal(values, actions):
    values = np.asarray(values)
    actions = np.asarray(actions)
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def main():
    args = parse_args()
    with np.load(args.validation_cache, allow_pickle=False) as source:
        predictions = source["predictions"].astype(np.float32)
        targets = source["targets"].astype(np.float32)
        actions = source["actions"].astype(np.int16)
    if predictions.shape[1:] != (44, 17, 3):
        raise ValueError(f"expected [N,44,17,3], got {predictions.shape}")
    errors = np.linalg.norm(predictions - targets[:, None], axis=-1) * 1000.0
    per_task = {}
    aggregate = {stage: {name: [] for name in VARIANTS} for stage in ("V2", "V3", "V4")}
    aggregate_direct = {stage: {name: [] for name in VARIANTS} for stage in ("V2", "V3", "V4")}
    for task in ORIGINAL:
        stage = f"V{len(task)}"
        available = [i for i, combo in enumerate(ALL) if set(combo).issubset(task)]
        task_name = "".join(str(i) for i in task)
        row = {"task": list(task), "available_count": len(available)}
        for name, indices in VARIANTS.items():
            indices = [i for i in indices if i in available]
            pool_errors = errors[:, indices]
            oracle = pool_errors.min(axis=1)
            row[name] = {
                "candidate_count": len(indices),
                "oracle_action_equal_all17_mm": action_equal(oracle, actions),
                "oracle_frame_weighted_all17_mm": float(oracle.mean()),
                "direct_best_action_equal_all17_mm": action_equal(
                    pool_errors.min(axis=2).min(axis=1), actions
                ),
            }
            aggregate[stage][name].append(oracle)
            aggregate_direct[stage][name].append(pool_errors.min(axis=2).min(axis=1))
        per_task[task_name] = row
    result = {"per_task": per_task, "aggregate": {}}
    for stage in ("V2", "V3", "V4"):
        result["aggregate"][stage] = {}
        repeated_actions = np.tile(actions, len(aggregate[stage]["existing_22"]))
        for name in VARIANTS:
            oracle = np.concatenate(aggregate[stage][name])
            direct = np.concatenate(aggregate_direct[stage][name])
            result["aggregate"][stage][name] = {
                "oracle_action_equal_all17_mm": action_equal(oracle, repeated_actions),
                "oracle_frame_weighted_all17_mm": float(oracle.mean()),
                "direct_best_action_equal_all17_mm": action_equal(direct, repeated_actions),
            }
    result["config"] = {
        "validation_cache": str(Path(args.validation_cache).resolve()),
        "candidate_count": 44,
        "candidate_order": [list(item) for item in ALL],
        "note": "zero-training oracle only; no checkpoint or hyperparameter selection",
    }
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# E2 V234 candidate-pool oracle diagnostic (2026-08-12)", "",
        "No model is trained and no checkpoint is selected.  Lower oracle means the candidate pool contains useful complementary hypotheses.", "",
        "| stage | existing 22 | +confidence | +IRLS | +both |", "|---|---:|---:|---:|---:|",
    ]
    for stage in ("V2", "V3", "V4"):
        vals = result["aggregate"][stage]
        lines.append(
            f"| {stage} | {vals['existing_22']['oracle_action_equal_all17_mm']:.3f} "
            f"| {vals['existing_plus_confidence']['oracle_action_equal_all17_mm']:.3f} "
            f"| {vals['existing_plus_irls']['oracle_action_equal_all17_mm']:.3f} "
            f"| {vals['existing_plus_confidence_irls']['oracle_action_equal_all17_mm']:.3f} |"
        )
    (out / "result.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
