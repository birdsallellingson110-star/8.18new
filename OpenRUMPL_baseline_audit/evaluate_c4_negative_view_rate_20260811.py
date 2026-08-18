#!/usr/bin/env python3
"""Compare H76 and C4 counterfactual monotonicity on S9/S11."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import predict_delta
from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    ArrayDataset,
    COMBINATIONS,
    JointUtilityScorer,
    TASK_COMBINATIONS,
    action_equal,
)
from train_h76_set_transformer_utility_20260811 import (
    SetTransformerJointUtility,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-type", choices=("c4", "set"), default="c4")
    return parser.parse_args()


def transition_summary(before, after, actions):
    before_error = np.linalg.norm(before["absolute"] - before["target"], axis=-1) * 1000
    after_error = np.linalg.norm(after["absolute"] - after["target"], axis=-1) * 1000
    delta = after_error - before_error
    before_relative = before["absolute"] - before["absolute"][:, :1]
    after_relative = after["absolute"] - after["absolute"][:, :1]
    target_relative = before["target"] - before["target"][:, :1]
    before_rel_error = np.linalg.norm(before_relative - target_relative, axis=-1) * 1000
    after_rel_error = np.linalg.norm(after_relative - target_relative, axis=-1) * 1000
    return {
        "before_mm": action_equal(before_error, actions),
        "after_mm": action_equal(after_error, actions),
        "mean_delta_mm": action_equal(delta, actions),
        "negative_view_rate": action_equal(delta > 0, actions),
        "negative_view_rate_gt1mm": action_equal(delta > 1, actions),
        "negative_view_rate_gt5mm": action_equal(delta > 5, actions),
        "pose_negative_rate": action_equal(delta.mean(axis=1) > 0, actions),
        "root_relative_before_mm": action_equal(before_rel_error, actions),
        "root_relative_after_mm": action_equal(after_rel_error, actions),
        "root_relative_nvr": action_equal(after_rel_error > before_rel_error, actions),
    }


def aggregate(transitions):
    keys = transitions[0].keys()
    return {key: float(np.mean([item[key] for item in transitions])) for key in keys}


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    cache_npz = np.load(args.validation_cache)
    arrays = {key: cache_npz[key] for key in cache_npz.files}
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.model_type == "set":
        model = SetTransformerJointUtility(
            checkpoint["mean"], checkpoint["std"],
            checkpoint["attention_depth"],
            checkpoint.get("view_cross_attention", False),
            checkpoint.get("joint_attention", "none"),
        )
        method_name = f"E-set-depth{checkpoint['attention_depth']}"
    else:
        model = JointUtilityScorer(checkpoint["mean"], checkpoint["std"])
        method_name = "C4"
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=2,
        pin_memory=True,
    )
    method_predictions = {combo: [] for combo in TASK_COMBINATIONS}
    with torch.inference_mode():
        for predictions, targets, rays, _ in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            for task_combo in TASK_COMBINATIONS:
                predicted, _, _, candidates, _ = predict_delta(
                    model, predictions, targets, rays, task_combo
                )
                weights = F.softmax(-predicted, dim=-1)
                fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                method_predictions[task_combo].append(fused.cpu().numpy())
    method_predictions = {
        combo: np.concatenate(values, axis=0)
        for combo, values in method_predictions.items()
    }

    targets = arrays["targets"]
    actions = arrays["actions"]
    h76_predictions = {
        combo: arrays["predictions"][:, COMBINATIONS.index(combo)]
        for combo in COMBINATIONS
    }
    transitions = {"H76": {"V2_to_V3": [], "V3_to_V4": []},
                   method_name: {"V2_to_V3": [], "V3_to_V4": []}}
    metadata = []
    for base_views in (2, 3):
        for base_combo in itertools.combinations(range(4), base_views):
            for added_view in sorted(set(range(4)) - set(base_combo)):
                larger_combo = tuple(sorted((*base_combo, added_view)))
                stage = "V2_to_V3" if base_views == 2 else "V3_to_V4"
                metadata.append({
                    "stage": stage, "base_combo": list(base_combo),
                    "larger_combo": list(larger_combo), "added_view": added_view,
                })
                for method in ("H76", method_name):
                    before_prediction = (
                        h76_predictions[base_combo]
                        if base_views == 2 or method == "H76"
                        else method_predictions[base_combo]
                    )
                    after_prediction = (
                        h76_predictions[larger_combo]
                        if method == "H76"
                        else method_predictions[larger_combo]
                    )
                    summary = transition_summary(
                        {"absolute": before_prediction, "target": targets},
                        {"absolute": after_prediction, "target": targets},
                        actions,
                    )
                    transitions[method][stage].append(summary)

    result = {
        "metric": "action-equal absolute All-17 unless root_relative prefix",
        "groups": len(targets),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "aggregate": {
            method: {
                stage: aggregate(values) for stage, values in stages.items()
            }
            for method, stages in transitions.items()
        },
        "per_transition": transitions,
        "transition_metadata": metadata,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# H76 vs C4 negative-view audit", "",
        "| Method | Stage | Before | After | NVR | >1mm | >5mm | Pose NVR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("H76", method_name):
        for stage in ("V2_to_V3", "V3_to_V4"):
            value = result["aggregate"][method][stage]
            lines.append(
                f"| {method} | {stage} | {value['before_mm']:.3f} | "
                f"{value['after_mm']:.3f} | {100*value['negative_view_rate']:.2f}% | "
                f"{100*value['negative_view_rate_gt1mm']:.2f}% | "
                f"{100*value['negative_view_rate_gt5mm']:.2f}% | "
                f"{100*value['pose_negative_rate']:.2f}% |"
            )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
