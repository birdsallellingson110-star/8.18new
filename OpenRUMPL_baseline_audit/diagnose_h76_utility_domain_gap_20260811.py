#!/usr/bin/env python3
"""Explain why H76 hypothesis utility training does or does not transfer."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_hypothesis_utility_20260811 import (
    ArrayDataset,
    COMBINATIONS,
    JointUtilityScorer,
    PoseHypothesisScorer,
    TASK_COMBINATIONS,
    load_arrays,
    task_spec,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--pose-checkpoint", required=True)
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def load_scorer(path: str, variant: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu")
    if variant == "pose":
        model = PoseHypothesisScorer(checkpoint["mean"], checkpoint["std"])
    else:
        model = JointUtilityScorer(checkpoint["mean"], checkpoint["std"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), float(checkpoint["temperature"])


def audit_split(arrays, indices, models, device, batch_size):
    loader = DataLoader(
        ArrayDataset(arrays, indices), batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    stores = {
        variant: defaultdict(lambda: defaultdict(float)) for variant in models
    }
    common = defaultdict(lambda: defaultdict(float))
    with torch.inference_mode():
        for predictions, targets, rays, _ in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            for task_combo in TASK_COMBINATIONS:
                stage = "V3" if len(task_combo) == 3 else "V4"
                available, candidate_masks, task_mask = task_spec(task_combo, device)
                candidates = predictions[:, available]
                baseline_global = COMBINATIONS.index(task_combo)
                baseline_local = available.index(baseline_global)
                error = torch.linalg.vector_norm(
                    candidates - targets[:, None], dim=-1
                )
                batch, count, joints = error.shape
                common[stage]["joint_count"] += batch * joints
                common[stage]["pose_count"] += batch
                common[stage]["joint_baseline_best"] += (
                    error.argmin(dim=1) == baseline_local
                ).sum().item()
                common[stage]["pose_baseline_best"] += (
                    error.mean(dim=-1).argmin(dim=1) == baseline_local
                ).sum().item()
                common[stage]["baseline_error"] += error[:, baseline_local].sum().item()
                common[stage]["joint_oracle_error"] += error.min(dim=1).values.sum().item()

                for variant, (model, temperature) in models.items():
                    logits = model(candidates, rays, candidate_masks, task_mask)
                    if logits.ndim == 2:
                        weights = F.softmax(logits / temperature, dim=1)
                        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=1)
                        store_count = batch
                        top1 = weights.argmax(dim=1)
                        baseline_weight = weights[:, baseline_local]
                    else:
                        weights = F.softmax(logits / temperature, dim=-1)
                        entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=-1)
                        store_count = batch * joints
                        top1 = weights.argmax(dim=-1)
                        baseline_weight = weights[..., baseline_local]
                    stores[variant][stage]["count"] += store_count
                    stores[variant][stage]["baseline_weight"] += baseline_weight.sum().item()
                    stores[variant][stage]["baseline_top1"] += (
                        top1 == baseline_local
                    ).sum().item()
                    stores[variant][stage]["entropy"] += entropy.sum().item()
                    stores[variant][stage]["max_entropy"] += store_count * np.log(count)

    result = {"counterfactual_labels": {}, "learned_weights": {}}
    for stage, values in common.items():
        result["counterfactual_labels"][stage] = {
            "baseline_is_joint_oracle_rate": (
                values["joint_baseline_best"] / values["joint_count"]
            ),
            "baseline_is_pose_oracle_rate": (
                values["pose_baseline_best"] / values["pose_count"]
            ),
            "baseline_frame_weighted_mm": (
                1000.0 * values["baseline_error"] / values["joint_count"]
            ),
            "joint_oracle_frame_weighted_mm": (
                1000.0 * values["joint_oracle_error"] / values["joint_count"]
            ),
        }
    for variant, stages in stores.items():
        result["learned_weights"][variant] = {}
        for stage, values in stages.items():
            result["learned_weights"][variant][stage] = {
                "baseline_mean_weight": values["baseline_weight"] / values["count"],
                "baseline_top1_rate": values["baseline_top1"] / values["count"],
                "normalized_entropy": values["entropy"] / values["max_entropy"],
            }
    return result


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    models = {
        "pose": load_scorer(args.pose_checkpoint, "pose", device),
        "joint": load_scorer(args.joint_checkpoint, "joint", device),
    }
    holdout_indices = np.flatnonzero(train["group_indices"] % 10 == 0)
    result = {
        "train_subject_internal_holdout": audit_split(
            train, holdout_indices, models, device, args.batch_size
        ),
        "S9_S11": audit_split(
            validation, np.arange(len(validation["targets"])),
            models, device, args.batch_size,
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
