#!/usr/bin/env python3
"""Zero-shot E2 evaluation after appending learned-triangulation candidates."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    ArrayDataset,
    JOINT_NAMES,
    TASK_COMBINATIONS,
)
from train_h76_pairwise_set_transformer_20260812 import (
    EXPANDED_COMBINATIONS as BASE_COMBINATIONS,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


LEARNED_COMBINATIONS = tuple(
    combo for combo in itertools.chain(
        itertools.combinations(range(4), 2),
        itertools.combinations(range(4), 3),
        itertools.combinations(range(4), 4),
    )
    if len(combo) >= 3
)
ALL_COMBINATIONS = BASE_COMBINATIONS + LEARNED_COMBINATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def task_spec(task_combo, device):
    available = [
        index for index, combo in enumerate(ALL_COMBINATIONS)
        if set(combo).issubset(task_combo)
    ]
    masks = torch.zeros(len(available), 4, device=device)
    for row, index in enumerate(available):
        masks[row, list(ALL_COMBINATIONS[index])] = 1.0
    task_mask = torch.zeros(4, device=device)
    task_mask[list(task_combo)] = 1.0
    return available, masks, task_mask


def evaluate(model, loader, device, temperature):
    model.eval()
    stores = {
        stage: {mode: [] for mode in ("h76", "soft", "hard", "oracle")}
        for stage in ("V3", "V4")
    }
    actions_by_stage = {"V3": [], "V4": []}
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            for task_combo in TASK_COMBINATIONS:
                stage = f"V{len(task_combo)}"
                available, masks, task_mask = task_spec(task_combo, device)
                candidates = predictions[:, available]
                scores = model(candidates, rays, masks, task_mask)
                baseline_local = available.index(BASE_COMBINATIONS.index(task_combo))
                errors = torch.linalg.vector_norm(
                    candidates - targets[:, None], dim=-1
                )
                weights = F.softmax(-scores / temperature, dim=-1)
                soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                hard_index = scores.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                for name, pose in (
                    ("h76", candidates[:, baseline_local]),
                    ("soft", soft),
                    ("hard", hard),
                ):
                    stores[stage][name].append(
                        torch.linalg.vector_norm(pose - targets, dim=-1).cpu().numpy()
                        * 1000.0
                    )
                stores[stage]["oracle"].append(errors.min(dim=1).values.cpu().numpy() * 1000.0)
                actions_by_stage[stage].append(actions.numpy().copy())
    result = {}
    for stage in stores:
        actions = np.concatenate(actions_by_stage[stage])
        result[stage] = {}
        for mode, chunks in stores[stage].items():
            values = np.concatenate(chunks)
            result[stage][mode] = {
                "action_equal_all17_mm": action_equal(values, actions),
                "frame_weighted_all17_mm": float(values.mean()),
                "per_joint_mm": {
                    joint: action_equal(values[:, index], actions)
                    for index, joint in enumerate(JOINT_NAMES)
                },
            }
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    cache = np.load(args.cache)
    arrays = {key: cache[key] for key in cache.files}
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SetTransformerJointUtility(
        checkpoint["mean"].to(device), checkpoint["std"].to(device),
        checkpoint.get("attention_depth", 2),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = DataLoader(
        ArrayDataset(arrays, np.arange(len(arrays["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    result = {
        "method": "zero-shot E2 scoring of learned triangulation candidates",
        "base_candidate_count": len(BASE_COMBINATIONS),
        "expanded_candidate_count": len(ALL_COMBINATIONS),
        "temperature": args.temperature,
        "checkpoint": args.checkpoint,
        "test": evaluate(model, loader, device, args.temperature),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
