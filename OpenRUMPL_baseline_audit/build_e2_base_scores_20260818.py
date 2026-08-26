#!/usr/bin/env python3
"""Pre-compute frozen E2-C2 utility scores for temporal screening.

The candidate generator and the E2 scorer remain frozen.  This utility cache
contains only the scores emitted by the already trained C2 Set-Transformer;
it does not use 3-D labels.  Keeping the scores on disk makes the temporal
screen cheap and makes the identity-at-T=1 control exact (up to float32
serialization).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from train_e2_v234_universal_20260812 import ORIGINAL_COMBINATIONS
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


TASKS = ORIGINAL_COMBINATIONS
ALL_CANDIDATES = ORIGINAL_COMBINATIONS + ORIGINAL_COMBINATIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True, help="Output .npy memmap")
    p.add_argument("--gpu", default="0")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def task_spec(task: tuple[int, ...], device: torch.device):
    available = [
        i for i, combo in enumerate(ALL_CANDIDATES)
        if set(combo).issubset(task)
    ]
    masks = torch.zeros(len(available), 4, device=device)
    for row, index in enumerate(available):
        masks[row, list(ALL_CANDIDATES[index])] = 1.0
    task_mask = torch.zeros(4, device=device)
    task_mask[list(task)] = 1.0
    return available, masks, task_mask


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    source = np.load(args.cache, allow_pickle=False)
    required = {"predictions", "rays", "group_indices", "subjects", "actions"}
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"cache missing fields: {sorted(missing)}")
    predictions = source["predictions"]
    rays = source["rays"]
    if predictions.ndim != 4 or predictions.shape[1:] != (22, 17, 3):
        raise ValueError(f"expected (N,22,17,3), got {predictions.shape}")
    if rays.shape[1:] != (17, 4, 7):
        raise ValueError(f"expected rays (N,17,4,7), got {rays.shape}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    mean = checkpoint["mean"].to(device)
    std = checkpoint["std"].to(device)
    model = SetTransformerJointUtility(
        mean, std, checkpoint.get("attention_depth", 2),
        stage_heads=checkpoint.get("stage_heads", False),
        canonical_geometry=checkpoint.get("canonical_geometry", False),
        fixed_metric_normalization=checkpoint.get(
            "fixed_metric_normalization", False
        ),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    # Scores are padded to the fixed 11-task x 22-candidate layout.  Entries
    # outside a task are ignored by the temporal trainer and remain zero.
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(len(predictions), len(TASKS), 17, len(ALL_CANDIDATES)),
    )
    scores[:] = 0.0
    with torch.inference_mode():
        for start in range(0, len(predictions), args.batch_size):
            stop = min(start + args.batch_size, len(predictions))
            pred = torch.from_numpy(predictions[start:stop]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            ray = torch.from_numpy(rays[start:stop]).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            for task_index, task in enumerate(TASKS):
                available, masks, task_mask = task_spec(task, device)
                task_scores = model(pred[:, available], ray, masks, task_mask)
                score_array = scores[start:stop, task_index]
                score_array[:, :, available] = (
                    task_scores.detach().cpu().numpy().astype(np.float32)
                )
                scores[start:stop, task_index] = score_array
            if start == 0 or (start // args.batch_size) % 20 == 0:
                print(f"scores {stop}/{len(predictions)}", flush=True)
    scores.flush()
    meta = {
        "method": "frozen E2-C2 utility score cache for temporal residual screen",
        "input_cache": str(Path(args.cache).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "shape": list(scores.shape),
        "tasks": [list(x) for x in TASKS],
        "candidate_order": "H76 11 + confidence-weighted 11",
        "dtype": "float32",
        "label_free_generation": True,
    }
    output.with_suffix(".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
