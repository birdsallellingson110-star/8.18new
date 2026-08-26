#!/usr/bin/env python3
"""Materialize the exact frozen limb-proposal K96 pose for every dense frame.

Sampling order, seed, task order, and batch size match the formal K96
evaluation.  The resulting N x 11 x 17 x 3 memmap can replace the obsolete
22-candidate E2-C2 temporal anchor without regenerating 2D detections/rays.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from train_e2_v234_universal_20260812 import ORIGINAL_COMBINATIONS
from train_failure_informed_map_20260820 import FrozenK96Anchor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--e2-checkpoint", required=True)
    p.add_argument("--proposal-checkpoint", required=True)
    p.add_argument("--k96-checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    action_ids = tuple(range(2, 17))
    return float(np.mean([
        values[actions == action].mean()
        for action in action_ids if np.any(actions == action)
    ]))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(10000 + args.seed)
    torch.cuda.manual_seed_all(10000 + args.seed)
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    device = torch.device(f"cuda:{args.gpu}")
    arrays = trainer.load_arrays([args.cache], 22)
    total = len(arrays["targets"])
    if args.max_samples:
        total = min(total, args.max_samples)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    poses = np.lib.format.open_memmap(
        out / "fused_poses.npy", mode="w+", dtype=np.float32,
        shape=(total, len(ORIGINAL_COMBINATIONS), 17, 3),
    )
    anchor = FrozenK96Anchor(args, device)
    errors = {f"V{k}": [] for k in (2, 3, 4)}
    actions = np.asarray(arrays["actions"][:total])
    with torch.inference_mode():
        for start in range(0, total, args.batch_size):
            stop = min(start + args.batch_size, total)
            predictions = torch.from_numpy(
                np.asarray(arrays["predictions"][start:stop])
            ).to(device)
            rays = torch.from_numpy(
                np.asarray(arrays["rays"][start:stop])
            ).to(device)
            targets = torch.from_numpy(
                np.asarray(arrays["targets"][start:stop])
            ).to(device)
            for task_index, combo in enumerate(ORIGINAL_COMBINATIONS):
                fused = anchor(predictions, rays, combo)
                poses[start:stop, task_index] = fused.cpu().numpy()
                error = torch.linalg.vector_norm(
                    fused - targets, dim=-1
                ).cpu().numpy() * 1000.0
                errors[f"V{len(combo)}"].append(error)
            if start == 0 or (start // args.batch_size) % 25 == 0:
                print(f"K96 {stop}/{total}", flush=True)
    poses.flush()
    metrics = {}
    for stage, chunks in errors.items():
        value = np.concatenate(chunks, axis=0)
        # chunks are task-major within each batch.  Repeat action labels for
        # the number of camera combinations represented by this stage.
        repeat = {"V2": 6, "V3": 4, "V4": 1}[stage]
        stage_actions = np.concatenate([
            np.repeat(actions[s:s + args.batch_size], repeat)
            for s in range(0, total, args.batch_size)
        ])
        # Reorder errors from [batch, task] chunks to the same flattened order.
        grouped = []
        cursor = 0
        task_chunks = []
        for s in range(0, total, args.batch_size):
            n = min(args.batch_size, total - s)
            part = chunks[cursor:cursor + repeat]
            cursor += repeat
            task_chunks.append(np.stack(part, axis=1).reshape(n * repeat, 17))
        value = np.concatenate(task_chunks, axis=0)
        metrics[stage] = {
            "action_equal_all17_mm": action_equal(value, stage_actions),
            "frame_weighted_all17_mm": float(value.mean()),
        }
    manifest = {
        "method": "exact frozen limb-utility proposal K96 temporal anchor",
        "input_cache": str(Path(args.cache).resolve()),
        "k96_checkpoint": str(Path(args.k96_checkpoint).resolve()),
        "seed": args.seed,
        "sampling_seed": 10000 + args.seed,
        "batch_size": args.batch_size,
        "shape": list(poses.shape),
        "tasks": [list(x) for x in ORIGINAL_COMBINATIONS],
        "metrics": metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "COMPLETED").write_text("completed\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
