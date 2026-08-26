#!/usr/bin/env python3
"""Build frame-level E2-C2 soft-fusion poses for temporal training.

The E2 candidate generator and the calibrated C2 score model remain frozen.
This script only materializes the *same* per-frame soft pose that is used by
the H16 screen, so the temporal experiment can train a direct pose residual
without repeatedly loading the 22-candidate tensors on every iteration.

No target is read by the fusion operation.  Targets/actions/subjects are
copied only as metadata for the downstream, subject-level holdout protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_e2_v234_universal_20260812 import ORIGINAL_COMBINATIONS


TASKS = ORIGINAL_COMBINATIONS
ALL_CANDIDATES = ORIGINAL_COMBINATIONS + ORIGINAL_COMBINATIONS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--temperature-v2", type=float, default=0.4)
    p.add_argument("--temperature-v3", type=float, default=1.8)
    p.add_argument("--temperature-v4", type=float, default=1.8)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def task_spec(task_index: int):
    task = TASKS[task_index]
    available = [
        i for i, combo in enumerate(ALL_CANDIDATES)
        if set(combo).issubset(task)
    ]
    baseline_local = available.index(task_index)
    return task, np.asarray(available, dtype=np.int64), baseline_local


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    source = np.load(args.cache, allow_pickle=False)
    required = {"predictions", "rays", "actions", "subjects", "group_indices"}
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"{args.cache} missing {sorted(missing)}")
    predictions = source["predictions"]
    scores = np.load(args.scores, mmap_mode="r")
    expected = (len(predictions), len(TASKS), 17, len(ALL_CANDIDATES))
    if tuple(scores.shape) != expected:
        raise ValueError(f"score shape {scores.shape} != expected {expected}")
    if tuple(predictions.shape[1:]) != (22, 17, 3):
        raise ValueError(f"candidate shape {predictions.shape} is not (N,22,17,3)")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fused = np.lib.format.open_memmap(
        out / "fused_poses.npy", mode="w+", dtype=np.float32,
        shape=(len(predictions), len(TASKS), 17, 3),
    )
    # Keep the source ordering and metadata explicit.  This avoids a silent
    # mismatch between temporal windows and the corresponding dense cache.
    for name in ("targets", "actions", "subjects", "group_indices"):
        np.save(out / f"{name}.npy", np.asarray(source[name]))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    temperatures = {2: args.temperature_v2, 3: args.temperature_v3,
                    4: args.temperature_v4}
    with torch.inference_mode():
        for start in range(0, len(predictions), args.chunk_size):
            stop = min(start + args.chunk_size, len(predictions))
            pred = torch.from_numpy(np.asarray(predictions[start:stop])).to(device)
            score = torch.from_numpy(np.asarray(scores[start:stop])).to(device)
            for task_index, task in enumerate(TASKS):
                _, available, baseline_local = task_spec(task_index)
                candidate = pred[:, available]  # B,C,J,3
                task_score = score[:, task_index, :, available]  # B,J,C
                delta = task_score - task_score[:, :, baseline_local:baseline_local + 1]
                weights = torch.softmax(-delta / temperatures[len(task)], dim=-1)
                fused[start:stop, task_index] = torch.einsum(
                    "bjc,bcjd->bjd", weights, candidate
                ).cpu().numpy().astype(np.float32)
            if start == 0 or (start // args.chunk_size) % 20 == 0:
                print(f"fused {stop}/{len(predictions)}", flush=True)
    fused.flush()
    manifest = {
        "method": "frozen calibrated E2-C2 soft-fusion frame cache",
        "input_cache": str(Path(args.cache).resolve()),
        "score_cache": str(Path(args.scores).resolve()),
        "tasks": [list(x) for x in TASKS],
        "candidate_count": len(ALL_CANDIDATES),
        "temperatures": {f"V{k}": v for k, v in temperatures.items()},
        "fused_shape": [len(predictions), len(TASKS), 17, 3],
        "dtype": "float32",
        "label_free_fusion": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
