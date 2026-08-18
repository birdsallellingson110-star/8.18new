#!/usr/bin/env python3
"""Materialize indexed T=9 windows into shared read-only memmaps.

The scientific protocol is unchanged.  This is only an I/O optimization for
the temporal experiments: workers no longer perform a random fancy-index into
the 1.5 GB decompressed frame cache for every sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame-cache", required=True)
    p.add_argument("--temporal-index", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--chunk-windows", type=int, default=512)
    args = p.parse_args()
    task_ids = np.asarray(
        sorted(set(int(x) for x in args.task_ids.split(",") if x.strip())),
        dtype=np.int64,
    )
    if len(task_ids) == 0 or task_ids.min() < 0 or task_ids.max() >= 11:
        raise ValueError("task ids must be in 0..10")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = np.load(args.frame_cache)
    index = np.load(args.temporal_index)
    predictions = cache["predictions"]
    targets = cache["targets"]
    windows = index["window_indices"].astype(np.int64)
    n_windows, length = windows.shape
    joints = predictions.shape[-2]
    poses_path = out / "window_poses.npy"
    target_path = out / "window_targets.npy"
    poses = np.lib.format.open_memmap(
        poses_path, mode="w+", dtype=np.float32,
        shape=(len(task_ids), n_windows, length, joints, 3),
    )
    target_memmap = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32,
        shape=(n_windows, joints, 3),
    )
    center = length // 2
    for start in range(0, n_windows, args.chunk_windows):
        stop = min(start + args.chunk_windows, n_windows)
        frame_indices = windows[start:stop]
        # B,T,C,J,3 -> C,B,T,J,3, with C restricted to the requested tasks.
        chunk = predictions[frame_indices][:, :, task_ids]
        poses[:, start:stop] = chunk.transpose(2, 0, 1, 3, 4)
        target_memmap[start:stop] = targets[frame_indices[:, center]]
        if start == 0 or stop == n_windows or start % (args.chunk_windows * 20) == 0:
            print(f"materialized {stop}/{n_windows}", flush=True)
    poses.flush()
    target_memmap.flush()
    manifest = {
        "frame_cache": args.frame_cache,
        "temporal_index": args.temporal_index,
        "task_ids": task_ids.tolist(),
        "window_count": int(n_windows),
        "window_length": int(length),
        "pose_shape": list(poses.shape),
        "target_shape": list(target_memmap.shape),
        "poses": str(poses_path),
        "targets": str(target_path),
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
