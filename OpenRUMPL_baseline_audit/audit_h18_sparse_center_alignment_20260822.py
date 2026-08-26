#!/usr/bin/env python3
"""Evaluate a trained H18 pose residual on sparse single-frame test centers.

The H18 model is trained/evaluated on the dense stride-5 temporal H36M cache,
whereas the registered T=1 table uses the sparse validation PKL.  This audit
selects dense T=9 windows whose physical centre frame occurs in the sparse PKL
and evaluates both the frozen E2-C2 centre pose and the trained H18 prediction.
It does not train, select a checkpoint, or read test metrics for tuning.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from train_e2_clean_temporal_residual_20260818 import (
    TemporalPoseModel,
    build_windows,
    evaluate,
    metadata_from_pkl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-cache", required=True)
    parser.add_argument("--dense-fused", required=True)
    parser.add_argument("--dense-pkl", required=True)
    parser.add_argument("--sparse-pkl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-length", type=int, default=9)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def ordered_physical_keys(path: str) -> list[tuple[int, int, int, int]]:
    with open(path, "rb") as handle:
        records = pickle.load(handle)
    groups: collections.OrderedDict[tuple[int, int, int, int], None]
    groups = collections.OrderedDict()
    for record in records:
        key = (
            int(record["subject"]),
            int(record["action"]),
            int(record["subaction"]),
            int(record["image_id"]),
        )
        groups.setdefault(key, None)
    return list(groups)


def main() -> None:
    args = parse_args()
    dense_cache = np.load(args.dense_cache, allow_pickle=False)
    dense_fused = np.load(args.dense_fused, mmap_mode="r")
    dense_meta = metadata_from_pkl(args.dense_pkl, len(dense_cache["targets"]))
    dense_keys = ordered_physical_keys(args.dense_pkl)
    sparse_keys = ordered_physical_keys(args.sparse_pkl)
    if len(dense_keys) != len(dense_cache["targets"]):
        raise RuntimeError("dense PKL/cache group count mismatch")
    sparse_key_set = set(sparse_keys)
    missing = [key for key in sparse_keys if key not in set(dense_keys)]
    if missing:
        raise RuntimeError(f"sparse physical frames missing from dense PKL: {missing[:3]}")

    windows = build_windows(dense_meta, args.window_length, args.frame_stride)
    center = args.window_length // 2
    keep = np.asarray(
        [dense_keys[int(row[center])] in sparse_key_set for row in windows],
        dtype=bool,
    )
    aligned_windows = windows[keep]
    if not len(aligned_windows):
        raise RuntimeError("no sparse centres have a complete temporal window")

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = state["args"]
    model = TemporalPoseModel(
        int(saved_args["window_length"]),
        int(saved_args["hidden_dim"]),
        int(saved_args["layers"]),
        float(saved_args["residual_scale_m"]),
        camera_independent=bool(saved_args.get("camera_independent", False)),
        continuous_time=bool(saved_args.get("continuous_time", False)),
        reference_dt_s=float(saved_args.get("reference_dt_s", 0.1)),
        max_time_period_s=float(saved_args.get("max_time_period_s", 2.0)),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)

    baseline = evaluate(
        None, dense_cache, dense_fused, aligned_windows, device, center,
        args.batch_size,
    )
    temporal = evaluate(
        model, dense_cache, dense_fused, aligned_windows, device, center,
        args.batch_size, dense_meta["frame_ids"],
        float(saved_args.get("source_fps", 50.0)),
    )
    report = {
        "purpose": "H18 evaluation at sparse single-frame physical centres",
        "selection": "all sparse centres with a complete dense T=9 window",
        "checkpoint_selection": "none; uses the already frozen S8-selected checkpoint",
        "camera_independent": model.camera_independent,
        "continuous_time": model.continuous_time,
        "dense_frame_groups": len(dense_keys),
        "sparse_frame_groups": len(sparse_keys),
        "aligned_complete_windows": int(len(aligned_windows)),
        "sparse_centres_without_complete_window": int(
            len(sparse_keys) - len(aligned_windows)
        ),
        "baseline": baseline,
        "temporal": temporal,
        "args": vars(args),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
