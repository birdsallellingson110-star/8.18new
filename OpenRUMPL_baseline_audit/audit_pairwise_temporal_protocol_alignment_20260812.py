#!/usr/bin/env python3
"""Align dense T=9 windows to the sparse single-frame H36M protocol.

The temporal cache contains every complete 5-frame group, while the strict
single-frame validation cache contains every 13th available group.  This
script maps the two pkl files by their physical H36M key and evaluates the
already-trained temporal checkpoints only at sparse centers that have a full
T=9 context.  No checkpoint or test metric is selected here.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from train_temporal_candidate_utility_20260812 import (
    CandidateFrameArrays,
    TemporalCandidateUtility,
    configure_candidate_pool,
    evaluate,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--temporal-cache", required=True)
    p.add_argument("--temporal-index", required=True)
    p.add_argument("--temporal-pkl", required=True)
    p.add_argument("--sparse-pkl", required=True)
    p.add_argument("--checkpoint", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--gpu", default="0")
    p.add_argument("--task-stage", choices=("all", "v3", "v4"), default="all")
    return p.parse_args()


def pkl_keys(path: str):
    db = pickle.load(open(path, "rb"))
    groups = collections.OrderedDict()
    for record in db:
        key = (
            int(record["subject"]), int(record["action"]),
            int(record["subaction"]), int(record["image_id"]),
        )
        groups.setdefault(key, None)
    return list(groups)


def make_aligned_index(temporal_pkl, sparse_pkl, temporal_index, output):
    temporal_keys = pkl_keys(temporal_pkl)
    sparse_keys = pkl_keys(sparse_pkl)
    temporal_lookup = {key: i for i, key in enumerate(temporal_keys)}
    missing = [key for key in sparse_keys if key not in temporal_lookup]
    if missing:
        raise RuntimeError(f"sparse keys missing from temporal pkl: {missing[:3]}")
    sparse_centers = np.asarray(
        [temporal_lookup[key] for key in sparse_keys], dtype=np.int64
    )
    index = np.load(temporal_index)
    centers = np.asarray(index["center_group_indices"], dtype=np.int64)
    selected = np.flatnonzero(np.isin(centers, sparse_centers))
    selected_centers = centers[selected]
    payload = {
        key: np.asarray(index[key])[selected]
        for key in (
            "window_indices", "window_frame_ids", "subjects", "actions",
            "sequence_ids", "center_group_indices",
        )
        if key in index.files
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    report = {
        "temporal_frame_groups": len(temporal_keys),
        "sparse_frame_groups": len(sparse_keys),
        "sparse_centers_with_full_T9_context": int(len(selected)),
        "sparse_centers_without_full_T9_context": int(
            len(sparse_keys) - len(selected)
        ),
        "selected_center_group_indices_first": selected_centers[:10].tolist(),
        "selected_center_group_indices_last": selected_centers[-10:].tolist(),
        "aligned_index": str(output),
    }
    return report


def evaluate_checkpoints(args, aligned_index, output):
    configure_candidate_pool(True, args.task_stage)
    device = torch.device(f"cuda:{args.gpu}")
    arrays = CandidateFrameArrays(
        args.temporal_cache, str(aligned_index), max_windows=0,
        aux_file="candidate_confidence.npy",
    )
    active_count = 5 if args.task_stage == "all" else (4 if args.task_stage == "v3" else 1)
    indices = np.arange(arrays.window_count * active_count, dtype=np.int64)
    results = {}
    for checkpoint_path in args.checkpoint:
        state = torch.load(checkpoint_path, map_location="cpu")
        model = TemporalCandidateUtility(
            arrays.window_length,
            int(state["hidden_dim"]), int(state["layers"]), int(state["heads"]),
        ).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        result = evaluate(model, arrays, indices, device, args.batch_size)
        results[str(checkpoint_path)] = result
        del model
        torch.cuda.empty_cache()
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    aligned = out / "aligned_validation_temporal_index.npz"
    alignment = make_aligned_index(
        args.temporal_pkl, args.sparse_pkl,
        args.temporal_index, aligned,
    )
    results = evaluate_checkpoints(
        args, aligned, out / "aligned_checkpoint_results.json"
    )
    report = {"args": vars(args), "alignment": alignment, "results": results}
    (out / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
