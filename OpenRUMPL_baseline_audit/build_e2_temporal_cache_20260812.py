#!/usr/bin/env python3
"""Build protocol-safe temporal caches from the frozen H76 + E-2 pipeline.

The temporal model must not see the sparse ``annot_filtered_5_64`` validation
split.  This script therefore exports frame-level E-2 outputs first and then
builds T=9 windows directly from the official temporal pkl metadata.  The
window index is kept separate from the pose cache so overlapping windows do
not duplicate several hundred MB of floating-point data.
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT))
from train_h76_hypothesis_utility_20260811 import (  # noqa: E402
    COMBINATIONS,
    TASK_COMBINATIONS,
    load_arrays,
    task_spec,
)
from train_h76_set_transformer_utility_20260811 import (  # noqa: E402
    SetTransformerJointUtility,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-shards", nargs="+", required=True)
    p.add_argument("--validation-shards", nargs="+", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window-length", type=int, default=9)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def _metadata_from_pkl(path: str, expected_groups: int):
    """Return the exact grouping order used by MultiViewH36M_RUMPL."""

    with open(path, "rb") as handle:
        db = pickle.load(handle)
    # MultiViewH36M_RUMPL.get_group inserts keys in first-seen order while
    # scanning db, and its key is precisely this four-field tuple.
    groups = collections.OrderedDict()
    for record in db:
        key = (
            int(record["subject"]),
            int(record["action"]),
            int(record["subaction"]),
            int(record["image_id"]),
        )
        entry = groups.setdefault(
            key,
            {"camera_ids": set(), "subject": key[0], "action": key[1],
             "subaction": key[2], "frame_id": key[3]},
        )
        entry["camera_ids"].add(int(record["camera_id"]))
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"{path}: pkl groups={len(groups)} but exported cache has "
            f"{expected_groups}; refusing to guess an alignment"
        )
    keys = list(groups)
    bad = [
        (key, sorted(groups[key]["camera_ids"]))
        for key in keys
        if groups[key]["camera_ids"] != {0, 1, 2, 3}
    ]
    if bad:
        raise RuntimeError(f"incomplete four-view groups in {path}: {bad[:3]}")

    sequence_numbers = {}
    sequence_ids = []
    for subject, action, subaction, _ in keys:
        seq = (subject, action, subaction)
        if seq not in sequence_numbers:
            sequence_numbers[seq] = len(sequence_numbers)
        sequence_ids.append(sequence_numbers[seq])
    return {
        "group_keys": keys,
        "subjects": np.asarray([key[0] for key in keys], dtype=np.int16),
        "actions": np.asarray([key[1] for key in keys], dtype=np.int16),
        "frame_ids": np.asarray([key[3] for key in keys], dtype=np.int64),
        "sequence_ids": np.asarray(sequence_ids, dtype=np.int32),
        "sequence_count": len(sequence_numbers),
    }


def _build_windows(metadata, length: int, stride: int):
    by_sequence = collections.defaultdict(list)
    for index, sequence_id in enumerate(metadata["sequence_ids"].tolist()):
        by_sequence[int(sequence_id)].append(index)
    windows = []
    for sequence_id, indices in by_sequence.items():
        indices.sort(key=lambda i: int(metadata["frame_ids"][i]))
        for start in range(0, len(indices) - length + 1):
            candidate = indices[start : start + length]
            frame_ids = metadata["frame_ids"][candidate]
            if not np.all(np.diff(frame_ids) == stride):
                continue
            windows.append(candidate)
    if not windows:
        raise RuntimeError(f"no T={length} windows with frame stride {stride}")
    windows = np.asarray(windows, dtype=np.int64)
    center = length // 2
    return {
        "window_indices": windows,
        "window_frame_ids": metadata["frame_ids"][windows],
        "subjects": metadata["subjects"][windows[:, center]],
        "actions": metadata["actions"][windows[:, center]],
        "sequence_ids": metadata["sequence_ids"][windows[:, center]],
        "center_group_indices": windows[:, center],
    }


def _load_e2(checkpoint_path: str, device: torch.device):
    state = torch.load(checkpoint_path, map_location="cpu")
    model = SetTransformerJointUtility(
        state["mean"], state["std"], int(state["attention_depth"]),
        bool(state.get("view_cross_attention", False)),
        state.get("joint_attention", "none"),
    )
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval()


def _fuse_e2(arrays, model, device, batch_size: int):
    predictions = arrays["predictions"]
    rays = arrays["rays"]
    output = predictions.copy()
    for start in range(0, len(predictions), batch_size):
        stop = min(start + batch_size, len(predictions))
        prediction = torch.from_numpy(predictions[start:stop]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        ray = torch.from_numpy(rays[start:stop]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        with torch.inference_mode():
            for task_combo in TASK_COMBINATIONS:
                available, candidate_masks, task_mask = task_spec(
                    task_combo, device
                )
                candidates = prediction[:, available]
                raw = model(candidates, ray, candidate_masks, task_mask)
                baseline = available.index(COMBINATIONS.index(task_combo))
                delta = raw - raw[..., baseline : baseline + 1]
                weights = F.softmax(-delta, dim=-1)
                fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                output[start:stop, COMBINATIONS.index(task_combo)] = (
                    fused.cpu().numpy().astype(np.float32)
                )
        if start == 0 or stop == len(predictions) or start % (batch_size * 20) == 0:
            print(f"fused {stop}/{len(predictions)}", flush=True)
    return output


def _action_equal_all17(predictions, targets, actions):
    errors = np.linalg.norm(predictions - targets, axis=-1)
    frame_values = errors.mean(axis=-1)
    values = [frame_values[actions == action].mean() for action in sorted(set(actions))]
    return float(np.mean(values) * 1000.0)


def _write_split(name, arrays, pkl, model, args, out_dir, device):
    metadata = _metadata_from_pkl(pkl, len(arrays["targets"]))
    for key in ("subjects", "actions"):
        if not np.array_equal(arrays[key], metadata[key]):
            raise RuntimeError(f"{name}: {key} mismatch between export and pkl")
    fused = _fuse_e2(arrays, model, device, args.batch_size)
    cache_path = out_dir / f"{name}_e2_frame_cache.npz"
    temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        group_indices=arrays["group_indices"],
        subjects=arrays["subjects"],
        actions=arrays["actions"],
        predictions=fused,
        targets=arrays["targets"],
    )
    temporary.replace(cache_path)
    windows = _build_windows(metadata, args.window_length, args.frame_stride)
    index_path = out_dir / f"{name}_temporal_index.npz"
    np.savez_compressed(index_path, **windows)
    taskwise = {
        str(COMBINATIONS[i]): _action_equal_all17(
            fused[:, i], arrays["targets"], arrays["actions"]
        )
        for i in range(len(COMBINATIONS))
    }
    report = {
        "split": name,
        "frame_groups": int(len(arrays["targets"])),
        "sequence_count": int(metadata["sequence_count"]),
        "windows": int(len(windows["window_indices"])),
        "window_length": args.window_length,
        "frame_stride": args.frame_stride,
        "cache": str(cache_path),
        "index": str(index_path),
        "stage_task_average_action_equal_all17_mm": {
            "V2": float(np.mean([taskwise[str(COMBINATIONS[i])] for i in range(0, 6)])),
            "V3": float(np.mean([taskwise[str(COMBINATIONS[i])] for i in range(6, 10)])),
            "V4": taskwise[str(COMBINATIONS[10])],
        },
    }
    report["taskwise_action_equal_all17_mm"] = taskwise
    (out_dir / f"{name}_temporal_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}")
    model = _load_e2(args.checkpoint, device)
    train = load_arrays(args.train_shards)
    validation = load_arrays(args.validation_shards)
    reports = {
        "train": _write_split(
            "train", train, args.train_pkl, model, args, out_dir, device
        ),
        "validation": _write_split(
            "validation", validation, args.validation_pkl, model, args, out_dir, device
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps({"args": vars(args), "reports": reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
