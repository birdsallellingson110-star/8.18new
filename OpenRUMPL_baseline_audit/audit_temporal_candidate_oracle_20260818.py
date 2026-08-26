#!/usr/bin/env python3
"""Zero-training temporal candidate oracle for the current C2 22-candidate pool.

Candidate generation is label-free.  This script reads labels only to measure
an upper bound after generation.  For every temporal window it selects a
candidate per joint using the mean error over the window, then scores that
candidate on the center frame.  This is an optimistic proxy for a temporal
utility head and is therefore a go/no-go test, not a deployable method.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pickle
from pathlib import Path

import numpy as np


COMBINATIONS = tuple(
    combo
    for views in (2, 3, 4)
    for combo in itertools.combinations(range(4), views)
)
EXPANDED = COMBINATIONS + COMBINATIONS  # H76 + confidence-weighted duplicate
STAGES = {
    "V2": tuple(combo for combo in COMBINATIONS if len(combo) == 2),
    "V3": tuple(combo for combo in COMBINATIONS if len(combo) == 3),
    "V4": tuple(combo for combo in COMBINATIONS if len(combo) == 4),
}
COMB_INDEX = {combo: i for i, combo in enumerate(COMBINATIONS)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window-lengths", nargs="+", type=int, default=[1, 3, 5, 9])
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    return p.parse_args()


def metadata_from_pkl(path: str, expected_groups: int) -> dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        records = pickle.load(handle)
    groups: collections.OrderedDict[tuple[int, int, int, int], set[int]] = (
        collections.OrderedDict()
    )
    for record in records:
        key = (
            int(record["subject"]),
            int(record["action"]),
            int(record["subaction"]),
            int(record["image_id"]),
        )
        groups.setdefault(key, set()).add(int(record["camera_id"]))
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"pkl has {len(groups)} synchronized groups, cache has {expected_groups}"
        )
    bad = [(key, sorted(cams)) for key, cams in groups.items() if cams != {0, 1, 2, 3}]
    if bad:
        raise RuntimeError(f"incomplete four-view groups: {bad[:3]}")
    keys = list(groups)
    seq_map: dict[tuple[int, int, int], int] = {}
    sequence_ids = []
    for subject, action, subaction, _ in keys:
        seq = (subject, action, subaction)
        seq_map.setdefault(seq, len(seq_map))
        sequence_ids.append(seq_map[seq])
    return {
        "subjects": np.asarray([key[0] for key in keys], dtype=np.int16),
        "actions": np.asarray([key[1] for key in keys], dtype=np.int16),
        "frame_ids": np.asarray([key[3] for key in keys], dtype=np.int64),
        "sequence_ids": np.asarray(sequence_ids, dtype=np.int32),
    }


def build_windows(metadata: dict[str, np.ndarray], length: int, stride: int):
    n = len(metadata["frame_ids"])
    if length == 1:
        windows = np.arange(n, dtype=np.int64)[:, None]
    else:
        by_sequence: dict[int, list[int]] = collections.defaultdict(list)
        for index, seq in enumerate(metadata["sequence_ids"].tolist()):
            by_sequence[int(seq)].append(index)
        rows: list[list[int]] = []
        for indices in by_sequence.values():
            indices.sort(key=lambda i: int(metadata["frame_ids"][i]))
            for start in range(0, len(indices) - length + 1):
                row = indices[start : start + length]
                if np.all(np.diff(metadata["frame_ids"][row]) == stride):
                    rows.append(row)
        if not rows:
            raise RuntimeError(f"no windows for T={length}, stride={stride}")
        windows = np.asarray(rows, dtype=np.int64)
    center = length // 2
    return {
        "window_indices": windows,
        "subjects": metadata["subjects"][windows[:, center]],
        "actions": metadata["actions"][windows[:, center]],
        "center_group_indices": windows[:, center],
    }


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(
        np.mean(
            [values[actions == action].mean() for action in sorted(set(actions.tolist()))]
        )
    )


def task_indices(task: tuple[int, ...]) -> tuple[np.ndarray, int]:
    available = np.asarray(
        [i for i, combo in enumerate(EXPANDED) if set(combo).issubset(task)],
        dtype=np.int64,
    )
    baseline = COMB_INDEX[task]
    return available, baseline


def audit_task(
    predictions,
    targets,
    windows: np.ndarray,
    task: tuple[int, ...],
    batch_size: int,
) -> dict[str, float]:
    available, baseline = task_indices(task)
    center = windows.shape[1] // 2
    sums = collections.defaultdict(float)
    counts = 0
    action_values: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        frame_indices = windows[start:stop]
        pose = np.asarray(predictions[frame_indices][:, :, available], dtype=np.float32)
        gt = np.asarray(targets[frame_indices], dtype=np.float32)
        errors = np.linalg.norm(pose - gt[:, :, None, :, :], axis=-1)
        center_errors = errors[:, center]
        baseline_error = center_errors[:, np.where(available == baseline)[0][0]]
        center_oracle = center_errors.min(axis=1)
        # Aggregate each candidate over time, then select one candidate per joint.
        temporal_scores = errors.mean(axis=1)
        temporal_choice = temporal_scores.argmin(axis=1)
        temporal_selected = np.take_along_axis(
            center_errors, temporal_choice[:, None, :], axis=1
        )[:, 0]
        # The candidate oracle is per joint; average joints only after selection.
        values = {
            "baseline": baseline_error.mean(axis=-1),
            "center_oracle": center_oracle.mean(axis=-1),
            "temporal_oracle": temporal_selected.mean(axis=-1),
        }
        counts += stop - start
        for key, value in values.items():
            sums[key] += float(value.sum())
        # The caller supplies action labels through the window order separately.
        for key, value in values.items():
            action_values[key]["all"].extend((value * 1000.0).tolist())
    # Frame-weighted values are retained for diagnostics. Action-equal values
    # are filled by the caller from the per-window vectors below.
    return {
        key + "_frame_weighted_mm": value / max(counts, 1) * 1000.0
        for key, value in sums.items()
    }


def audit(args: argparse.Namespace) -> dict:
    cache = np.load(args.cache, mmap_mode="r")
    required = {"predictions", "targets", "subjects", "actions", "group_indices"}
    missing = required.difference(cache.files)
    if missing:
        raise RuntimeError(f"cache missing fields: {sorted(missing)}")
    predictions = cache["predictions"]
    targets = cache["targets"]
    if predictions.ndim != 4 or predictions.shape[1:] != (22, 17, 3):
        raise RuntimeError(f"expected C2 22-candidate cache, got {predictions.shape}")
    metadata = metadata_from_pkl(args.validation_pkl, len(targets))
    for key in ("subjects", "actions"):
        if not np.array_equal(cache[key], metadata[key]):
            raise RuntimeError(f"cache/{key} does not match validation pkl order")

    report = {
        "method": "zero-training temporal candidate utility oracle",
        "cache": str(Path(args.cache).resolve()),
        "validation_pkl": str(Path(args.validation_pkl).resolve()),
        "candidate_count": int(predictions.shape[1]),
        "candidate_order": "H76 11 + confidence-weighted 11",
        "frame_stride": int(args.frame_stride),
        "subjects": sorted(set(metadata["subjects"].tolist())),
        "stages": {},
    }
    for length in args.window_lengths:
        windows = build_windows(metadata, length, args.frame_stride)["window_indices"]
        center = length // 2
        stage_report = {
            "window_count": int(len(windows)),
            "tasks": {},
            "mean_action_equal_mm": {},
            "mean_frame_weighted_mm": {},
        }
        for stage, tasks in STAGES.items():
            task_results = {}
            action_results = collections.defaultdict(list)
            frame_results = collections.defaultdict(list)
            for task in tasks:
                available, baseline = task_indices(task)
                baseline_local = int(np.where(available == baseline)[0][0])
                task_values: dict[str, list[np.ndarray]] = collections.defaultdict(list)
                task_actions: list[np.ndarray] = []
                for start in range(0, len(windows), args.batch_size):
                    stop = min(start + args.batch_size, len(windows))
                    frame_indices = windows[start:stop]
                    pose = np.asarray(predictions[frame_indices][:, :, available], dtype=np.float32)
                    gt = np.asarray(targets[frame_indices], dtype=np.float32)
                    errors = np.linalg.norm(pose - gt[:, :, None, :, :], axis=-1)
                    center_errors = errors[:, center]
                    temporal_scores = errors.mean(axis=1)
                    choice = temporal_scores.argmin(axis=1)
                    selected = np.take_along_axis(
                        center_errors, choice[:, None, :], axis=1
                    )[:, 0]
                    vals = {
                        "baseline": center_errors[:, baseline_local].mean(axis=-1),
                        "center_oracle": center_errors.min(axis=1).mean(axis=-1),
                        "temporal_oracle": selected.mean(axis=-1),
                    }
                    for key, value in vals.items():
                        task_values[key].append(value * 1000.0)
                    task_actions.append(metadata["actions"][frame_indices[:, center]])
                values = {key: np.concatenate(parts) for key, parts in task_values.items()}
                actions = np.concatenate(task_actions)
                task_results[str(task)] = {
                    "candidate_combinations": [list(EXPANDED[i]) for i in available],
                    "baseline_candidate": list(task),
                    "action_equal_mm": {
                        key: action_equal(value, actions) for key, value in values.items()
                    },
                    "frame_weighted_mm": {
                        key: float(value.mean()) for key, value in values.items()
                    },
                }
                for key in values:
                    action_results[key].append(task_results[str(task)]["action_equal_mm"][key])
                    frame_results[key].append(task_results[str(task)]["frame_weighted_mm"][key])
            stage_report["tasks"][stage] = task_results
            stage_report["mean_action_equal_mm"][stage] = {
                key: float(np.mean(value)) for key, value in action_results.items()
            }
            stage_report["mean_frame_weighted_mm"][stage] = {
                key: float(np.mean(value)) for key, value in frame_results.items()
            }
        report["stages"][f"T{length}"] = stage_report
    return report


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = audit(args)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "COMPLETED").write_text("completed\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
