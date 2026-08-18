#!/usr/bin/env python3
"""Audit temporal signal in frozen E-2 candidates and prepare fair windows.

This is intentionally a zero-training stage.  It measures whether the oracle
candidate identity and counterfactual candidate error are temporally stable,
then writes T=3/T=5/T=9 indices with exactly the same center frames as the
existing strict T=9 protocol.  No S9/S11 value is used for model selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


COMBINATIONS = tuple(
    combo
    for views in (2, 3, 4)
    for combo in itertools.combinations(range(4), views)
)
TASKS = tuple(itertools.combinations(range(4), 3)) + ((0, 1, 2, 3),)
STAGE_TASKS = {"V3": TASKS[:4], "V4": TASKS[4:]}
TASK_INDEX = {combo: index for index, combo in enumerate(COMBINATIONS)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-index", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-index", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    return p.parse_args()


def task_candidates(task):
    available = [
        i for i, combo in enumerate(COMBINATIONS)
        if set(combo).issubset(task)
    ]
    baseline = available.index(TASK_INDEX[task])
    return np.asarray(available, dtype=np.int64), baseline


def add_corr(acc, x, y):
    """Accumulate Pearson sufficient statistics along a flattened batch."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    acc[0] += float(x.size)
    acc[1] += float(x.sum())
    acc[2] += float(y.sum())
    acc[3] += float(np.dot(x, x))
    acc[4] += float(np.dot(y, y))
    acc[5] += float(np.dot(x, y))


def finish_corr(acc):
    n, sx, sy, sxx, syy, sxy = acc
    if n < 2:
        return None
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    cov = sxy - sx * sy / n
    denom = np.sqrt(max(vx, 0.0) * max(vy, 0.0))
    return float(cov / denom) if denom > 1e-12 else 0.0


def build_common_indices(index9, output_dir, split):
    windows9 = index9["window_indices"].astype(np.int64)
    frames9 = index9["window_frame_ids"].astype(np.int64)
    center = windows9.shape[1] // 2
    for length in (3, 5, 9):
        half = length // 2
        if length == 9:
            windows = windows9
            frames = frames9
        else:
            windows = windows9[:, center - half:center + half + 1]
            frames = frames9[:, center - half:center + half + 1]
        payload = {
            "window_indices": windows,
            "window_frame_ids": frames,
            "subjects": index9["subjects"].astype(np.int16),
            "actions": index9["actions"].astype(np.int16),
            "sequence_ids": index9["sequence_ids"].astype(np.int32),
            "center_group_indices": index9["center_group_indices"].astype(np.int64),
        }
        out = output_dir / f"{split}_temporal_index_T{length}.npz"
        np.savez_compressed(out, **payload)
    return int(len(windows9))


def audit_split(cache_path, index_path, output_dir, split, batch_size):
    cache = np.load(cache_path, mmap_mode="r")
    index = np.load(index_path)
    predictions = cache["predictions"]
    targets = cache["targets"]
    windows = index["window_indices"].astype(np.int64)
    subjects = index["subjects"].astype(np.int16)
    actions = index["actions"].astype(np.int16)
    if predictions.shape[0] != len(cache["group_indices"]):
        raise RuntimeError(f"{split}: cache group count mismatch")
    if windows.shape[1] != 9:
        raise RuntimeError(f"{split}: expected strict T=9 index")

    report = {
        "split": split,
        "window_count": int(len(windows)),
        "subjects": sorted(set(subjects.tolist())),
        "stages": {},
    }
    for stage, tasks in STAGE_TASKS.items():
        stage_report = {"tasks": {}, "mean": {}}
        for task in tasks:
            available, baseline_local = task_candidates(task)
            task_index = TASK_INDEX[task]
            # One accumulator per offset for candidate delta correlation and
            # baseline error correlation.  Offset 0 is defined as 1.0.
            delta_acc = [[0.0] * 6 for _ in range(9)]
            base_acc = [[0.0] * 6 for _ in range(9)]
            persistence_count = np.zeros(9, dtype=np.int64)
            persistence_total = np.zeros(9, dtype=np.int64)
            oracle_sum = 0.0
            baseline_sum = 0.0
            sample_count = 0
            for start in range(0, len(windows), batch_size):
                stop = min(start + batch_size, len(windows))
                frame_indices = windows[start:stop]
                pose = np.asarray(predictions[frame_indices][:, :, available], dtype=np.float32)
                gt = np.asarray(targets[frame_indices], dtype=np.float32)
                # B,T,C,J,3 -> B,T,C,J
                errors = np.linalg.norm(pose - gt[:, :, None, :, :], axis=-1)
                baseline = errors[:, :, baseline_local]
                delta = errors - baseline[:, :, None]
                oracle = errors.min(axis=2)
                oracle_sum += float(oracle.sum())
                baseline_sum += float(baseline.sum())
                sample_count += int(np.prod(oracle.shape))
                oracle_id = errors.argmin(axis=2)
                center = oracle_id[:, 4]
                for offset_index, offset in enumerate(range(-4, 5)):
                    current = delta[:, 4 + offset]
                    add_corr(delta_acc[offset_index], current, delta[:, 4])
                    current_base = np.linalg.norm(
                        np.asarray(
                            predictions[frame_indices[:, 4 + offset], task_index],
                            dtype=np.float32,
                        ) - np.asarray(targets[frame_indices[:, 4 + offset]], dtype=np.float32),
                        axis=-1,
                    )
                    add_corr(base_acc[offset_index], current_base, baseline[:, 4])
                    equal = (oracle_id[:, 4 + offset] == center)
                    persistence_count[offset_index] += int(equal.sum())
                    persistence_total[offset_index] += int(equal.size)
            task_report = {
                "candidate_count": int(len(available)),
                "candidate_combinations": [COMBINATIONS[i] for i in available.tolist()],
                "baseline_candidate": task,
                "oracle_error_frame_weighted_mm": oracle_sum / sample_count * 1000.0,
                "baseline_error_frame_weighted_mm": baseline_sum / sample_count * 1000.0,
                "oracle_identity_persistence": {
                    str(offset): float(persistence_count[i] / max(persistence_total[i], 1))
                    for i, offset in enumerate(range(-4, 5))
                },
                "candidate_delta_pearson_to_center": {
                    str(offset): finish_corr(delta_acc[i])
                    for i, offset in enumerate(range(-4, 5))
                },
                "baseline_error_pearson_to_center": {
                    str(offset): finish_corr(base_acc[i])
                    for i, offset in enumerate(range(-4, 5))
                },
            }
            stage_report["tasks"][str(task)] = task_report
        for key in (
            "oracle_error_frame_weighted_mm",
            "baseline_error_frame_weighted_mm",
        ):
            values = [v[key] for v in stage_report["tasks"].values()]
            stage_report["mean"][key] = float(np.mean(values))
        for key in (
            "oracle_identity_persistence",
            "candidate_delta_pearson_to_center",
            "baseline_error_pearson_to_center",
        ):
            offsets = {}
            for offset in range(-4, 5):
                vals = [
                    stage_report["tasks"][task_key][key][str(offset)]
                    for task_key in stage_report["tasks"]
                ]
                offsets[str(offset)] = float(np.mean(vals))
            stage_report["mean"][key] = offsets
        report["stages"][stage] = stage_report
    return report


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for split, cache, index_path in (
        ("train", args.train_cache, args.train_index),
        ("validation", args.validation_cache, args.validation_index),
    ):
        index = np.load(index_path)
        count = build_common_indices(index, output_dir, split)
        reports[split] = audit_split(
            cache, index_path, output_dir, split, args.batch_size
        )
        reports[split]["common_center_count"] = count
        reports[split]["common_center_subject_counts"] = {
            str(subject): int((index["subjects"] == subject).sum())
            for subject in sorted(set(index["subjects"].tolist()))
        }
    payload = {
        "method": "zero-training temporal candidate utility audit",
        "window_lengths": [3, 5, 9],
        "same_centers": True,
        "frame_stride": 5,
        "reports": reports,
    }
    (output_dir / "I0_temporal_candidate_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
