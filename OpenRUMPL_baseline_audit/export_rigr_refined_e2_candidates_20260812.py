#!/usr/bin/env python3
"""Export aligned RIGR-refined candidates for a subsequent E2 retraining.

The HRNet feature checkpoint was trained on the fixed, coverage-balanced 20k
group list.  This script preserves that exact order for feature inference, then
joins the refined first eleven H76 candidates with the already exported frozen
pairwise/learned candidates.  The resulting NPZ shards are suitable for
``train_h76_learned_candidate_e2_20260814.py`` and contain no GT-derived
inference feature.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_rigr_e2_cascade_20260812 import load_rigr, refine_candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h76-train", nargs="+", required=True)
    p.add_argument("--e2-train", nargs="+", required=True)
    p.add_argument("--selected-group-ids", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--rigr-checkpoint", required=True)
    p.add_argument("--validation-h76", required=True)
    p.add_argument("--validation-e2", required=True)
    p.add_argument("--validation-tokens", required=True)
    p.add_argument("--validation-rigr-checkpoint", default="",
                   help="Defaults to --rigr-checkpoint; kept explicit for audit metadata")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--shards", type=int, default=2)
    return p.parse_args()


def load_files(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([source[key] for source in loaded], axis=0) for key in keys}
    order = np.argsort(arrays["group_indices"])
    return {key: value[order] for key, value in arrays.items()}


def select_rows(arrays: dict[str, np.ndarray], group_ids: np.ndarray) -> dict[str, np.ndarray]:
    lookup = {int(group): index for index, group in enumerate(arrays["group_indices"])}
    rows = np.asarray([lookup[int(group)] for group in group_ids], dtype=np.int64)
    return {key: value[rows] for key, value in arrays.items()}


def replace_first_eleven(base: dict[str, np.ndarray], refined: np.ndarray) -> dict[str, np.ndarray]:
    predictions = np.array(base["predictions"], copy=True)
    if predictions.shape[1:] != (22, 17, 3):
        raise ValueError(f"expected 22-candidate E2 cache, got {predictions.shape}")
    if refined.shape != (len(predictions), 11, 17, 3):
        raise ValueError(f"bad refined H76 candidates {refined.shape}")
    predictions[:, :11] = refined
    return {
        "group_indices": base["group_indices"],
        "actions": base["actions"],
        "subjects": base["subjects"],
        "predictions": predictions.astype(np.float32),
        "targets": base["targets"].astype(np.float32),
        "rays": base["rays"].astype(np.float32),
    }


def write_shards(arrays: dict[str, np.ndarray], output_dir: Path, prefix: str, count: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(np.arange(len(arrays["targets"])), count)
    paths = []
    for index, rows in enumerate(chunks):
        path = output_dir / f"{prefix}_shard{index}of{count}.npz"
        np.savez_compressed(path, **{key: value[rows] for key, value in arrays.items()})
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h76_train = load_files(args.h76_train)
    e2_train = load_files(args.e2_train)
    selected = np.asarray(np.load(args.selected_group_ids), dtype=np.int64).reshape(-1)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("duplicate selected group IDs")
    selected_h76 = select_rows(h76_train, selected)
    selected_e2 = select_rows(e2_train, selected)
    train_tokens = np.load(args.tokens, mmap_mode="r")
    model = load_rigr(args.rigr_checkpoint, train_tokens.shape, device)
    refined_train = refine_candidates(
        selected_h76, args.tokens, model, device, args.batch_size
    )
    train_combined = replace_first_eleven(selected_e2, refined_train)
    train_paths = write_shards(train_combined, output_dir, "train_rigr_e2", args.shards)

    h76_val = load_files([args.validation_h76])
    e2_val = load_files([args.validation_e2])
    val_tokens = np.load(args.validation_tokens, mmap_mode="r")
    val_checkpoint = args.validation_rigr_checkpoint or args.rigr_checkpoint
    val_model = load_rigr(val_checkpoint, val_tokens.shape, device)
    refined_val = refine_candidates(
        h76_val, args.validation_tokens, val_model, device, args.batch_size
    )
    val_combined = replace_first_eleven(e2_val, refined_val)
    val_path = output_dir / "validation_rigr_e2.npz"
    np.savez_compressed(val_path, **val_combined)
    np.save(output_dir / "train_refined_first11.npy", refined_train)
    np.save(output_dir / "validation_refined_first11.npy", refined_val)
    manifest = {
        "method": "HRNet intermediate-feature RIGR candidate correction followed by E2 retraining",
        "train_groups": int(len(train_combined["targets"])),
        "validation_groups": int(len(val_combined["targets"])),
        "candidate_count": 22,
        "refined_candidate_count": 11,
        "selected_group_ids": str(Path(args.selected_group_ids).resolve()),
        "train_tokens": str(Path(args.tokens).resolve()),
        "validation_tokens": str(Path(args.validation_tokens).resolve()),
        "rigr_checkpoint": str(Path(args.rigr_checkpoint).resolve()),
        "train_shards": train_paths,
        "validation_cache": str(val_path),
        "args": vars(args),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
