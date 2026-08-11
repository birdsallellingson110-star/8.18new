#!/usr/bin/env python3
"""Combine strict H36M stage-V shards into a leakage-free synthetic split."""

from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np


ARRAY_KEYS = [
    "joints_3d",
    "joints_2d_mmpose",
    "confs_2d_mmpose",
    "joints_2d_amass",
    "triangulated_3d_mmpose",
    "camera_setup_used",
    "views_used",
]


def split_id(path: str) -> int:
    return int(Path(path).name.split("_")[1])


def combine(paths: list[str], output_path: Path) -> None:
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    cameras: list[list[dict]] = []
    for index, path in enumerate(paths, 1):
        with open(path, "rb") as handle:
            data = pickle.load(handle)
        for key in ARRAY_KEYS:
            arrays[key].append(np.asarray(data[key]))
        cameras.extend(data["camera_parameters_all"])
        if index % 10 == 0 or index == len(paths):
            print(f"loaded={index}/{len(paths)} samples={len(cameras)}", flush=True)

    combined = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    combined["camera_parameters_all"] = cameras
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(combined, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"saved={output_path} samples={combined['joints_3d'].shape[0]} "
        f"j3={combined['joints_3d'].shape} j2={combined['joints_2d_mmpose'].shape}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-start", type=int, default=90)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob), key=split_id)
    if [split_id(path) for path in paths] != list(range(99)):
        raise SystemExit("Expected exactly the complete split IDs 0..98")
    train_paths = [path for path in paths if split_id(path) < args.validation_start]
    validation_paths = [path for path in paths if split_id(path) >= args.validation_start]
    print(
        f"train_shards={len(train_paths)} validation_shards={len(validation_paths)} "
        f"validation_ids={[split_id(path) for path in validation_paths]}",
        flush=True,
    )
    combine(train_paths, args.output_dir / "amass_mmpose_joints_train.pkl")
    combine(validation_paths, args.output_dir / "amass_mmpose_joints_validation.pkl")


if __name__ == "__main__":
    main()
