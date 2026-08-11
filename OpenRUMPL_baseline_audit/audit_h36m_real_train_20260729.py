#!/usr/bin/env python3
"""Gate the real-H36M RUMPL training set before launching a long run."""

from __future__ import annotations

import argparse
import collections
import pickle
import sys
from pathlib import Path

import numpy as np


EXPECTED_SUBJECTS = {1, 5, 6, 7, 8}
EXPECTED_CAMERAS = {0, 1, 2, 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-pkl", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-records", type=int, default=312188)
    parser.add_argument("--expected-groups", type=int, default=78047)
    parser.add_argument(
        "--rumpl-root",
        default="/home/lixiaob/cjy/OpenRUMPL/RUMPL",
    )
    return parser.parse_args()


def check_serialized_dataset(path: Path, expected_records: int, expected_groups: int):
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if len(records) != expected_records:
        raise RuntimeError(f"record count {len(records)} != {expected_records}")

    subjects = {record["subject"] for record in records}
    if subjects != EXPECTED_SUBJECTS:
        raise RuntimeError(f"unexpected train subjects: {sorted(subjects)}")
    if subjects & {9, 11}:
        raise RuntimeError("S9/S11 leakage into the training set")

    groups = collections.defaultdict(list)
    confidence_min = np.inf
    confidence_max = -np.inf
    for index, record in enumerate(records):
        key = (
            record["subject"],
            record["action"],
            record["subaction"],
            record["image_id"],
        )
        groups[key].append(record["camera_id"])
        joints = np.asarray(record["joints_2d"])
        confidence = np.asarray(record["joints_2d_conf"])
        if joints.shape != (17, 2) or confidence.shape != (17, 1):
            raise RuntimeError(
                f"record {index}: invalid 2D shapes {joints.shape}/{confidence.shape}"
            )
        if not np.isfinite(joints).all() or not np.isfinite(confidence).all():
            raise RuntimeError(f"record {index}: non-finite 2D detection")
        confidence_min = min(confidence_min, float(confidence.min()))
        confidence_max = max(confidence_max, float(confidence.max()))

    if len(groups) != expected_groups:
        raise RuntimeError(f"group count {len(groups)} != {expected_groups}")
    bad_groups = [
        key for key, cameras in groups.items()
        if len(cameras) != 4 or set(cameras) != EXPECTED_CAMERAS
    ]
    if bad_groups:
        raise RuntimeError(f"{len(bad_groups)} incomplete camera groups; first={bad_groups[0]}")
    # MMPose heatmap maxima are detector scores rather than calibrated
    # probabilities.  The official RUMPL H36M/CMU preprocessors preserve
    # them verbatim, and a small fraction can therefore be slightly above
    # one.  Keep the raw values for protocol fidelity; only reject negative
    # or clearly corrupt magnitudes here.
    if confidence_min < 0.0 or confidence_max > 10.0:
        raise RuntimeError(
            "invalid MMPose score range: "
            f"[{confidence_min}, {confidence_max}]"
        )
    del records
    return confidence_min, confidence_max


def check_runtime_loader(rumpl_root: Path, config_path: Path, expected_groups: int):
    lib_path = str(rumpl_root / "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    from core.config import config, update_config
    from dataset.multiview_h36m_rumpl import MultiViewH36M_RUMPL

    update_config(str(config_path))
    if config.DATASET.MAX_NUM_VIEWS != 4 or config.DATASET.MIN_NUM_VIEWS != 2:
        raise RuntimeError(
            "real H36M must use random view capacity 2-4, got "
            f"{config.DATASET.MIN_NUM_VIEWS}-{config.DATASET.MAX_NUM_VIEWS}"
        )
    if not config.DATASET.TRAIN_RANDOM_NUM_VIEWS:
        raise RuntimeError("TRAIN_RANDOM_NUM_VIEWS must be enabled")
    if set(config.DATASET.TRAIN_H36M_CALIB_ACTORS) != EXPECTED_SUBJECTS:
        raise RuntimeError(
            "training subject config mismatch: "
            f"{config.DATASET.TRAIN_H36M_CALIB_ACTORS}"
        )

    dataset = MultiViewH36M_RUMPL(
        config,
        config.DATASET.TRAIN_SUBSET,
        True,
        transform=None,
    )
    if len(dataset) != expected_groups:
        raise RuntimeError(f"runtime loader groups {len(dataset)} != {expected_groups}")

    sample = dataset[0]
    if len(sample) != 6:
        raise RuntimeError(f"runtime sample has {len(sample)} fields, expected 6")
    middle, closest, target, rays, _, joints_2d = sample
    expected_shapes = (
        ((17, 1, 3), tuple(middle.shape), "middle"),
        ((17, 4, 4), tuple(closest.shape), "closest"),
        ((17, 3), tuple(target.shape), "target"),
        ((17, 4, 7), tuple(rays.shape), "rays"),
        # Runtime packs 2D coordinates, confidence and the camera terms used
        # by the optional reprojection path.  function_rumpl.py explicitly
        # defines the current interface as 20 channels.
        ((17, 4, 20), tuple(joints_2d.shape), "joints_2d"),
    )
    for expected, actual, name in expected_shapes:
        if actual != expected:
            raise RuntimeError(f"{name} shape {actual} != {expected}")
    for value in (middle, closest, target, rays, joints_2d):
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        if not np.isfinite(array).all():
            raise RuntimeError("runtime sample contains non-finite values")


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_pkl)
    config_path = Path(args.config)
    confidence_min, confidence_max = check_serialized_dataset(
        dataset_path,
        args.expected_records,
        args.expected_groups,
    )
    check_runtime_loader(
        Path(args.rumpl_root),
        config_path,
        args.expected_groups,
    )
    print(
        "H36M real-train gate passed: "
        f"records={args.expected_records}, groups={args.expected_groups}, "
        f"subjects={sorted(EXPECTED_SUBJECTS)}, cameras={sorted(EXPECTED_CAMERAS)}, "
        f"confidence=[{confidence_min:.6f}, {confidence_max:.6f}]"
    )


if __name__ == "__main__":
    main()
