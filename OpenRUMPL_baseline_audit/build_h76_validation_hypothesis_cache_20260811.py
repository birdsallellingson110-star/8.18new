#!/usr/bin/env python3
"""Combine existing strict H76 V2/V3/V4 exports into one validation pool."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))

import dataset  # noqa: E402
from core.config import config, update_config  # noqa: E402
from diagnose_h76_multiview_bottleneck_20260808 import (  # noqa: E402
    build_four_view_groups,
    load_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mmpose-type", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_config(args.cfg)
    config.DATASET.TEST_H36M_DATASET_NAME = args.dataset_name
    config.DATASET.TEST_MMPOSE_TYPE = args.mmpose_type
    config.DATASET.USE_MMPOSE_VAL = True
    config.DATASET.USE_MMPOSE_TEST = True
    config.DATASET.TEST_ON_ALL_CAMERAS = True
    config.DATASET.TEST_VIEWS = [1, 2, 3, 4]
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4

    dataset_class = getattr(dataset, config.DATASET.TEST_DATASET)
    pose_dataset = dataset_class(
        config, config.DATASET.TEST_SUBSET, False, transform=None
    )
    groups = build_four_view_groups(pose_dataset.db)
    if len(groups) != len(pose_dataset):
        raise ValueError("four-view group reconstruction mismatch")
    targets = []
    rays = []
    for group_index in range(len(groups)):
        _, _, target, ray, _, _ = pose_dataset[group_index]
        targets.append(target.numpy())
        rays.append(ray.numpy())
    targets = np.asarray(targets, dtype=np.float32)
    rays = np.asarray(rays, dtype=np.float32)

    candidate_predictions = []
    candidate_combinations = []
    prediction_root = Path(args.prediction_root)
    for views in (2, 3, 4):
        predictions, exported_targets = load_predictions(prediction_root, views)
        combinations = list(itertools.combinations(range(4), views))
        predictions = predictions.reshape(len(groups), len(combinations), 17, 3)
        exported_targets = exported_targets.reshape(
            len(groups), len(combinations), 17, 3
        )
        expected_targets = np.repeat(
            targets[:, None], len(combinations), axis=1
        )
        if not np.allclose(exported_targets, expected_targets, atol=2e-4):
            raise ValueError(f"V{views} target ordering mismatch")
        candidate_predictions.append(predictions.astype(np.float32))
        candidate_combinations.extend(combinations)
    predictions = np.concatenate(candidate_predictions, axis=1)
    actions = np.asarray(
        [int(pose_dataset.db[group[0]]["action"]) for group in groups],
        dtype=np.int16,
    )
    subjects = np.asarray(
        [int(pose_dataset.db[group[0]]["subject"]) for group in groups],
        dtype=np.int16,
    )

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        group_indices=np.arange(len(groups), dtype=np.int64),
        actions=actions,
        subjects=subjects,
        predictions=predictions,
        targets=targets,
        rays=rays,
    )
    temporary.replace(output)
    metadata = {
        "purpose": "H76 GHT-style validation hypothesis pool",
        "split": "H36M S9/S11 validation/test only; never utility training",
        "groups": len(groups),
        "subjects": sorted(set(subjects.tolist())),
        "candidate_combinations_zero_based": [
            list(combo) for combo in candidate_combinations
        ],
        "prediction_shape": list(predictions.shape),
        "target_shape": list(targets.shape),
        "ray_shape": list(rays.shape),
        "prediction_root": str(prediction_root.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
