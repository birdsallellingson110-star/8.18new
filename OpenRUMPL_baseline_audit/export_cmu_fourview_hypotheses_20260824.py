#!/usr/bin/env python3
"""Export H36M-trained RUMPL hypotheses on an unseen four-camera CMU set.

The script mirrors ``export_h76_train_subset_hypotheses_20260811.py`` but does
not assume Human3.6M metadata.  No CMU target is used by candidate generation;
targets are copied only so a separate evaluator can compute zero-shot MPJPE.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parent / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))
sys.path.insert(0, str(AUDIT))

from core.config import config, update_config  # noqa: E402
import dataset  # noqa: E402
from export_h76_train_subset_hypotheses_20260811 import load_model, sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mmpose-type", required=True)
    parser.add_argument("--test-views", type=int, nargs=4, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def record_value(record: dict, key: str, default: int) -> int:
    value = record.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def main() -> None:
    args = parse_args()
    if len(set(args.test_views)) != 4:
        raise ValueError("--test-views must contain four unique camera IDs")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)

    update_config(args.cfg)
    config.DATASET.TEST_DATASET = "multiview_cmu_panoptic_rumpl"
    config.DATASET.TEST_SUBSET = "validation"
    config.DATASET.TEST_CMU_DATASET_NAME = args.dataset_name
    config.DATASET.TEST_MMPOSE_TYPE = args.mmpose_type
    config.DATASET.CMU_KEYPOINT_STANDARD = "h36m"
    config.DATASET.USE_MMPOSE_VAL = True
    config.DATASET.USE_MMPOSE_TEST = True
    config.DATASET.TEST_ON_ALL_CAMERAS = False
    config.DATASET.TEST_VIEWS = list(args.test_views)
    config.DATASET.ALL_VIEWS_CMU = list(args.test_views)
    config.DATASET.MAX_NUM_VIEWS = 4
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4
    config.DATASET.FLIP_LOWER_BODY_KP_TEST = False
    config.DATASET.APPLY_NOISE = False
    config.DATASET.APPLY_NOISE_MISSING = False
    config.DATASET.MISSING_LEVEL = 0.0
    config.DATASET.USE_MMPOSE_VAL = True
    config.GPUS = "0"
    config.WORKERS = args.workers
    config.WANDB = False

    dataset_class = getattr(dataset, config.DATASET.TEST_DATASET)
    pose_dataset = dataset_class(config, "validation", False, transform=None)
    if args.max_groups:
        pose_dataset.grouping = pose_dataset.grouping[: args.max_groups]
        pose_dataset.group_size = len(pose_dataset.grouping)
    if not pose_dataset.grouping:
        raise RuntimeError("CMU loader produced no synchronized four-view groups")

    actions = []
    subjects = []
    frame_keys = []
    for group in pose_dataset.grouping:
        record = pose_dataset.db[group[0]]
        pose_id = str(record.get("pose_id", ""))
        actions.append(5 if pose_id.endswith("pose5") else 6)
        subjects.append(record_value(record, "subject", 0))
        frame_keys.append(record_value(record, "image_id", len(frame_keys)))

    loader = DataLoader(
        pose_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda:0")
    checkpoint = Path(args.checkpoint).resolve()
    model = load_model(checkpoint, device)
    combinations = tuple(
        combo
        for count in (2, 3, 4)
        for combo in itertools.combinations(range(4), count)
    )

    prediction_batches = []
    target_batches = []
    ray_batches = []
    for batch_index, (_, _, target, rays, _, _) in enumerate(loader):
        rays_gpu = rays.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.inference_mode():
            candidates = [
                model(rays_gpu[:, :, list(combo), :], is_training=False)
                for combo in combinations
            ]
        prediction_batches.append(
            torch.stack(candidates, dim=1).cpu().numpy().astype(np.float32)
        )
        target_batches.append(target.numpy().astype(np.float32))
        ray_batches.append(rays.numpy().astype(np.float32))
        if batch_index % 10 == 0:
            done = min((batch_index + 1) * args.batch_size, len(pose_dataset))
            print(f"views={args.test_views} groups={done}/{len(pose_dataset)}", flush=True)

    predictions = np.concatenate(prediction_batches)
    targets = np.concatenate(target_batches)
    rays = np.concatenate(ray_batches)
    expected = (len(pose_dataset), len(combinations), 17, 3)
    if predictions.shape != expected:
        raise ValueError(f"expected predictions {expected}, got {predictions.shape}")
    if rays.shape[1:] != (17, 4, 7):
        raise ValueError(f"unexpected ray shape {rays.shape}")
    if not all(np.isfinite(item).all() for item in (predictions, targets, rays)):
        raise ValueError("non-finite CMU export values")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        group_indices=np.arange(len(pose_dataset), dtype=np.int64),
        frame_keys=np.asarray(frame_keys, dtype=np.int64),
        actions=np.asarray(actions, dtype=np.int16),
        subjects=np.asarray(subjects, dtype=np.int16),
        predictions=predictions,
        targets=targets,
        rays=rays,
    )
    temporary.replace(output)

    metadata = {
        "purpose": "H36M-trained frozen generator zero-shot CMU hypothesis export",
        "training_dataset": "H36M only",
        "test_dataset": "CMU Panoptic pose5/pose6",
        "uses_cmu_targets_for_candidate_generation": False,
        "camera_views": list(args.test_views),
        "camera_slots_are_local_and_have_no_learned_embedding": True,
        "groups": len(pose_dataset),
        "candidate_combinations_local_slots": [list(item) for item in combinations],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(Path(args.cfg).resolve()),
        "dataset_name": args.dataset_name,
        "mmpose_type": args.mmpose_type,
        "prediction_shape": list(predictions.shape),
        "target_shape": list(targets.shape),
        "ray_shape": list(rays.shape),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
