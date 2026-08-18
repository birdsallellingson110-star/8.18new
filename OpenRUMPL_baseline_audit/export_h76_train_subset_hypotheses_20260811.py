#!/usr/bin/env python3
"""Export frozen-H76 predictions for every 2/3/4-view training subset.

The export is deliberately restricted to the official H36M training subjects.
It produces the hypothesis pool needed by the paper-backed GHT-style scorer
without training on, or selecting labels from, S9/S11.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))

from core.config import config, update_config  # noqa: E402
import dataset  # noqa: E402
import models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mmpose-type", required=True)
    parser.add_argument(
        "--flip-lower-body-kp-test",
        choices=("true", "false"),
        default="false",
        help=(
            "Match the strict coordinate-level evaluation protocol. The "
            "GBT-style fair line disables the legacy H36M lower-body swap."
        ),
    )
    parser.add_argument(
        "--subset", choices=("train", "validation"), default="train",
        help="H36M split to export; train is the historical default.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model = getattr(models, config.MODEL).get_multiview_rumpl_net(
        config, is_train=False
    )
    state = torch.load(checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    # H76 was trained with the historical two-view ``weighted_mean`` tensor,
    # which is inactive in its FPT path.  Official V3/V4 evaluation constructs
    # the four-view-capacity model and skips exactly that shape-only tensor.
    from utils.rumpl_checkpoint_adapt import merge_pretrained_into_model_state

    model_state = model.state_dict()
    shape_mismatches = {
        key for key, value in state.items()
        if key in model_state and value.shape != model_state[key].shape
    }
    # Depending on the checkpoint/config pair, the inactive weighted-mean
    # tensor can either be the historical two-view shape or already match the
    # four-view-capacity model.  Both are valid for frozen candidate export;
    # reject only unexpected real-layer mismatches.
    if shape_mismatches not in (set(), {"features.weighted_mean.weight"}):
        raise RuntimeError(
            f"unexpected H76 checkpoint shape mismatches: {shape_mismatches}"
        )
    merged, skipped = merge_pretrained_into_model_state(
        model_state, state, strict_shapes=False
    )
    if skipped:
        raise RuntimeError(f"unexpected H76 checkpoint skips: {skipped}")
    model.load_state_dict(merged, strict=True)
    model.to(device).eval()
    return model


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.set_grad_enabled(False)

    update_config(args.cfg)
    config.DATASET.TEST_SUBSET = args.subset
    config.DATASET.TEST_H36M_DATASET_NAME = args.dataset_name
    config.DATASET.TEST_MMPOSE_TYPE = args.mmpose_type
    config.DATASET.USE_MMPOSE_VAL = True
    config.DATASET.USE_MMPOSE_TEST = True
    config.DATASET.TEST_ON_ALL_CAMERAS = True
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4
    config.DATASET.TEST_VIEWS = [1, 2, 3, 4]
    config.DATASET.FLIP_LOWER_BODY_KP_TEST = args.flip_lower_body_kp_test == "true"
    config.GPUS = "0"
    config.WORKERS = args.workers
    config.WANDB = False

    dataset_class = getattr(dataset, config.DATASET.TEST_DATASET)
    pose_dataset = dataset_class(config, args.subset, False, transform=None)
    total_groups = len(pose_dataset.grouping)
    start = total_groups * args.shard_index // args.num_shards
    stop = total_groups * (args.shard_index + 1) // args.num_shards
    if args.max_groups:
        stop = min(stop, start + args.max_groups)
    original_grouping = pose_dataset.grouping
    selected_grouping = original_grouping[start:stop]
    actions = np.asarray(
        [int(pose_dataset.db[group[0]]["action"]) for group in selected_grouping],
        dtype=np.int16,
    )
    subjects = np.asarray(
        [int(pose_dataset.db[group[0]]["subject"]) for group in selected_grouping],
        dtype=np.int16,
    )
    pose_dataset.grouping = selected_grouping
    pose_dataset.group_size = len(selected_grouping)

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

    candidate_combinations = [
        combo
        for views in (2, 3, 4)
        for combo in itertools.combinations(range(4), views)
    ]
    prediction_batches = []
    target_batches = []
    ray_batches = []
    for batch_index, (_, _, target, rays, _, _) in enumerate(loader):
        rays_gpu = rays.to(device=device, dtype=torch.float32, non_blocking=True)
        candidates = []
        with torch.inference_mode():
            for combo in candidate_combinations:
                subset = rays_gpu[:, :, list(combo), :]
                candidates.append(model(subset, is_training=False))
        prediction_batches.append(
            torch.stack(candidates, dim=1).cpu().numpy().astype(np.float32)
        )
        target_batches.append(target.numpy().astype(np.float32))
        ray_batches.append(rays.numpy().astype(np.float32))
        if batch_index % 50 == 0:
            completed = min((batch_index + 1) * args.batch_size, len(pose_dataset))
            print(
                f"shard={args.shard_index}/{args.num_shards} "
                f"groups={completed}/{len(pose_dataset)}",
                flush=True,
            )

    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    rays = np.concatenate(ray_batches, axis=0)
    if predictions.shape != (len(pose_dataset), 11, 17, 3):
        raise ValueError(f"unexpected predictions shape {predictions.shape}")
    if not np.isfinite(predictions).all():
        raise ValueError("non-finite predictions")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        group_indices=np.arange(start, stop, dtype=np.int64),
        actions=actions,
        subjects=subjects,
        predictions=predictions,
        targets=targets,
        rays=rays,
    )
    temporary.replace(output)
    metadata = {
        "purpose": "H76 GHT-style training hypothesis pool",
        "split": f"H36M {args.subset} split",
        "subjects": sorted(set(subjects.tolist())),
        "total_four_view_groups": total_groups,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "group_range": [start, stop],
        "groups": len(pose_dataset),
        "candidate_combinations_zero_based": [
            list(combo) for combo in candidate_combinations
        ],
        "prediction_shape": list(predictions.shape),
        "target_shape": list(targets.shape),
        "ray_shape": list(rays.shape),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(Path(args.cfg).resolve()),
        "dataset_name": args.dataset_name,
        "mmpose_type": args.mmpose_type,
        "flip_lower_body_kp_test": bool(config.DATASET.FLIP_LOWER_BODY_KP_TEST),
        "h76_environment": {
            name: os.environ.get(name)
            for name in (
                "RUMPL_TRI_ANCHOR", "RUMPL_TRI_ANCHOR_REG",
                "RUMPL_TRI_ANCHOR_CONF_EPS", "RUMPL_PFT_REPEAT_LAST",
                "RUMPL_ANCHOR_CENTERED_RAYS", "RUMPL_INPUT_PLUCKER",
            )
        },
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
