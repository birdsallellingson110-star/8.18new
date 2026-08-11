#!/usr/bin/env python3
"""Evaluate a RUMPL state dict without constructing the training dataset."""

import argparse
import json
import logging
import os
import random
import sys

import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from core.config import config, update_config
from core.function_rumpl import validate
from core.loss import MPJPE
import dataset
import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--use-mmpose-val",
        choices=("config", "true", "false"),
        default="config",
        help="Override DATASET.USE_MMPOSE_VAL for protocol diagnostics.",
    )
    parser.add_argument(
        "--flip-lower-body-kp-test",
        choices=("config", "true", "false"),
        default="config",
        help="Override the H36M lower-body semantic correction.",
    )
    parser.add_argument(
        "--test-views",
        type=int,
        nargs="+",
        help="Override DATASET.TEST_VIEWS (for example: --test-views 1 3).",
    )
    parser.add_argument(
        "--train-views",
        type=int,
        nargs="+",
        help=(
            "Override DATASET.TRAIN_VIEWS. The public model constructor uses "
            "this length even in evaluation when all-camera mode is disabled."
        ),
    )
    parser.add_argument(
        "--test-on-all-cameras",
        choices=("config", "true", "false"),
        default="config",
        help="Override DATASET.TEST_ON_ALL_CAMERAS.",
    )
    parser.add_argument(
        "--n-views-combinations",
        type=int,
        choices=(2, 3, 4, 5),
        help=(
            "When TEST_ON_ALL_CAMERAS is enabled, evaluate all C(5,k) "
            "camera subsets from ALL_VIEWS_CMU (RUMPL CMU V2..V5 protocol)."
        ),
    )
    parser.add_argument(
        "--all-views-cmu",
        type=int,
        nargs="+",
        help="Override DATASET.ALL_VIEWS_CMU (default 3 6 12 13 23).",
    )
    parser.add_argument(
        "--test-mmpose-type",
        help="Override DATASET.TEST_MMPOSE_TYPE for an input-pipeline audit.",
    )
    parser.add_argument(
        "--test-subset",
        help="Override DATASET.TEST_SUBSET (for example: train).",
    )
    parser.add_argument(
        "--sample-groups-per-action",
        type=int,
        default=0,
        help=(
            "After constructing the test view combinations, retain a "
            "deterministic stratified sample per H36M action."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--selection-manifest",
        help="Write selected database indices/camera IDs as JSON.",
    )
    args = parser.parse_args()

    update_config(args.cfg)
    if args.use_mmpose_val != "config":
        config.DATASET.USE_MMPOSE_VAL = args.use_mmpose_val == "true"
    if args.flip_lower_body_kp_test != "config":
        config.DATASET.FLIP_LOWER_BODY_KP_TEST = (
            args.flip_lower_body_kp_test == "true"
        )
    if args.test_views is not None:
        config.DATASET.TEST_VIEWS = args.test_views
    if args.train_views is not None:
        config.DATASET.TRAIN_VIEWS = args.train_views
    if args.test_on_all_cameras != "config":
        config.DATASET.TEST_ON_ALL_CAMERAS = (
            args.test_on_all_cameras == "true"
        )
    if args.n_views_combinations is not None:
        config.DATASET.N_VIEWS_TRAIN_TEST_ALL = args.n_views_combinations
        config.DATASET.TEST_ON_ALL_CAMERAS = True
        config.DATASET.TRAIN_ON_ALL_CAMERAS = True
        config.DATASET.TEST_VIEWS = list(
            config.DATASET.ALL_VIEWS_CMU or [3, 6, 12, 13, 23]
        )
    if args.all_views_cmu is not None:
        config.DATASET.ALL_VIEWS_CMU = list(args.all_views_cmu)
    if args.test_mmpose_type is not None:
        config.DATASET.TEST_MMPOSE_TYPE = args.test_mmpose_type
    if args.test_subset is not None:
        config.DATASET.TEST_SUBSET = args.test_subset
    config.GPUS = "0"
    config.WORKERS = args.workers
    config.WANDB = False

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    model = getattr(models, config.MODEL).get_multiview_rumpl_net(
        config, is_train=False
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    strict_load = os.environ.get("RUMPL_EVAL_STRICT", "1") == "1"
    if not strict_load:
        from utils.rumpl_checkpoint_adapt import merge_pretrained_into_model_state

        model_state = model.state_dict()
        state, skipped = merge_pretrained_into_model_state(
            model_state, state, strict_shapes=False
        )
        if skipped:
            logging.info("load_state_dict skipped shape-mismatch keys: %s", skipped)
    incompatible = model.load_state_dict(state, strict=strict_load)
    if not strict_load and incompatible:
        logging.info(
            "load_state_dict strict=False missing=%s unexpected=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = getattr(dataset, config.DATASET.TEST_DATASET)(
        config,
        config.DATASET.TEST_SUBSET,
        False,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    if args.sample_groups_per_action:
        by_action = {}
        for grouping_index, group in enumerate(valid_dataset.grouping):
            action = int(valid_dataset.db[group[0]]["action"])
            by_action.setdefault(action, []).append(grouping_index)
        rng = random.Random(args.sample_seed)
        selected_indices = []
        for action in sorted(by_action):
            candidates = by_action[action]
            rng.shuffle(candidates)
            selected_indices.extend(
                candidates[: args.sample_groups_per_action]
            )
        selected_indices.sort()
        valid_dataset.grouping = [
            valid_dataset.grouping[index] for index in selected_indices
        ]
        # multiview_h36m_rumpl caches this value for __len__.
        if hasattr(valid_dataset, "group_size"):
            valid_dataset.group_size = len(valid_dataset.grouping)
    if args.selection_manifest:
        manifest = {
            "test_subset": config.DATASET.TEST_SUBSET,
            "test_views": list(config.DATASET.TEST_VIEWS),
            "sample_groups_per_action": args.sample_groups_per_action,
            "sample_seed": args.sample_seed,
            "groups": [
                {
                    "record_indices": [int(index) for index in group],
                    "camera_ids": [
                        int(valid_dataset.db[index]["camera_id"])
                        for index in group
                    ],
                    "action": int(
                        valid_dataset.db[group[0]]["action"]
                    ),
                    "images": [
                        valid_dataset.db[index]["image"] for index in group
                    ],
                }
                for group in valid_dataset.grouping
            ],
        }
        manifest_path = os.path.abspath(args.selection_manifest)
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        temporary = manifest_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, manifest_path)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
    )
    criterion = MPJPE(config).cuda()
    validate(
        config,
        valid_loader,
        valid_dataset,
        model,
        criterion,
        args.output_dir,
        epoch=0,
        is_mmpose=config.DATASET.USE_MMPOSE_VAL,
    )


if __name__ == "__main__":
    main()
