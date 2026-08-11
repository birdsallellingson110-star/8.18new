#!/usr/bin/env python3
"""Evaluate a temporal H76 extension on dense stride-5 Human3.6M.

Each 9-frame causal window predicts its latest frame.  Camera combinations are
kept as separate sequences and the reported primary metric is the arithmetic
mean of the 15 action MPJPE values, matching the project's single-frame table.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "lib"))

import dataset
from core.config import config, update_config
from dataset.temporal_h36m_rumpl import TemporalH36MRUMPL, collate_temporal_h36m
from models.multiview_rumpl import get_multiview_rumpl_net
from models.temporal_gbt_rumpl import TemporalJointViewRUMPL


ACTION_NAMES = {
    2: "Direction", 3: "Discuss", 4: "Eating", 5: "Greet", 6: "Phone",
    7: "Photo", 8: "Pose", 9: "Purchase", 10: "Sitting",
    11: "SittingDown", 12: "Smoke", 13: "Wait", 14: "WalkDog",
    15: "Walk", 16: "WalkTwo",
}
KP_STAR = [11, 14, 12, 15, 13, 16, 5, 2, 6, 3]


def clean_state_dict(payload):
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in payload.items()
    }


def set_backbone_environment():
    os.environ["RUMPL_TRI_ANCHOR"] = "1"
    os.environ["RUMPL_TRI_ANCHOR_REG"] = "1e-4"
    os.environ["RUMPL_TRI_ANCHOR_CONF_EPS"] = "0.05"
    os.environ["RUMPL_ANCHOR_CENTERED_RAYS"] = "1"
    os.environ["RUMPL_ANCHOR_CENTER_PER_JOINT"] = "0"
    os.environ["RUMPL_INPUT_PLUCKER"] = "1"
    os.environ["RUMPL_INPUT_HARMONIC_L"] = "0"
    if os.environ.get("RUMPL_BACKBONE_FLAVOR", "h76").lower() == "h81":
        os.environ.setdefault("RUMPL_PER_JOINT_RESIDUAL_GATE", "1")
        for name in (
            "GBT_GLOBAL_JV_DEPTH", "GBT_LEARNABLE_BIAS", "RUMPL_GBT_SET_DECODER",
            "RUMPL_RELATIVE_VIEW_FUSION", "RUMPL_GEOMETRY_UNCERTAINTY_TOKEN",
            "RUMPL_POST_PFT_GRAPH_RESIDUAL", "RUMPL_JOINT_SPECIFIC_HEAD",
        ):
            os.environ.setdefault(name, "0")
    else:
        for name in (
            "GBT_GLOBAL_JV_DEPTH", "GBT_LEARNABLE_BIAS", "RUMPL_GBT_SET_DECODER",
            "RUMPL_RELATIVE_VIEW_FUSION", "RUMPL_GEOMETRY_UNCERTAINTY_TOKEN",
            "RUMPL_PER_JOINT_RESIDUAL_GATE", "RUMPL_POST_PFT_GRAPH_RESIDUAL",
            "RUMPL_JOINT_SPECIFIC_HEAD",
        ):
            os.environ[name] = "0"


def set_h76_environment():
    os.environ["RUMPL_BACKBONE_FLAVOR"] = "h76"
    set_backbone_environment()


class _FrameCacheBuilder(Dataset):
    """Worker-side extraction of only the tensors used by temporal eval."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base.grouping)

    def __getitem__(self, index):
        sample = self.base[index]
        # The temporal model consumes rays and the 3-D target.  Keeping the
        # other frame-level fields out of the cache avoids serializing the
        # large per-image metadata dictionaries; it cannot alter inference.
        return index, sample[2], sample[3]


class _CachedFrameDataset:
    """Frame-level dataset facade backed by a read-only tensor cache."""

    def __init__(self, base_dataset, targets, rays):
        self.grouping = base_dataset.grouping
        self.db = base_dataset.db
        self.max_random_n_views = None
        self.targets = targets
        self.rays = rays

    def __len__(self):
        return int(self.targets.shape[0])

    def __getitem__(self, index):
        target = self.targets[index]
        rays = self.rays[index]
        # These fields are retained as empty tensors only because the shared
        # temporal collate contract expects them.  They are never read by the
        # evaluator or the model.
        return (
            torch.empty(0), torch.empty(0), target, rays, {}, torch.empty(0)
        )


def _load_or_build_frame_cache(base_dataset, cache_path, workers):
    """Load or construct a deterministic rays/GT cache for one view count."""

    cache_path = Path(cache_path)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        targets, rays = payload["targets"], payload["rays"]
        if int(targets.shape[0]) != len(base_dataset.grouping):
            raise RuntimeError(
                f"cache length mismatch: {cache_path} has {targets.shape[0]}, "
                f"dataset has {len(base_dataset.grouping)}"
            )
        return _CachedFrameDataset(base_dataset, targets, rays)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    builder = _FrameCacheBuilder(base_dataset)
    loader = DataLoader(
        builder,
        batch_size=64,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
        persistent_workers=workers > 0,
    )
    targets = None
    rays = None
    for indices, batch_targets, batch_rays in loader:
        if targets is None:
            targets = torch.empty(
                (len(base_dataset.grouping),) + tuple(batch_targets.shape[1:]),
                dtype=batch_targets.dtype,
            )
            rays = torch.empty(
                (len(base_dataset.grouping),) + tuple(batch_rays.shape[1:]),
                dtype=batch_rays.dtype,
            )
        targets[indices] = batch_targets
        rays[indices] = batch_rays
    if targets is None or rays is None:
        raise RuntimeError("cannot build an empty H36M frame cache")
    torch.save({"targets": targets, "rays": rays}, cache_path)
    return _CachedFrameDataset(base_dataset, targets, rays)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mmpose-type", required=True)
    parser.add_argument("--dataset-name", default="annot_temporal_5_5")
    parser.add_argument("--num-views", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--window-length", type=int, default=9)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--output-frame",
        choices=("latest", "center"),
        default="latest",
        help=(
            "frame scored from each temporal window; center matches MixSTE's "
            "bidirectional offline setting, latest is causal"
        ),
    )
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--biased", action="store_true")
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument(
        "--fusion-mode",
        choices=(
            "global-residual", "query-residual", "mixste-ttb",
            "mixste-ttb-residual",
            "mixste-alternating",
            "mixste-pose-residual",
        ),
        default="global-residual",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--frame-cache",
        help="optional CPU cache for frame-level rays/GT; built once per view count",
    )
    parser.add_argument(
        "--cache-workers",
        type=int,
        default=0,
        help="workers used only when constructing --frame-cache",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backbone-flavor",
        choices=("h76", "h81"),
        default="h76",
        help="Match single-frame checkpoint (H81 enables per-joint gate).",
    )
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Evaluate frozen single-frame RUMPL on each window's latest frame (no temporal branch).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["RUMPL_BACKBONE_FLAVOR"] = args.backbone_flavor
    set_backbone_environment()
    update_config(args.cfg)
    # Same protocol as single-frame Table-II (e.g. launch_H59): all C(4,k) camera
    # subsets with k=--num-views.  Ignore RUMPL_N_VIEWS_TRAIN_TEST_ALL=4 from training.
    dataset_num_views = args.num_views
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = dataset_num_views
    config.DATASET.TEST_H36M_DATASET_NAME = args.dataset_name
    config.DATASET.TEST_MMPOSE_TYPE = args.mmpose_type
    config.DATASET.USE_MMPOSE_VAL = True
    config.DATASET.FLIP_LOWER_BODY_KP_TEST = True
    config.DATASET.TEST_ON_ALL_CAMERAS = True
    config.DATASET.TEST_VIEWS = list(range(1, args.num_views + 1))
    config.WANDB = False

    device = torch.device(args.device)
    base = get_multiview_rumpl_net(config, is_train=False)
    base_state = clean_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
    from utils.rumpl_checkpoint_adapt import merge_pretrained_into_model_state

    merged, skipped = merge_pretrained_into_model_state(
        base.state_dict(), base_state, strict_shapes=False
    )
    if skipped:
        print(f"[temporal-eval] base load skipped keys: {skipped[:20]}")
    base.load_state_dict(merged, strict=False)
    if args.backbone_only:
        model = base.to(device).eval()
    else:
        model = TemporalJointViewRUMPL(
            base, depth=args.depth, num_heads=args.heads, biased=args.biased,
            token_dropout=0.0, residual_gate=True,
            residual_scale=args.residual_scale,
            fusion_mode=args.fusion_mode,
            temporal_length=args.window_length,
        )
        if args.temporal_checkpoint:
            payload = torch.load(args.temporal_checkpoint, map_location="cpu")
            temporal_state = payload["model"]
            model_state = model.state_dict()
            # ``weighted_mean`` is a legacy optional view-count-specific
            # Conv1d parameter. H76 has linear weighted fusion disabled, so it
            # is never executed, yet its shape follows the requested V2/V3/V4
            # dataset cardinality and can prevent loading a V2-trained temporal
            # checkpoint. Skip only this known dead parameter; retain strict
            # validation for every active backbone and temporal key.
            skipped_temporal = []
            for key in list(temporal_state):
                if (
                    key == "backbone.weighted_mean.weight"
                    and key in model_state
                    and temporal_state[key].shape != model_state[key].shape
                ):
                    skipped_temporal.append(
                        (key, tuple(temporal_state[key].shape), tuple(model_state[key].shape))
                    )
                    temporal_state = dict(temporal_state)
                    temporal_state.pop(key)
                    break
            incompatible = model.load_state_dict(temporal_state, strict=False)
            expected_missing = {item[0] for item in skipped_temporal}
            if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "temporal checkpoint mismatch: "
                    f"missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            if skipped_temporal:
                print(
                    f"[temporal-eval] skipped inactive cardinality parameter: {skipped_temporal}",
                    flush=True,
                )
        else:
            if hasattr(model, "global_gate"):
                model.global_gate.data.zero_()
            if hasattr(model, "query_residual_head"):
                model.query_residual_head.weight.data.zero_()
                model.query_residual_head.bias.data.zero_()
        model = model.to(device).eval()

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    base_dataset = getattr(dataset, config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([transforms.ToTensor(), normalize]),
    )
    if args.frame_cache:
        base_dataset = _load_or_build_frame_cache(
            base_dataset, args.frame_cache, args.cache_workers
        )
    temporal_dataset = TemporalH36MRUMPL(
        base_dataset, window_length=args.window_length,
        frame_stride=args.frame_stride, window_step=1,
    )
    loader = DataLoader(
        temporal_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, collate_fn=collate_temporal_h36m,
    )
    print(
        f"[temporal-eval] num_views={args.num_views} "
        f"N_VIEWS_TRAIN_TEST_ALL={config.DATASET.N_VIEWS_TRAIN_TEST_ALL} "
        f"temporal_windows={len(temporal_dataset)} loader_batches={len(loader)}",
        flush=True,
    )

    predictions, targets, actions, fnames = [], [], [], []
    output_index = -1 if args.output_frame == "latest" else args.window_length // 2
    with torch.inference_mode():
        for batch_index, (_, _, target, rays, metadata, _) in enumerate(loader):
            rays = rays.to(device, non_blocking=True)
            if args.backbone_only:
                selected_rays = rays[:, output_index].clone()
                output = model(selected_rays, is_training=False)
            else:
                # Dataset already carries k views; do not random-resample at eval.
                output, _ = model(rays)
            predictions.append(
                output[:, output_index].cpu().numpy()
                if not args.backbone_only else output.cpu().numpy()
            )
            targets.append(target[:, output_index].numpy())
            for item in metadata:
                subject, action, subaction, cameras = item["sequence_key"]
                frame = item["frame_ids"][output_index]
                actions.append(int(action))
                camera_text = "-".join(str(value + 1) for value in cameras)
                fnames.append(
                    f"sub_{subject}_act_{action}_subact_{subaction}_"
                    f"frame_{frame}_cams_{camera_text}"
                )
            if (batch_index + 1) % 250 == 0:
                print(f"batches={batch_index + 1}/{len(loader)}", flush=True)

    pred = np.concatenate(predictions)
    gt = np.concatenate(targets)
    actions_np = np.asarray(actions)
    error = np.linalg.norm(pred - gt, axis=-1) * 1000.0
    per_action = {}
    for action, name in ACTION_NAMES.items():
        selected = actions_np == action
        if not selected.any():
            raise RuntimeError(f"no evaluation windows for {name}")
        per_action[name] = {
            "windows": int(selected.sum()),
            "all17_mm": float(error[selected].mean()),
            "kp_star_mm": float(error[selected][:, KP_STAR].mean()),
        }
    result = {
        "protocol": (
            f"{'causal' if args.output_frame == 'latest' else 'bidirectional'}_"
            f"{args.output_frame}_frame_stride5_action_equal"
        ),
        "num_views": args.num_views,
        "window_length": args.window_length,
        "windows": int(len(pred)),
        "temporal_checkpoint": args.temporal_checkpoint,
        "frame_weighted": {
            "all17_mm": float(error.mean()),
            "kp_star_mm": float(error[:, KP_STAR].mean()),
        },
        "table2_action_equal": {
            "all17_mm": float(np.mean([x["all17_mm"] for x in per_action.values()])),
            "kp_star_mm": float(np.mean([x["kp_star_mm"] for x in per_action.values()])),
        },
        "per_action": per_action,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "preds_gt_temporal_dict.pkl").open("wb") as handle:
        pickle.dump({"pred": pred, "gt": gt, "fnames": fnames}, handle)
    with (output_dir / "table2.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
