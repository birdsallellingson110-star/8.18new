#!/usr/bin/env python3
"""Train feature-level epipolar input correction in front of frozen H76."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from epipolar_feature_ray_corrector_20260813 import EpipolarFeatureRayCorrector

# Some RUMPL feature switches are consumed while its model module is imported,
# not only when the network instance is constructed.  Establish the audited
# H76 environment before importing the checkpoint loader/models.
os.environ["RUMPL_TRI_ANCHOR"] = "1"
os.environ["RUMPL_TRI_ANCHOR_REG"] = "1e-4"
os.environ["RUMPL_TRI_ANCHOR_CONF_EPS"] = "0.05"
os.environ["RUMPL_PFT_REPEAT_LAST"] = "1"
os.environ["RUMPL_ANCHOR_CENTERED_RAYS"] = "1"
os.environ["RUMPL_INPUT_PLUCKER"] = "1"
os.environ["RUMPL_INPUT_HARMONIC_L"] = "0"
os.environ["RUMPL_GEOMETRY_UNCERTAINTY_TOKEN"] = "0"
os.environ["RUMPL_SEMANTIC_GRAPH_PRE_VFT"] = "off"
os.environ["RUMPL_GRAFORMER_PFT"] = "off"

from export_h76_train_subset_hypotheses_20260811 import load_model
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, action_equal, load_arrays


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
sys.path.insert(0, str(REPO / "lib"))
from core.config import config, update_config  # noqa: E402


COMBINATIONS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--h76-checkpoint", required=True)
    parser.add_argument("--train-cache", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--train-descriptors", nargs="+", required=True)
    parser.add_argument("--validation-descriptors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("feature", "geometry"), default="feature")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-angle-degrees", type=float, default=0.5)
    parser.add_argument("--angle-regularizer", type=float, default=1e-4)
    parser.add_argument("--holdout-modulo", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def configure(cfg_path: str) -> None:
    # Recreate the exact H76 representation used to generate the caches.
    os.environ["RUMPL_TRI_ANCHOR"] = "1"
    os.environ["RUMPL_TRI_ANCHOR_REG"] = "1e-4"
    os.environ["RUMPL_TRI_ANCHOR_CONF_EPS"] = "0.05"
    os.environ["RUMPL_PFT_REPEAT_LAST"] = "1"
    os.environ["RUMPL_ANCHOR_CENTERED_RAYS"] = "1"
    os.environ["RUMPL_INPUT_PLUCKER"] = "1"
    os.environ["RUMPL_INPUT_HARMONIC_L"] = "0"
    os.environ["RUMPL_GEOMETRY_UNCERTAINTY_TOKEN"] = "0"
    os.environ["RUMPL_SEMANTIC_GRAPH_PRE_VFT"] = "off"
    os.environ["RUMPL_GRAFORMER_PFT"] = "off"
    update_config(cfg_path)
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4
    config.DATASET.TEST_VIEWS = [1, 2, 3, 4]
    config.GPUS = "0"
    config.WANDB = False


class DescriptorStore:
    def __init__(self, paths: list[str]) -> None:
        self.arrays: list[np.ndarray] = []
        self.locations: dict[int, tuple[int, int]] = {}
        self.descriptor_dim = -1
        for shard, path_string in enumerate(paths):
            path = Path(path_string)
            array = np.load(path, mmap_mode="r")
            ids = np.load(path.with_suffix(path.suffix + ".group_indices.npy"))
            if array.shape[:4] != (len(ids), 11, 4, 17):
                raise ValueError(f"bad descriptor shape {path}: {array.shape}")
            if self.descriptor_dim not in (-1, array.shape[-1]):
                raise ValueError("descriptor dimensions differ between shards")
            self.descriptor_dim = int(array.shape[-1])
            self.arrays.append(array)
            for row, group_id in enumerate(ids):
                group_id = int(group_id)
                if group_id in self.locations:
                    raise ValueError(f"duplicate descriptor group {group_id}")
                self.locations[group_id] = (shard, row)

    def get(self, group_id: int) -> np.ndarray:
        shard, row = self.locations[int(group_id)]
        return np.asarray(self.arrays[shard][row], dtype=np.float32)


class CorrectorDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        store: DescriptorStore,
        indices: np.ndarray,
    ) -> None:
        self.arrays = arrays
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        missing = [
            int(arrays["group_indices"][row]) for row in self.indices
            if int(arrays["group_indices"][row]) not in store.locations
        ]
        if missing:
            raise ValueError(f"descriptor store misses groups {missing[:8]}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row = int(self.indices[index])
        group_id = int(self.arrays["group_indices"][row])
        return (
            torch.from_numpy(self.arrays["predictions"][row]),
            torch.from_numpy(self.arrays["targets"][row]),
            torch.from_numpy(self.arrays["rays"][row]),
            int(self.arrays["actions"][row]),
            torch.from_numpy(self.store.get(group_id)),
        )


def correct_and_predict(corrector, h76, predictions, rays, descriptors, combo):
    combo_id = COMBINATIONS.index(combo)
    baseline = predictions[:, combo_id]
    subset = rays[:, :, list(combo), :]
    descriptor_subset = descriptors[:, combo_id, list(combo)]
    corrected, diagnostics = corrector(subset, baseline, descriptor_subset)
    pose = h76(corrected, is_training=False)
    return pose, diagnostics


def identity_and_permutation_check(corrector, h76, batch, device):
    predictions, _, rays, _, descriptors = batch
    predictions = predictions[:2].to(device)
    rays = rays[:2].to(device)
    descriptors = descriptors[:2].to(device)
    checks = {}
    ray_checks = {}
    corrector.eval()
    with torch.no_grad():
        for combo in COMBINATIONS:
            combo_id = COMBINATIONS.index(combo)
            subset = rays[:, :, list(combo)]
            corrected, _ = corrector(
                subset, predictions[:, combo_id],
                descriptors[:, combo_id, list(combo)],
            )
            pose = h76(corrected, is_training=False)
            fresh_identity = h76(subset, is_training=False)
            ray_checks[str(combo)] = float((corrected - subset).abs().max())
            checks[str(combo)] = {
                "corrected_vs_fresh_h76_abs_m": float(
                    (pose - fresh_identity).abs().max()
                ),
                "fresh_vs_export_cache_abs_m": float(
                    (fresh_identity - predictions[:, combo_id]).abs().max()
                ),
            }
        combo = COMBINATIONS[-1]
        combo_id = len(COMBINATIONS) - 1
        order = torch.tensor([2, 0, 3, 1], device=device)
        baseline = predictions[:, combo_id]
        corrected_a, _ = corrector(rays, baseline, descriptors[:, combo_id])
        corrected_b, _ = corrector(
            rays[:, :, order], baseline, descriptors[:, combo_id, order]
        )
        pose_a = h76(corrected_a, is_training=False)
        pose_b = h76(corrected_b, is_training=False)
        permutation_error = float((pose_a - pose_b).abs().max())
    maximum_ray = max(ray_checks.values())
    maximum_identity = max(
        item["corrected_vs_fresh_h76_abs_m"] for item in checks.values()
    )
    cache_replay_drift = max(
        item["fresh_vs_export_cache_abs_m"] for item in checks.values()
    )
    if maximum_ray != 0.0 or maximum_identity != 0.0 or permutation_error > 2e-5:
        raise RuntimeError(
            f"identity/permutation failure rays={maximum_ray} "
            f"identity={maximum_identity} permutation={permutation_error}"
        )
    return {
        "max_corrected_ray_abs": maximum_ray,
        "max_corrected_vs_fresh_h76_abs_m": maximum_identity,
        "max_fresh_h76_vs_export_cache_abs_m": cache_replay_drift,
        "view_permutation_max_abs_m": permutation_error,
        "ray_tasks": ray_checks,
        "tasks": checks,
    }


def evaluate(corrector, h76, loader, device):
    corrector.eval()
    errors = {views: [] for views in (2, 3, 4)}
    baseline_errors = {views: [] for views in (2, 3, 4)}
    actions = {views: [] for views in (2, 3, 4)}
    angles = {views: [] for views in (2, 3, 4)}
    with torch.inference_mode():
        for predictions, targets, rays, batch_actions, descriptors in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            descriptors = descriptors.to(device, non_blocking=True)
            action_np = batch_actions.numpy()
            for combo in COMBINATIONS:
                views = len(combo)
                pose, diagnostics = correct_and_predict(
                    corrector, h76, predictions, rays, descriptors, combo
                )
                baseline = predictions[:, COMBINATIONS.index(combo)]
                errors[views].append(
                    (torch.linalg.vector_norm(pose - targets, dim=-1) * 1000)
                    .cpu().numpy()
                )
                baseline_errors[views].append(
                    (torch.linalg.vector_norm(baseline - targets, dim=-1) * 1000)
                    .cpu().numpy()
                )
                actions[views].append(action_np.copy())
                angles[views].append(
                    diagnostics["angle_radians"].cpu().numpy()
                    * 180.0 / math.pi
                )
    result = {}
    for views in (2, 3, 4):
        error = np.concatenate(errors[views])
        baseline = np.concatenate(baseline_errors[views])
        action = np.concatenate(actions[views])
        angle = np.concatenate(angles[views])
        result[f"V{views}"] = {
            "action_equal_all17_mm": action_equal(error, action),
            "baseline_action_equal_all17_mm": action_equal(baseline, action),
            "frame_weighted_all17_mm": float(error.mean()),
            "mean_abs_angle_deg": float(np.abs(angle).mean()),
            "p95_abs_angle_deg": float(np.quantile(np.abs(angle), 0.95)),
        }
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    configure(args.cfg)
    device = torch.device(args.device)
    h76 = load_model(Path(args.h76_checkpoint), device)
    for parameter in h76.parameters():
        parameter.requires_grad_(False)

    train_arrays = load_arrays(args.train_cache)
    with np.load(args.validation_cache) as source:
        validation_arrays = {key: source[key].copy() for key in source.files}
    train_store = DescriptorStore(args.train_descriptors)
    validation_store = DescriptorStore([args.validation_descriptors])
    selected_rows = np.asarray([
        row for row, group_id in enumerate(train_arrays["group_indices"])
        if int(group_id) in train_store.locations
    ], dtype=np.int64)
    holdout = train_arrays["group_indices"][selected_rows] % args.holdout_modulo == 0
    train_rows = selected_rows[~holdout]
    holdout_rows = selected_rows[holdout]
    if args.smoke_batches:
        train_rows = train_rows[: args.batch_size * args.smoke_batches]
        holdout_rows = holdout_rows[: args.batch_size]
    validation_rows = np.asarray([
        row for row, group_id in enumerate(validation_arrays["group_indices"])
        if int(group_id) in validation_store.locations
    ], dtype=np.int64)
    if not args.smoke_batches and len(validation_rows) != len(validation_arrays["targets"]):
        raise ValueError(
            f"validation descriptor coverage {len(validation_rows)}/"
            f"{len(validation_arrays['targets'])}"
        )
    train_loader = DataLoader(
        CorrectorDataset(train_arrays, train_store, train_rows),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, generator=torch.Generator().manual_seed(args.seed),
    )
    holdout_loader = DataLoader(
        CorrectorDataset(train_arrays, train_store, holdout_rows),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        CorrectorDataset(validation_arrays, validation_store, validation_rows),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True,
    )
    corrector = EpipolarFeatureRayCorrector(
        descriptor_dim=train_store.descriptor_dim,
        max_angle_degrees=args.max_angle_degrees,
        use_descriptors=args.variant == "feature",
    ).to(device)
    first_batch = next(iter(holdout_loader))
    identity = identity_and_permutation_check(corrector, h76, first_batch, device)
    print(json.dumps({"identity_check": identity}), flush=True)

    optimizer = torch.optim.AdamW(
        corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pth.tar"
    task_by_views = {
        views: [combo for combo in COMBINATIONS if len(combo) == views]
        for views in (2, 3, 4)
    }
    sampled_views = (2, 2, 2, 3, 3, 4, 4)
    task_rng = random.Random(args.seed)
    history = []
    best_metric = math.inf
    best_epoch = -1
    for epoch in range(args.epochs):
        corrector.train()
        train_errors, train_angles = [], []
        for predictions, targets, rays, _, descriptors in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            descriptors = descriptors.to(device, non_blocking=True)
            views = task_rng.choice(sampled_views)
            combo = task_rng.choice(task_by_views[views])
            optimizer.zero_grad(set_to_none=True)
            pose, diagnostics = correct_and_predict(
                corrector, h76, predictions, rays, descriptors, combo
            )
            mpjpe = torch.linalg.vector_norm(pose - targets, dim=-1).mean()
            normalized_angle = (
                diagnostics["angle_radians"] / corrector.max_angle_radians
            )
            loss = mpjpe + args.angle_regularizer * normalized_angle.square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corrector.parameters(), 5.0)
            optimizer.step()
            train_errors.append(float(mpjpe * 1000.0))
            train_angles.append(float(
                diagnostics["angle_radians"].abs().mean() * 180.0 / math.pi
            ))
        holdout_result = evaluate(
            corrector, h76, holdout_loader, device
        )
        metric = float(np.mean([
            holdout_result[f"V{views}"]["action_equal_all17_mm"]
            for views in (2, 3, 4)
        ]))
        record = {
            "epoch": epoch,
            "train_sampled_mpjpe_mm": float(np.mean(train_errors)),
            "train_mean_abs_angle_deg": float(np.mean(train_angles)),
            "holdout_selection_metric_mm": metric,
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            torch.save({
                "state_dict": corrector.state_dict(),
                "descriptor_dim": train_store.descriptor_dim,
                "variant": args.variant,
                "epoch": epoch,
            }, checkpoint_path)

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    corrector.load_state_dict(best["state_dict"], strict=True)
    final = None if args.smoke_batches else evaluate(
        corrector, h76, validation_loader, device
    )
    result = {
        "method": "feature-level epipolar correspondence -> bounded ray correction -> frozen H76",
        "variant": args.variant,
        "paper_basis": "Epipolar Transformer (CVPR 2020) input correspondence; RUMPL ray backbone preserved",
        "camera_identity": False,
        "world_voxel": False,
        "identity_and_permutation_check": identity,
        "train_groups": len(selected_rows),
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": final,
        "args": vars(args),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"S9_S11_final_once": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
