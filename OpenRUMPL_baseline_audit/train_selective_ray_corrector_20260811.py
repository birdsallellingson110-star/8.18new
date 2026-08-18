#!/usr/bin/env python3
"""Train Stage-D selective ray correction while H76 and C4 stay frozen."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from export_h76_train_subset_hypotheses_20260811 import load_model
from selective_ray_corrector_20260811 import (
    SelectiveRayCorrector,
    counterfactual_harm_gate,
)
from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    ArrayDataset,
    COMBINATIONS,
    JOINT_NAMES,
    JointUtilityScorer,
    action_equal,
    load_arrays,
    task_spec,
)


REPO = Path(__file__).resolve().parents[1] / "OpenRUMPL" / "RUMPL"
if str(REPO / "lib") not in sys.path:
    sys.path.insert(0, str(REPO / "lib"))
from core.config import config, update_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--h76-checkpoint", required=True)
    parser.add_argument("--c4-checkpoint", default="")
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--variant", choices=("geometry", "utility"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--angle-regularizer", type=float, default=1e-4)
    parser.add_argument("--max-angle-degrees", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--holdout-modulo", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def configure(cfg_path: str, workers: int):
    update_config(cfg_path)
    config.DATASET.N_VIEWS_TRAIN_TEST_ALL = 4
    config.DATASET.TEST_VIEWS = [1, 2, 3, 4]
    config.GPUS = "0"
    config.WORKERS = workers
    config.WANDB = False


def load_c4(path: str, device: torch.device) -> JointUtilityScorer:
    if not path:
        raise ValueError("utility variant requires --c4-checkpoint")
    state = torch.load(path, map_location="cpu")
    model = JointUtilityScorer(state["mean"], state["std"])
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def task_gate(c4, predictions, rays, combo, use_utility):
    if not use_utility:
        return torch.ones(
            predictions.shape[0], 17, len(combo),
            device=predictions.device, dtype=predictions.dtype,
        )
    return counterfactual_harm_gate(
        c4, predictions, rays, combo, COMBINATIONS, task_spec
    )


def correct_and_predict(
    corrector, h76, c4, predictions, rays, combo, use_utility
):
    candidate_index = COMBINATIONS.index(combo)
    baseline = predictions[:, candidate_index]
    gate = task_gate(c4, predictions, rays, combo, use_utility)
    subset = rays[:, :, list(combo), :]
    corrected, diagnostics = corrector(subset, baseline, gate, use_utility)
    pose = h76(corrected, is_training=False)
    return pose, diagnostics


def identity_check(corrector, h76, predictions, rays, use_utility, c4):
    corrector.eval()
    checks = {}
    with torch.no_grad():
        for combo in COMBINATIONS:
            expected = predictions[:, COMBINATIONS.index(combo)]
            actual, diagnostics = correct_and_predict(
                corrector, h76, c4, predictions, rays, combo, use_utility
            )
            checks[str(combo)] = {
                "cached_pose_max_abs_m": float((actual - expected).abs().max()),
                "correction_max_angle_deg": float(
                    diagnostics["angle_radians"].abs().max() * 180.0 / math.pi
                ),
            }
    max_pose = max(item["cached_pose_max_abs_m"] for item in checks.values())
    max_angle = max(item["correction_max_angle_deg"] for item in checks.values())
    if max_pose > 2e-5 or max_angle != 0.0:
        raise RuntimeError(
            f"identity initialization failed: pose={max_pose}, angle={max_angle}, "
            f"tasks={checks}"
        )
    return {"max_pose_abs_m": max_pose, "max_angle_deg": max_angle, "tasks": checks}


def evaluate(corrector, h76, c4, loader, device, use_utility):
    corrector.eval()
    stores = {views: [] for views in (2, 3, 4)}
    baseline_stores = {views: [] for views in (2, 3, 4)}
    actions = {views: [] for views in (2, 3, 4)}
    angle_chunks = {views: [] for views in (2, 3, 4)}
    gate_chunks = {views: [] for views in (2, 3, 4)}
    with torch.inference_mode():
        for predictions, targets, rays, batch_actions in loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            batch_actions_np = batch_actions.numpy()
            for combo in COMBINATIONS:
                views = len(combo)
                pose, diagnostics = correct_and_predict(
                    corrector, h76, c4, predictions, rays, combo, use_utility
                )
                baseline = predictions[:, COMBINATIONS.index(combo)]
                stores[views].append(
                    (torch.linalg.vector_norm(pose - targets, dim=-1) * 1000.0)
                    .cpu().numpy()
                )
                baseline_stores[views].append(
                    (torch.linalg.vector_norm(baseline - targets, dim=-1) * 1000.0)
                    .cpu().numpy()
                )
                actions[views].append(batch_actions_np.copy())
                angle_chunks[views].append(
                    diagnostics["angle_radians"].abs().cpu().numpy()
                    * 180.0 / math.pi
                )
                gate_chunks[views].append(
                    diagnostics["harm_gate"].cpu().numpy()
                )
    result = {}
    for views in (2, 3, 4):
        error = np.concatenate(stores[views])
        baseline_error = np.concatenate(baseline_stores[views])
        stage_actions = np.concatenate(actions[views])
        angle = np.concatenate(angle_chunks[views]).reshape(-1)
        gate = np.concatenate(gate_chunks[views]).reshape(-1)
        result[f"V{views}"] = {
            "action_equal_all17_mm": action_equal(error, stage_actions),
            "baseline_action_equal_all17_mm": action_equal(
                baseline_error, stage_actions
            ),
            "frame_weighted_all17_mm": float(error.mean()),
            "mean_abs_angle_deg": float(angle.mean()),
            "p95_abs_angle_deg": float(np.quantile(angle, 0.95)),
            "mean_gate": float(gate.mean()),
            "gate_positive_fraction": float((gate > 0).mean()),
            "per_joint_mm": {
                name: action_equal(error[:, index], stage_actions)
                for index, name in enumerate(JOINT_NAMES)
            },
            "per_action_mm": {
                name: float(error[stage_actions == code].mean())
                for code, name in ACTION_NAMES.items()
                if np.any(stage_actions == code)
            },
        }
    return result


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    configure(args.cfg, args.workers)
    h76 = load_model(Path(args.h76_checkpoint).resolve(), device)
    for parameter in h76.parameters():
        parameter.requires_grad_(False)
    use_utility = args.variant == "utility"
    c4 = load_c4(args.c4_checkpoint, device) if use_utility else None
    corrector = SelectiveRayCorrector(args.max_angle_degrees).to(device)

    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % args.holdout_modulo == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if args.smoke_batches:
        train_indices = train_indices[: args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[: min(len(holdout_indices), args.batch_size)]
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    test_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    first = next(iter(holdout_loader))
    identity = identity_check(
        corrector, h76,
        first[0].to(device), first[2].to(device), use_utility, c4,
    )
    print(json.dumps({"identity_check": identity}), flush=True)

    optimizer = torch.optim.AdamW(
        corrector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pth.tar"
    history = []
    best_metric = math.inf
    best_epoch = -1
    task_by_views = {
        views: [combo for combo in COMBINATIONS if len(combo) == views]
        for views in (2, 3, 4)
    }
    # Deliberately give two-view batches modestly more exposure while retaining
    # balanced pressure on the V3/V4 paper targets.
    sampled_view_counts = (2, 2, 2, 3, 3, 4, 4)
    task_rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        corrector.train()
        losses = []
        mpjpes = []
        angles = []
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            views = task_rng.choice(sampled_view_counts)
            combo = task_rng.choice(task_by_views[views])
            optimizer.zero_grad(set_to_none=True)
            pose, diagnostics = correct_and_predict(
                corrector, h76, c4, predictions, rays, combo, use_utility
            )
            mpjpe = torch.linalg.vector_norm(pose - targets, dim=-1).mean()
            normalized_angle = (
                diagnostics["angle_radians"] / corrector.max_angle_radians
            )
            angle_penalty = normalized_angle.square().mean()
            loss = mpjpe + args.angle_regularizer * angle_penalty
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                corrector.parameters(), 5.0
            )
            if not torch.isfinite(grad_norm):
                raise RuntimeError("non-finite ray-corrector gradient")
            optimizer.step()
            losses.append(float(loss.item()))
            mpjpes.append(float(mpjpe.item() * 1000.0))
            angles.append(float(
                diagnostics["angle_radians"].abs().mean().item()
                * 180.0 / math.pi
            ))
        holdout_result = evaluate(
            corrector, h76, c4, holdout_loader, device, use_utility
        )
        metric = np.mean([
            holdout_result[f"V{views}"]["action_equal_all17_mm"]
            for views in (2, 3, 4)
        ])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_sampled_mpjpe_mm": float(np.mean(mpjpes)),
            "train_mean_abs_angle_deg": float(np.mean(angles)),
            "holdout_selection_metric_mm": float(metric),
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = float(metric)
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": corrector.state_dict(),
                    "variant": args.variant,
                    "epoch": epoch,
                    "max_angle_degrees": args.max_angle_degrees,
                },
                checkpoint_path,
            )

    best = torch.load(checkpoint_path, map_location=device)
    corrector.load_state_dict(best["state_dict"], strict=True)
    test_result = None if args.smoke_batches else evaluate(
        corrector, h76, c4, test_loader, device, use_utility
    )
    payload = {
        "variant": args.variant,
        "method": (
            "identity-initialized bounded tangent-plane ray correction"
            + (" gated by frozen C4 counterfactual negative-view utility"
               if use_utility else " with geometry-only features")
        ),
        "h76_checkpoint": str(Path(args.h76_checkpoint).resolve()),
        "c4_checkpoint": (
            str(Path(args.c4_checkpoint).resolve()) if use_utility else None
        ),
        "identity_check": identity,
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": test_result,
        "args": vars(args),
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
