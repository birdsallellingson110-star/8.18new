#!/usr/bin/env python3
"""Train one E2 utility scorer for all 2-, 3- and 4-view tasks.

This is the V234 completion of the earlier V3/V4-only E2 experiment.  The
candidate generator is frozen; the model only predicts per-joint candidate
utility and performs soft candidate fusion.  Checkpoint selection uses an
internal modulo-10 holdout and the mean of V2/V3/V4 action-equal MPJPE.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import (
    ACTION_NAMES,
    ArrayDataset,
    JOINT_NAMES,
)
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


ORIGINAL_COMBINATIONS = tuple(
    combo
    for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)
PAIRWISE_COMBINATIONS = tuple(itertools.combinations(range(4), 2))
LEARNED_COMBINATIONS = tuple(combo for combo in ORIGINAL_COMBINATIONS if len(combo) >= 3)
EXISTING_COMBINATIONS = ORIGINAL_COMBINATIONS + PAIRWISE_COMBINATIONS + LEARNED_COMBINATIONS
ALL_CANDIDATE_COMBINATIONS = EXISTING_COMBINATIONS + ORIGINAL_COMBINATIONS + ORIGINAL_COMBINATIONS
TASK_COMBINATIONS = ORIGINAL_COMBINATIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attention-depth", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument(
        "--stage-heads", action="store_true",
        help="Use separate utility calibration heads for V2/V3/V4 while sharing the encoder.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.8)
    parser.add_argument("--target-temperature-mm", type=float, default=5.0)
    parser.add_argument("--oracle-weight", type=float, default=1.0)
    parser.add_argument(
        "--task-cardinalities", nargs="+", type=int, choices=(2, 3, 4), default=None,
        help=(
            "Restrict utility training/evaluation to selected view cardinalities. "
            "Default keeps the audited V2/V3/V4 protocol; --task-cardinalities 2 "
            "is the GBT-style two-view-specialist control."
        ),
    )
    parser.add_argument(
        "--identity-hinge", type=float, default=0.0,
        help=(
            "Weight for the identity-preserving soft-fusion hinge "
            "max(0, E_soft-E_baseline), scaled in 1 cm units."
        ),
    )
    parser.add_argument(
        "--identity-v2-weight", type=float, default=1.0,
        help="Extra multiplier for the identity hinge on two-view tasks.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--smoke-validation-batches", type=int, default=0)
    return parser.parse_args()


def load_arrays(paths: list[str], expected_candidates: int) -> dict[str, np.ndarray]:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {
        key: np.concatenate([source[key] for source in loaded], axis=0)
        for key in keys
    }
    order = np.argsort(arrays["group_indices"])
    arrays = {key: value[order] for key, value in arrays.items()}
    if len(np.unique(arrays["group_indices"])) != len(arrays["group_indices"]):
        raise ValueError("duplicate group_indices after concatenation")
    if arrays["predictions"].shape[1:] != (expected_candidates, 17, 3):
        raise ValueError(
            f"bad candidate shape {arrays['predictions'].shape}; expected "
            f"(*,{expected_candidates},17,3)"
        )
    if arrays["targets"].shape[1:] != (17, 3):
        raise ValueError(f"bad target shape {arrays['targets'].shape}")
    return arrays


def task_spec(task_combo: tuple[int, ...], device: torch.device):
    available = [
        index for index, combo in enumerate(ALL_CANDIDATE_COMBINATIONS)
        if set(combo).issubset(task_combo)
    ]
    masks = torch.zeros(len(available), 4, device=device, dtype=torch.float32)
    for row, index in enumerate(available):
        masks[row, list(ALL_CANDIDATE_COMBINATIONS[index])] = 1.0
    task_mask = torch.zeros(4, device=device, dtype=torch.float32)
    task_mask[list(task_combo)] = 1.0
    baseline_global = ORIGINAL_COMBINATIONS.index(task_combo)
    baseline_local = available.index(baseline_global)
    return available, masks, task_mask, baseline_local


def predict_task(model, predictions, targets, rays, task_combo):
    available, masks, task_mask, baseline_local = task_spec(task_combo, predictions.device)
    candidates = predictions[:, available]
    raw = model(candidates, rays, masks, task_mask)
    error = torch.linalg.vector_norm(candidates - targets[:, None], dim=-1)
    true_error = error.permute(0, 2, 1)
    baseline_error = true_error[..., baseline_local:baseline_local + 1]
    return (
        raw - raw[..., baseline_local:baseline_local + 1],
        true_error - baseline_error,
        true_error,
        candidates,
        baseline_local,
    )


def task_loss(
    model, predictions, targets, rays, phase, temperature, target_temperature,
    oracle_weight, identity_hinge=0.0, identity_v2_weight=1.0,
):
    direct_losses = []
    ght_losses = []
    identity_losses = []
    for task_combo in TASK_COMBINATIONS:
        predicted, true_delta, true_error, candidates, baseline_local = predict_task(
            model, predictions, targets, rays, task_combo
        )
        direct_losses.append(training_loss(predicted, true_delta, "balanced_rank"))
        if phase == "ght" or identity_hinge > 0:
            weights = F.softmax(-predicted / temperature, dim=-1)
            expected = (weights * true_error).sum(dim=-1).mean()
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            if identity_hinge > 0:
                baseline_error = true_error[..., baseline_local]
                violation = F.relu(fused_error - baseline_error).mean()
                stage_weight = identity_v2_weight if len(task_combo) == 2 else 1.0
                identity_losses.append(
                    identity_hinge * stage_weight * violation / 0.01
                )
            # Keep the same GHT-style expected-risk scale as the established
            # E2 implementation; target_temperature_mm is an explicit audit
            # parameter retained in the manifest.
            if phase == "ght":
                ght_losses.append(
                    oracle_weight * (expected + 0.05 * fused_error) / 0.01
                    + 0.0 * target_temperature
                )
    loss = torch.stack(direct_losses).mean()
    if ght_losses:
        loss = loss + torch.stack(ght_losses).mean()
    if identity_losses:
        loss = loss + torch.stack(identity_losses).mean()
    return loss


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def evaluate(model, loader, device, temperature, max_batches=0):
    """Evaluate candidate fusion.

    ``temperature`` may be a scalar (the historical protocol) or a mapping
    such as ``{"V2": 0.6, "V3": 1.0, "V4": 1.0}``.  Per-cardinality
    calibration is selected only on the training holdout; it does not change
    the scorer or candidate generator and is useful for the GBT-style
    all-subset protocol where the same utility head sees different numbers of
    views.
    """
    def stage_temperature(stage):
        if isinstance(temperature, dict):
            return float(temperature[stage])
        return float(temperature)

    model.eval()
    stores = {
        f"V{count}": {mode: [] for mode in ("baseline", "hard", "soft", "oracle")}
        for count in (2, 3, 4)
    }
    actions_by_stage = {f"V{count}": [] for count in (2, 3, 4)}
    predicted_values, true_values = [], []
    with torch.inference_mode():
        for batch_index, (predictions, targets, rays, actions) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            batch_actions = actions.numpy()
            for task_combo in TASK_COMBINATIONS:
                stage = f"V{len(task_combo)}"
                predicted, true_delta, true_error, candidates, baseline_local = predict_task(
                    model, predictions, targets, rays, task_combo
                )
                hard_index = predicted.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                weights = F.softmax(
                    -predicted / stage_temperature(stage), dim=-1
                )
                soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                baseline = candidates[:, baseline_local]
                oracle = true_error.min(dim=-1).values
                for mode, pose in (("baseline", baseline), ("hard", hard), ("soft", soft)):
                    stores[stage][mode].append(
                        torch.linalg.vector_norm(pose - targets, dim=-1).cpu().numpy() * 1000.0
                    )
                stores[stage]["oracle"].append(oracle.cpu().numpy() * 1000.0)
                actions_by_stage[stage].append(batch_actions.copy())
                predicted_values.append(predicted.cpu().numpy().reshape(-1))
                true_values.append((true_delta.cpu().numpy() / 0.01).reshape(-1))
    result = {}
    for stage in ("V2", "V3", "V4"):
        if not stores[stage]["baseline"]:
            result[stage] = {"disabled": True}
            continue
        stage_actions = np.concatenate(actions_by_stage[stage])
        result[stage] = {}
        for mode, chunks in stores[stage].items():
            values = np.concatenate(chunks, axis=0)
            result[stage][mode] = {
                "action_equal_all17_mm": action_equal(values, stage_actions),
                "frame_weighted_all17_mm": float(values.mean()),
            }
            if mode in ("baseline", "hard", "soft"):
                result[stage][mode]["per_joint_mm"] = {
                    name: action_equal(values[:, index], stage_actions)
                    for index, name in enumerate(JOINT_NAMES)
                }
    if predicted_values:
        predicted_all = np.concatenate(predicted_values)
        true_all = np.concatenate(true_values)
        finite = np.isfinite(predicted_all) & np.isfinite(true_all)
        result["delta_pearson"] = float(np.corrcoef(predicted_all[finite], true_all[finite])[0, 1])
    result["num_batches"] = int(len(next(iter(stores["V2"].values()))))
    return result


def main() -> None:
    global TASK_COMBINATIONS
    args = parse_args()
    if args.task_cardinalities is not None:
        TASK_COMBINATIONS = tuple(
            combo for combo in ORIGINAL_COMBINATIONS
            if len(combo) in set(args.task_cardinalities)
        )
        if not TASK_COMBINATIONS:
            raise ValueError("--task-cardinalities selected no task combinations")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    expected = len(ALL_CANDIDATE_COMBINATIONS)
    train = load_arrays(args.train_shards, expected)
    validation = load_arrays([args.validation_cache], expected)
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if args.smoke_batches:
        train_indices = train_indices[: args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[: args.batch_size]
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = SetTransformerJointUtility(
        mean, std, args.attention_depth, stage_heads=args.stage_heads
    ).to(device)
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=args.workers,
        pin_memory=True,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    validation_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pth.tar"
    phases = ([('direct', 5e-4)] * args.pretrain_epochs
              + [('ght', 1e-4)] * args.finetune_epochs)
    best_metric, best_epoch, history = math.inf, -1, []
    optimizer = None
    previous_phase = None
    for epoch, (phase, lr) in enumerate(phases):
        if phase != previous_phase:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            previous_phase = phase
        model.train()
        losses = []
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(
                model, predictions, targets, rays, phase, args.temperature,
                args.target_temperature_mm / 1000.0, args.oracle_weight,
                args.identity_hinge, args.identity_v2_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate(model, holdout_loader, device, args.temperature)
        active_counts = sorted({len(combo) for combo in TASK_COMBINATIONS})
        metric = float(np.mean([
            holdout_result[f"V{count}"]["soft"]["action_equal_all17_mm"]
            for count in active_counts
        ]))
        row = {
            "epoch": epoch, "phase": phase, "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": metric, "holdout": holdout_result,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth, "epoch": epoch,
                "phase": phase, "candidate_count": expected,
                "stage_heads": args.stage_heads,
                "candidate_combinations": [list(item) for item in ALL_CANDIDATE_COMBINATIONS],
                "temperature": args.temperature,
            }, checkpoint_path)
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = None if args.smoke_batches else evaluate(
        model, validation_loader, device, args.temperature, args.smoke_validation_batches
    )
    payload = {
        "method": "E2 universal Set Transformer utility fusion",
        "active_task_cardinalities": sorted({len(combo) for combo in TASK_COMBINATIONS}),
        "stage_heads": args.stage_heads,
        "paper_basis": [
            "GHT-style stochastic hypothesis scoring (CVPR 2022)",
            "confidence-weighted/IRLS triangulation controls",
        ],
        "candidate_count": expected,
        "candidate_combinations": [list(item) for item in ALL_CANDIDATE_COMBINATIONS],
        "task_combinations": [list(item) for item in TASK_COMBINATIONS],
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": test_result, "args": vars(args),
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"S9_S11_final_once": test_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
