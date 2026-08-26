#!/usr/bin/env python3
"""Short T=9 temporal residual screen on the frozen E2-C2 candidate pool.

This is deliberately a narrow experiment.  The 2-D detector, H76 RUMPL
checkpoint, 22 candidate poses and E2 Set-Transformer are frozen.  The only
learned component is a zero-initialized temporal residual on the candidate
utility logits.  At initialization it is exactly the calibrated E2-C2
baseline, so a negative result cannot be hidden by changing the candidate
generator or the single-frame model.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_h76_counterfactual_delta_20260811 import training_loss
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, JOINT_NAMES
from train_e2_v234_universal_20260812 import ORIGINAL_COMBINATIONS


TASKS = ORIGINAL_COMBINATIONS
ALL_CANDIDATES = ORIGINAL_COMBINATIONS + ORIGINAL_COMBINATIONS
FRAME_STRIDE = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--train-scores", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--validation-scores", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window-length", type=int, default=9)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--direct-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ght-lr", type=float, default=5e-5)
    p.add_argument("--temperature-v2", type=float, default=0.4)
    p.add_argument("--temperature-v3", type=float, default=1.8)
    p.add_argument("--temperature-v4", type=float, default=1.8)
    p.add_argument("--identity-weight", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-holdout-windows", type=int, default=0)
    return p.parse_args()


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def metadata_from_pkl(path: str, expected_groups: int) -> dict[str, np.ndarray]:
    with open(path, "rb") as handle:
        records = pickle.load(handle)
    groups: collections.OrderedDict[tuple[int, int, int, int], set[int]] = (
        collections.OrderedDict()
    )
    for record in records:
        key = (
            int(record["subject"]), int(record["action"]),
            int(record["subaction"]), int(record["image_id"]),
        )
        groups.setdefault(key, set()).add(int(record["camera_id"]))
    if len(groups) != expected_groups:
        raise RuntimeError(
            f"{path} has {len(groups)} groups but cache has {expected_groups}"
        )
    bad = [(key, sorted(cams)) for key, cams in groups.items()
           if cams != {0, 1, 2, 3}]
    if bad:
        raise RuntimeError(f"incomplete four-view groups: {bad[:3]}")
    keys = list(groups)
    sequence_ids: list[int] = []
    seq_map: dict[tuple[int, int, int], int] = {}
    for subject, action, subaction, _ in keys:
        seq = (subject, action, subaction)
        seq_map.setdefault(seq, len(seq_map))
        sequence_ids.append(seq_map[seq])
    return {
        "subjects": np.asarray([x[0] for x in keys], dtype=np.int16),
        "actions": np.asarray([x[1] for x in keys], dtype=np.int16),
        "frame_ids": np.asarray([x[3] for x in keys], dtype=np.int64),
        "sequence_ids": np.asarray(sequence_ids, dtype=np.int32),
    }


def build_windows(meta: dict[str, np.ndarray], length: int) -> np.ndarray:
    if length == 1:
        return np.arange(len(meta["frame_ids"]), dtype=np.int64)[:, None]
    if length % 2 != 1 or length < 3:
        raise ValueError("window length must be an odd integer >= 1")
    by_sequence: dict[int, list[int]] = collections.defaultdict(list)
    for index, sequence in enumerate(meta["sequence_ids"].tolist()):
        by_sequence[int(sequence)].append(index)
    rows: list[list[int]] = []
    for indices in by_sequence.values():
        indices.sort(key=lambda i: int(meta["frame_ids"][i]))
        for start in range(0, len(indices) - length + 1):
            row = indices[start:start + length]
            if np.all(np.diff(meta["frame_ids"][row]) == FRAME_STRIDE):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no T={length} windows")
    return np.asarray(rows, dtype=np.int64)


def load_cache(path: str) -> dict[str, np.ndarray]:
    source = np.load(path, allow_pickle=False)
    required = {"predictions", "targets", "rays", "actions", "subjects",
                "group_indices"}
    missing = required.difference(source.files)
    if missing:
        raise ValueError(f"{path} missing {sorted(missing)}")
    arrays = {key: source[key] for key in required}
    if arrays["predictions"].shape[1:] != (22, 17, 3):
        raise ValueError(f"bad candidate shape {arrays['predictions'].shape}")
    return arrays


class TemporalResidualUtility(nn.Module):
    """Permutation-shared temporal encoder, initialized to the identity."""

    def __init__(self, feature_dim: int, window_length: int):
        super().__init__()
        self.window_length = window_length
        self.input = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.position = nn.Parameter(torch.zeros(window_length, 64))
        layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.0, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 1))
        # Exact identity at initialization: correction == 0 and E2 is
        # reproduced before the first optimizer step.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # B,T,C,J,F -> B,J,C scores for the center frame.
        batch, time, candidates, joints, feature_dim = features.shape
        if time != self.window_length:
            raise ValueError(f"expected T={self.window_length}, got {time}")
        tokens = features.permute(0, 2, 3, 1, 4).reshape(
            batch * candidates * joints, time, feature_dim
        )
        tokens = self.input(tokens) + self.position[None]
        # PyTorch's fused SDPA kernel has a grid-size limit for the number of
        # independent sequences.  Candidate/joint tokens from a 256-window
        # batch exceed it even though memory is sufficient; chunking is exact
        # because these sequences never attend to one another.
        encoded_chunks = []
        for start in range(0, len(tokens), 32768):
            encoded_chunks.append(self.encoder(tokens[start:start + 32768]))
        encoded = torch.cat(encoded_chunks, dim=0)
        center = encoded[:, time // 2]
        return self.output(center).reshape(batch, candidates, joints).permute(0, 2, 1)


def task_spec(task_index: int):
    task = TASKS[task_index]
    available = [
        i for i, combo in enumerate(ALL_CANDIDATES)
        if set(combo).issubset(task)
    ]
    masks = np.zeros((len(available), 4), dtype=np.float32)
    for row, index in enumerate(available):
        masks[row, list(ALL_CANDIDATES[index])] = 1.0
    baseline_local = available.index(task_index)
    return task, np.asarray(available, dtype=np.int64), masks, baseline_local


def task_temperature(task: tuple[int, ...], args: argparse.Namespace) -> float:
    return {
        2: args.temperature_v2, 3: args.temperature_v3,
        4: args.temperature_v4,
    }[len(task)]


def make_features(
    predictions: torch.Tensor, rays: torch.Tensor, scores: torch.Tensor,
    available: np.ndarray, masks: torch.Tensor, center: int,
) -> torch.Tensor:
    """Build label-free B,T,C,J,F temporal features."""
    candidate = predictions[:, :, available]  # B,T,C,J,3
    root_relative = candidate - candidate[..., :1, :]
    center_root = root_relative[:, center:center + 1]
    displacement = (root_relative - center_root) / 0.1
    displacement_norm = torch.linalg.vector_norm(
        displacement, dim=-1, keepdim=True
    )
    confidence = rays[..., 6]  # B,T,J,V
    mask = masks.to(device=predictions.device, dtype=predictions.dtype)
    confidence = (
        confidence[:, :, None, :, :] * mask[None, None, :, None, :]
    ).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)[None, None, :, None]
    confidence = confidence.unsqueeze(-1)
    # Scores are indexed B,T,J,C; make them B,T,C,J,1.  Dividing by five
    # keeps the residual encoder well-conditioned without changing E2.
    score = scores[:, :, :, available].permute(0, 1, 3, 2).unsqueeze(-1) / 5.0
    score_delta = score - score[:, center:center + 1]
    return torch.cat(
        (root_relative, displacement, displacement_norm, confidence,
         score, score_delta), dim=-1
    )


def forward_task(
    model: TemporalResidualUtility | None,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rays: torch.Tensor,
    base_scores: torch.Tensor,
    task_index: int,
    center: int,
    args: argparse.Namespace,
):
    task, available, masks_np, baseline_local = task_spec(task_index)
    masks = torch.from_numpy(masks_np)
    features = make_features(
        predictions, rays, base_scores[:, :, task_index], available, masks, center
    )
    correction = (
        model(features) if model is not None else
        torch.zeros(
            predictions.shape[0], 17, len(available),
            device=predictions.device, dtype=predictions.dtype,
        )
    )
    center_scores = base_scores[:, center, task_index, :, available]
    center_scores = center_scores.permute(0, 2, 1)  # B,C,J
    total_scores = center_scores + correction.permute(0, 2, 1)
    predicted_delta = (
        total_scores - total_scores[:, baseline_local:baseline_local + 1]
    ).permute(0, 2, 1)  # B,J,C
    candidate_center = predictions[:, center, available]
    errors = torch.linalg.vector_norm(
        candidate_center - targets[:, None], dim=-1
    ).permute(0, 2, 1)  # B,J,C
    baseline_error = errors[..., baseline_local:baseline_local + 1]
    true_delta = errors - baseline_error
    weights = F.softmax(
        -predicted_delta / task_temperature(task, args), dim=-1
    )
    fused = torch.einsum("bjc,bcjd->bjd", weights, candidate_center)
    fused_error = torch.linalg.vector_norm(fused - targets, dim=-1)
    return predicted_delta, true_delta, errors, fused_error, baseline_error.squeeze(-1)


def batch_loss(
    model, predictions, targets, rays, base_scores, center, args, phase,
):
    direct, ght, identity = [], [], []
    for task_index in range(len(TASKS)):
        predicted, true_delta, errors, fused_error, baseline_error = forward_task(
            model, predictions, targets, rays, base_scores, task_index,
            center, args,
        )
        direct.append(training_loss(predicted, true_delta, "balanced_rank"))
        if phase == "ght":
            weights = F.softmax(
                -predicted / task_temperature(TASKS[task_index], args), dim=-1
            )
            expected = (weights * errors).sum(dim=-1).mean()
            identity_violation = F.relu(fused_error - baseline_error).mean()
            # Match the established E2 risk scale while explicitly protecting
            # the existing calibrated baseline.
            ght.append((expected + 0.05 * fused_error.mean()) / 0.01)
            identity.append(args.identity_weight * identity_violation / 0.01)
    result = torch.stack(direct).mean()
    if ght:
        result = result + torch.stack(ght).mean() + torch.stack(identity).mean()
    return result


def evaluate(
    model, arrays, scores, windows, device, center, args, batch_size,
    max_batches=0,
):
    model.eval() if model is not None else None
    store: dict[str, dict[str, list[np.ndarray]]] = {
        f"V{count}": {"baseline": [], "temporal": []}
        for count in (2, 3, 4)
    }
    actions_store = {f"V{count}": [] for count in (2, 3, 4)}
    with torch.inference_mode():
        for batch_no, start in enumerate(range(0, len(windows), batch_size)):
            if max_batches and batch_no >= max_batches:
                break
            rows = windows[start:start + batch_size]
            pred = torch.from_numpy(arrays["predictions"][rows]).to(
                device=device, dtype=torch.float32
            )
            ray = torch.from_numpy(arrays["rays"][rows]).to(
                device=device, dtype=torch.float32
            )
            base = torch.from_numpy(np.asarray(scores[rows])).to(
                device=device, dtype=torch.float32
            )
            target = torch.from_numpy(
                arrays["targets"][rows[:, center]]
            ).to(device=device, dtype=torch.float32)
            actions = arrays["actions"][rows[:, center]]
            for task_index, task in enumerate(TASKS):
                predicted, _, _, fused_error, baseline_error = forward_task(
                    model, pred, target, ray, base, task_index, center, args
                )
                # Reconstruct baseline pose error and temporal fused error;
                # baseline_error is per joint and temporal is fused per joint.
                stage = f"V{len(task)}"
                store[stage]["baseline"].append(
                    baseline_error.detach().cpu().numpy() * 1000.0
                )
                store[stage]["temporal"].append(
                    fused_error.detach().cpu().numpy() * 1000.0
                )
                actions_store[stage].append(actions.copy())
    result: dict[str, object] = {}
    stage_values = []
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(actions_store[stage])
        result[stage] = {}
        for name in ("baseline", "temporal"):
            values = np.concatenate(store[stage][name], axis=0)
            metric = action_equal(values, actions)
            result[stage][name] = {
                "action_equal_all17_mm": metric,
                "frame_weighted_all17_mm": float(values.mean()),
            }
            if name == "temporal":
                stage_values.append(metric)
        result[stage]["delta_mm_temporal_minus_baseline"] = (
            result[stage]["temporal"]["action_equal_all17_mm"]
            - result[stage]["baseline"]["action_equal_all17_mm"]
        )
    result["mean_temporal_mm"] = float(np.mean(stage_values))
    result["windows"] = int(len(windows))
    return result


def main() -> None:
    args = parse_args()
    if args.window_length != 9:
        raise ValueError("H16 is intentionally a T=9 screen")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_cache(args.train_cache)
    validation = load_cache(args.validation_cache)
    train_scores = np.load(args.train_scores, mmap_mode="r")
    validation_scores = np.load(args.validation_scores, mmap_mode="r")
    expected_shape = (len(train["targets"]), len(TASKS), 17, len(ALL_CANDIDATES))
    if tuple(train_scores.shape) != expected_shape:
        raise ValueError(f"bad train score shape {train_scores.shape}, expected {expected_shape}")
    expected_val_shape = (len(validation["targets"]), len(TASKS), 17, len(ALL_CANDIDATES))
    if tuple(validation_scores.shape) != expected_val_shape:
        raise ValueError(f"bad validation score shape {validation_scores.shape}, expected {expected_val_shape}")

    train_meta = metadata_from_pkl(args.train_pkl, len(train["targets"]))
    val_meta = metadata_from_pkl(args.validation_pkl, len(validation["targets"]))
    for name, arrays, meta in (
        ("train", train, train_meta), ("validation", validation, val_meta)
    ):
        if not np.array_equal(arrays["subjects"], meta["subjects"]):
            raise RuntimeError(f"{name} subjects do not match pkl order")
        if not np.array_equal(arrays["actions"], meta["actions"]):
            raise RuntimeError(f"{name} actions do not match pkl order")

    train_windows_all = build_windows(train_meta, args.window_length)
    val_windows = build_windows(val_meta, args.window_length)
    train_subjects = train_meta["subjects"][train_windows_all[:, args.window_length // 2]]
    train_windows = train_windows_all[np.isin(train_subjects, [1, 5, 6, 7])]
    holdout_windows = train_windows_all[train_subjects == 8]
    if args.max_train_windows and len(train_windows) > args.max_train_windows:
        # Cover the complete subject/action ordering instead of taking only
        # the first sequence when doing the short screening run.
        keep = np.linspace(
            0, len(train_windows) - 1, args.max_train_windows,
            dtype=np.int64,
        )
        train_windows = train_windows[keep]
    if args.max_holdout_windows and len(holdout_windows) > args.max_holdout_windows:
        keep = np.linspace(
            0, len(holdout_windows) - 1, args.max_holdout_windows,
            dtype=np.int64,
        )
        holdout_windows = holdout_windows[keep]
    if len(train_windows) == 0 or len(holdout_windows) == 0:
        raise RuntimeError("empty train or S8 holdout window set")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "T-CVU residual on frozen calibrated E2-C2 scores",
        "train_cache": str(Path(args.train_cache).resolve()),
        "validation_cache": str(Path(args.validation_cache).resolve()),
        "train_pkl": str(Path(args.train_pkl).resolve()),
        "validation_pkl": str(Path(args.validation_pkl).resolve()),
        "window_length": args.window_length, "frame_stride": FRAME_STRIDE,
        "candidate_count": len(ALL_CANDIDATES),
        "candidate_order": "H76 11 + confidence-weighted 11",
        "train_subjects": [1, 5, 6, 7], "holdout_subjects": [8],
        "train_windows": int(len(train_windows)),
        "holdout_windows": int(len(holdout_windows)),
        "validation_windows": int(len(val_windows)),
        "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Identity baseline on the internal holdout is the selection floor.
    baseline_holdout = evaluate(
        None, train, train_scores, holdout_windows, device,
        args.window_length // 2, args, args.batch_size,
    )
    baseline_val = evaluate(
        None, validation, validation_scores, val_windows, device,
        args.window_length // 2, args, args.batch_size,
    )
    (out / "baseline_holdout.json").write_text(
        json.dumps(baseline_holdout, indent=2) + "\n"
    )
    (out / "baseline_validation.json").write_text(
        json.dumps(baseline_val, indent=2) + "\n"
    )
    print(json.dumps({"baseline_holdout": baseline_holdout,
                      "baseline_validation": baseline_val}, indent=2), flush=True)

    model = TemporalResidualUtility(10, args.window_length).to(device)
    best_metric = float(baseline_holdout["mean_temporal_mm"])
    best_epoch = -1
    history = []
    optimizer = None
    for epoch in range(args.epochs):
        phase = "direct" if epoch < args.direct_epochs else "ght"
        lr = args.lr if phase == "direct" else args.ght_lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_windows))
        losses = []
        for batch_no, offset in enumerate(range(0, len(order), args.batch_size)):
            rows = train_windows[order[offset:offset + args.batch_size]]
            pred = torch.from_numpy(train["predictions"][rows]).to(
                device=device, dtype=torch.float32
            )
            ray = torch.from_numpy(train["rays"][rows]).to(
                device=device, dtype=torch.float32
            )
            base = torch.from_numpy(np.asarray(train_scores[rows])).to(
                device=device, dtype=torch.float32
            )
            target = torch.from_numpy(
                train["targets"][rows[:, args.window_length // 2]]
            ).to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(
                model, pred, target, ray, base,
                args.window_length // 2, args, phase,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout = evaluate(
            model, train, train_scores, holdout_windows, device,
            args.window_length // 2, args, args.batch_size,
        )
        metric = float(holdout["mean_temporal_mm"])
        row = {
            "epoch": epoch, "phase": phase, "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": metric, "holdout": holdout,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "epoch": epoch,
                "feature_dim": 10, "window_length": args.window_length,
                "best_holdout_metric_mm": best_metric,
            }, out / "model_best.pth.tar")

    if best_epoch >= 0:
        best = torch.load(out / "model_best.pth.tar", map_location=device,
                          weights_only=False)
        model.load_state_dict(best["state_dict"], strict=True)
    final_val = evaluate(
        model if best_epoch >= 0 else None, validation, validation_scores,
        val_windows, device, args.window_length // 2, args, args.batch_size,
    )
    result = {
        **manifest,
        "baseline_holdout": baseline_holdout,
        "baseline_validation": baseline_val,
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": final_val,
        "decision": (
            "temporal residual retained for follow-up"
            if best_epoch >= 0 and final_val["mean_temporal_mm"] < baseline_val["mean_temporal_mm"]
            else "no temporal gain; keep calibrated E2-C2 and stop this temporal branch"
        ),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "COMPLETED").write_text("completed\n")
    print(json.dumps({"S9_S11_final_once": final_val,
                      "decision": result["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
