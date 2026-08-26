#!/usr/bin/env python3
"""Causal MixSTE temporal residual on frozen E2-C2 candidates.

H18/H19 smoothed already-fused 3D.  This experiment keeps H76 and E2 frozen
and only learns a zero-initialized residual on candidate utilities, using
the past-plus-current T=9 window and reporting the latest frame.  MixSTE's
alternating spatial/temporal blocks come from the official factorized
encoder used in prior screens, not a new architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_e2_clean_temporal_residual_20260818 import (
    action_equal,
    build_windows,
    metadata_from_pkl,
    seed_everything,
    select_rows,
)

FEATURE_DIM = 17

from train_h76_counterfactual_delta_20260811 import training_loss
from train_temporal_e2_c2_screen_20260818 import TASKS, load_cache, task_spec


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
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--temperature-v2", type=float, default=0.4)
    p.add_argument("--temperature-v3", type=float, default=1.8)
    p.add_argument("--temperature-v4", type=float, default=1.8)
    p.add_argument("--identity-weight", type=float, default=0.5)
    p.add_argument("--v2-loss-weight", type=float, default=1.0)
    p.add_argument("--temporal-loss-weight", type=float, default=0.0)
    p.add_argument("--geometry-gate", action="store_true")
    p.add_argument("--encoder", choices=("mixste", "joint"), default="mixste")
    p.add_argument("--train-window-stride", type=int, default=3)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-holdout-windows", type=int, default=0)
    p.add_argument("--max-validation-windows", type=int, default=0)
    p.add_argument("--no-refit-all", action="store_true")
    return p.parse_args()


def task_temperature(task: tuple[int, ...], args: argparse.Namespace) -> float:
    return {2: args.temperature_v2, 3: args.temperature_v3,
            4: args.temperature_v4}[len(task)]


class MixSTECandidateTemporal(nn.Module):
    """Joint-then-time MixSTE encoder with identity-initialized logits."""

    def __init__(self, feature_dim: int, window_length: int, hidden_dim: int,
                 layers: int, dropout: float, geometry_gate: bool):
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden-dim must be divisible by 8")
        self.window_length = window_length
        self.geometry_gate = geometry_gate
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.time_embedding = nn.Parameter(torch.zeros(window_length, hidden_dim))
        self.joint_embedding = nn.Parameter(torch.zeros(17, hidden_dim))
        self.spatial = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        self.candidate_embedding = nn.Parameter(torch.zeros(22, hidden_dim))
        nn.init.trunc_normal_(self.candidate_embedding, std=0.02)
        if geometry_gate:
            self.gate = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, 4.0)
        else:
            self.gate = None

    def _run_blocks(self, tokens: torch.Tensor, joints: int) -> torch.Tensor:
        encoded = []
        for start in range(0, len(tokens), 16384):
            encoded.append(self._one_chunk(tokens[start:start + 16384], joints))
        return torch.cat(encoded, dim=0)

    def _one_chunk(self, tokens: torch.Tensor, joints: int) -> torch.Tensor:
        # tokens: N,T,J,H  with N = B*C
        n_seq, time, _, hidden = tokens.shape
        x = tokens
        for spatial, temporal in zip(self.spatial, self.temporal):
            x = spatial(x.reshape(n_seq * time, joints, hidden))
            x = x.reshape(n_seq, time, joints, hidden)
            x = x.permute(0, 2, 1, 3).reshape(n_seq * joints, time, hidden)
            x = temporal(x)
            x = x.reshape(n_seq, joints, time, hidden).permute(0, 2, 1, 3)
        return x

    def forward(self, features: torch.Tensor,
                gate_features: torch.Tensor | None = None,
                available: np.ndarray | None = None,
                return_sequence: bool = False) -> torch.Tensor:
        batch, time, candidates, joints, _ = features.shape
        if time != self.window_length:
            raise ValueError(f"expected T={self.window_length}, got {time}")
        x = self.input(features)
        x = x + self.time_embedding[None, :, None, None]
        x = x + self.joint_embedding[None, None, None]
        if available is None:
            x = x + self.candidate_embedding[:candidates][None, None, :, None]
        else:
            idx = torch.as_tensor(available, device=x.device, dtype=torch.long)
            x = x + self.candidate_embedding[idx][None, None, :, None]
        x = x.reshape(batch * candidates, time, joints, -1)
        x = self._run_blocks(x, joints)
        residual_seq = self.output(x).reshape(batch, candidates, time, joints)
        latest = residual_seq[:, :, -1]
        if self.gate is not None and gate_features is not None:
            gate = torch.sigmoid(self.gate(gate_features)).permute(0, 2, 1)
            latest = latest * gate
            residual_seq = residual_seq * gate[:, :, None]
        if return_sequence:
            return latest, residual_seq
        return latest


class JointTemporalResidual(nn.Module):
    """H16 per-joint temporal encoder, but causal: latest frame only."""

    def __init__(self, feature_dim: int, window_length: int, hidden_dim: int,
                 layers: int, dropout: float, geometry_gate: bool):
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden-dim must be divisible by 8")
        self.window_length = window_length
        self.geometry_gate = geometry_gate
        self.input = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.position = nn.Parameter(torch.zeros(window_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        nn.init.trunc_normal_(self.position, std=0.02)
        if geometry_gate:
            self.gate = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, 4.0)
        else:
            self.gate = None

    def forward(self, features: torch.Tensor,
                gate_features: torch.Tensor | None = None,
                available: np.ndarray | None = None,
                return_sequence: bool = False) -> torch.Tensor:
        batch, time, candidates, joints, feature_dim = features.shape
        if time != self.window_length:
            raise ValueError(f"expected T={self.window_length}, got {time}")
        tokens = features.permute(0, 2, 3, 1, 4).reshape(
            batch * candidates * joints, time, feature_dim
        )
        tokens = self.input(tokens) + self.position[None]
        encoded_chunks = []
        for start in range(0, len(tokens), 16384):
            encoded_chunks.append(self.encoder(tokens[start:start + 16384]))
        encoded = torch.cat(encoded_chunks, dim=0)
        residual_seq = self.output(encoded).reshape(
            batch, candidates, joints, time
        ).permute(0, 1, 3, 2)
        latest = residual_seq[:, :, -1]
        if self.gate is not None and gate_features is not None:
            gate = torch.sigmoid(self.gate(gate_features)).permute(0, 2, 1)
            latest = latest * gate
            residual_seq = residual_seq * gate[:, :, None]
        if return_sequence:
            return latest, residual_seq
        return latest


def make_features(predictions, rays, scores, available, masks):
    """Causal features referenced to the latest frame, no future context."""
    candidate = predictions[:, :, available]
    root_relative = candidate - candidate[..., :1, :]
    last = root_relative[:, -1:]
    displacement = (root_relative - last) / 0.1
    displacement_norm = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    velocity = torch.diff(candidate, dim=1, prepend=candidate[:, :1])
    acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1])
    confidence = rays[..., 6]
    mask = masks.to(device=predictions.device, dtype=predictions.dtype)
    confidence = (
        confidence[:, :, None, :, :] * mask[None, None, :, None, :]
    ).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)[None, None, :, None]
    confidence = confidence.unsqueeze(-1)
    score = scores[:, :, :, available].permute(0, 1, 3, 2).unsqueeze(-1) / 5.0
    score_delta = score - score[:, -1:]
    spread = candidate.std(dim=2, keepdim=True).mean(dim=-1, keepdim=True)
    spread = spread.expand(-1, -1, candidate.shape[2], -1, -1)
    features = torch.cat(
        (root_relative, displacement, displacement_norm, velocity, acceleration,
         confidence, score, score_delta, spread), dim=-1,
    )
    if features.shape[-1] != FEATURE_DIM:
        raise RuntimeError(f"feature dim {features.shape[-1]} != {FEATURE_DIM}")
    gate = torch.stack(
        (confidence[:, -1, :, :, 0].mean(dim=1),
         spread[:, -1, 0, :, 0],
         score[:, -1, :, :, 0].std(dim=1)), dim=-1,
    )
    return features, gate


def forward_task(model, predictions, targets, rays, base_scores, task_index, args):
    task, available, masks_np, baseline_local = task_spec(task_index)
    masks = torch.from_numpy(masks_np)
    task_scores = base_scores[:, :, task_index]
    features, gate = make_features(predictions, rays, task_scores, available, masks)
    want_seq = bool(args.temporal_loss_weight > 0)
    if model is None:
        correction = torch.zeros(
            predictions.shape[0], len(available), 17,
            device=predictions.device, dtype=predictions.dtype,
        )
        correction_seq = None
    elif want_seq:
        correction, correction_seq = model(
            features, gate, available, return_sequence=True
        )
    else:
        correction = model(features, gate, available)
        correction_seq = None
    center_scores = task_scores[:, -1, :, available].permute(0, 2, 1)
    total_scores = center_scores + correction
    predicted_delta = (
        total_scores - total_scores[:, baseline_local:baseline_local + 1]
    ).permute(0, 2, 1)
    candidate_last = predictions[:, -1, available]
    target_last = targets[:, -1]
    errors = torch.linalg.vector_norm(
        candidate_last - target_last[:, None], dim=-1
    ).permute(0, 2, 1)
    baseline_error = errors[..., baseline_local:baseline_local + 1]
    true_delta = errors - baseline_error
    weights = F.softmax(-predicted_delta / task_temperature(task, args), dim=-1)
    fused = torch.einsum("bjc,bcjd->bjd", weights, candidate_last)
    fused_error = torch.linalg.vector_norm(fused - target_last, dim=-1)
    fused_seq = None
    if args.temporal_loss_weight > 0:
        fused_seq = []
        for time_index in range(predictions.shape[1]):
            cand_t = predictions[:, time_index, available]
            score_t = task_scores[:, time_index, :, available].permute(0, 2, 1)
            if correction_seq is not None:
                score_t = score_t + correction_seq[:, :, time_index]
            elif time_index == predictions.shape[1] - 1:
                score_t = score_t + correction
            delta_t = (score_t - score_t[:, baseline_local:baseline_local + 1]).permute(0, 2, 1)
            w_t = F.softmax(-delta_t / task_temperature(task, args), dim=-1)
            fused_seq.append(torch.einsum("bjc,bcjd->bjd", w_t, cand_t))
        fused_seq = torch.stack(fused_seq, dim=1)
    return predicted_delta, true_delta, errors, fused_error, baseline_error.squeeze(-1), fused_seq


def batch_loss(model, predictions, targets, rays, base_scores, args):
    direct, identity, temporal = [], [], []
    for task_index, task in enumerate(TASKS):
        predicted, true_delta, errors, fused_error, baseline_error, fused_seq = forward_task(
            model, predictions, targets, rays, base_scores, task_index, args
        )
        weight = args.v2_loss_weight if len(task) == 2 else 1.0
        direct.append(weight * training_loss(predicted, true_delta, "balanced_rank"))
        identity.append(args.identity_weight * F.relu(fused_error - baseline_error).mean())
        if fused_seq is not None:
            temporal.append(F.smooth_l1_loss(
                torch.diff(fused_seq, dim=1), torch.diff(targets, dim=1),
                beta=0.005,
            ))
    loss = torch.stack(direct).mean() + torch.stack(identity).mean()
    if temporal:
        loss = loss + args.temporal_loss_weight * torch.stack(temporal).mean()
    return loss


def evaluate(model, arrays, scores, windows, device, args, batch_size):
    if model is not None:
        model.eval()
    store = {f"V{k}": {"baseline": [], "temporal": []} for k in (2, 3, 4)}
    actions_store = {f"V{k}": [] for k in (2, 3, 4)}
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            rows = windows[start:start + batch_size]
            pred = torch.from_numpy(arrays["predictions"][rows]).to(device=device, dtype=torch.float32)
            ray = torch.from_numpy(arrays["rays"][rows]).to(device=device, dtype=torch.float32)
            base = torch.from_numpy(np.asarray(scores[rows])).to(device=device, dtype=torch.float32)
            target = torch.from_numpy(arrays["targets"][rows]).to(device=device, dtype=torch.float32)
            actions = arrays["actions"][rows[:, -1]]
            for task_index, task in enumerate(TASKS):
                _, _, _, fused_error, baseline_error, _ = forward_task(
                    model, pred, target, ray, base, task_index, args
                )
                stage = f"V{len(task)}"
                store[stage]["baseline"].append(baseline_error.cpu().numpy() * 1000.0)
                store[stage]["temporal"].append(fused_error.cpu().numpy() * 1000.0)
                actions_store[stage].append(actions.copy())
    result = {}
    means = []
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(actions_store[stage])
        base = action_equal(np.concatenate(store[stage]["baseline"]), actions)
        temporal = action_equal(np.concatenate(store[stage]["temporal"]), actions)
        result[stage] = {
            "baseline_absolute_mm": float(base),
            "temporal_absolute_mm": float(temporal),
            "delta_absolute_mm": float(temporal - base),
        }
        means.append(temporal)
    result["mean_v234_mm"] = float(np.mean(means))
    result["windows"] = int(len(windows))
    return result


def train_epoch(model, optimizer, arrays, scores, windows, device, args, seed):
    model.train()
    order = np.random.default_rng(seed).permutation(len(windows))
    losses = []
    for offset in range(0, len(order), args.batch_size):
        rows = windows[order[offset:offset + args.batch_size]]
        pred = torch.from_numpy(arrays["predictions"][rows]).to(device=device, dtype=torch.float32)
        ray = torch.from_numpy(arrays["rays"][rows]).to(device=device, dtype=torch.float32)
        base = torch.from_numpy(np.asarray(scores[rows])).to(device=device, dtype=torch.float32)
        target = torch.from_numpy(arrays["targets"][rows]).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, pred, target, ray, base, args)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def make_model(args, feature_dim: int, device):
    cls = JointTemporalResidual if args.encoder == "joint" else MixSTECandidateTemporal
    return cls(
        feature_dim, args.window_length, args.hidden_dim, args.layers,
        args.dropout, args.geometry_gate,
    ).to(device)


def main() -> None:
    args = parse_args()
    if args.window_length < 2:
        raise ValueError("causal windows need at least two frames")
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("loading caches", flush=True)
    train = load_cache(args.train_cache)
    validation = load_cache(args.validation_cache)
    train_scores = np.load(args.train_scores, mmap_mode="r")
    val_scores = np.load(args.validation_scores, mmap_mode="r")
    train_meta = metadata_from_pkl(args.train_pkl, len(train["targets"]))
    val_meta = metadata_from_pkl(args.validation_pkl, len(validation["targets"]))
    for name, arrays, meta in (("train", train, train_meta),
                               ("validation", validation, val_meta)):
        if not np.array_equal(arrays["subjects"], meta["subjects"]):
            raise RuntimeError(f"{name} subject order mismatch")
        if not np.array_equal(arrays["actions"], meta["actions"]):
            raise RuntimeError(f"{name} action order mismatch")

    print("building causal windows", flush=True)
    all_train = build_windows(train_meta, args.window_length, args.frame_stride)
    all_val = build_windows(val_meta, args.window_length, args.frame_stride)
    latest = all_train[:, -1]
    selection = all_train[np.isin(train_meta["subjects"][latest], [1, 5, 6, 7])]
    holdout = all_train[train_meta["subjects"][latest] == 8]
    official = all_train[np.isin(train_meta["subjects"][latest], [1, 5, 6, 7, 8])]
    if args.train_window_stride > 1:
        selection = selection[::args.train_window_stride]
        official = official[::args.train_window_stride]
    selection = select_rows(selection, args.max_train_windows)
    holdout = select_rows(holdout, args.max_holdout_windows)
    official = select_rows(official, args.max_train_windows)
    all_val = select_rows(all_val, args.max_validation_windows)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("evaluating causal E2-C2 baseline", flush=True)
    baseline_holdout = evaluate(None, train, train_scores, holdout, device, args, args.batch_size)
    baseline_val = evaluate(None, validation, val_scores, all_val, device, args, args.batch_size)
    feature_dim = FEATURE_DIM
    model = make_model(args, feature_dim, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    probe = holdout[: min(256, len(holdout))]
    init_probe = evaluate(model, train, train_scores, probe, device, args, args.batch_size)
    base_probe = evaluate(None, train, train_scores, probe, device, args, args.batch_size)
    init_gap = abs(init_probe["mean_v234_mm"] - base_probe["mean_v234_mm"])
    print(json.dumps({"identity_probe_gap_mm": init_gap,
                      "init_probe": init_probe, "base_probe": base_probe},
                     indent=2), flush=True)
    if init_gap > 0.05:
        raise RuntimeError(f"zero-init identity failed: gap {init_gap:.4f} mm")
    manifest = {
        "method": "causal candidate-utility residual on frozen E2-C2",
        "protocol": {
            "evaluation_output": "latest frame",
            "future_context": False,
            "encoder": args.encoder,
            "spatial_temporal": (
                "MixSTE alternating STB/TTB" if args.encoder == "mixste"
                else "H16 per-joint temporal"
            ),
            "geometry_gate": args.geometry_gate,
            "v2_loss_weight": args.v2_loss_weight,
            "temporal_loss_weight": args.temporal_loss_weight,
            "train_window_stride": args.train_window_stride,
        },
        "selection_windows": int(len(selection)),
        "holdout_windows": int(len(holdout)),
        "official_windows": int(len(official)),
        "validation_windows": int(len(all_val)),
        "args": vars(args),
        "identity_probe_gap_mm": init_gap,
        "baseline_holdout": baseline_holdout,
        "baseline_validation": baseline_val,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"baseline_holdout": baseline_holdout,
                      "baseline_validation": baseline_val}, indent=2), flush=True)

    best_metric = float(baseline_holdout["mean_v234_mm"])
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, optimizer, train, train_scores, selection, device, args,
            args.seed + epoch,
        )
        holdout_result = evaluate(
            model, train, train_scores, holdout, device, args, args.batch_size
        )
        row = {"epoch": epoch, "train_loss": train_loss,
               "holdout": holdout_result,
               "holdout_selection_metric_mm": holdout_result["mean_v234_mm"]}
        history.append(row)
        print(json.dumps({"selection": row}, default=str), flush=True)
        if holdout_result["mean_v234_mm"] < best_metric:
            best_metric = float(holdout_result["mean_v234_mm"])
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "args": vars(args)}, out / "model_selection_best.pth.tar")

    refit_epochs = max(best_epoch + 1, 0)
    refit_history = []
    if best_epoch >= 0 and not args.no_refit_all:
        seed_everything(args.seed)
        final_model = make_model(args, feature_dim, device)
        final_opt = torch.optim.AdamW(
            final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for epoch in range(refit_epochs):
            stats = train_epoch(
                final_model, final_opt, train, train_scores, official, device,
                args, args.seed + epoch,
            )
            refit_history.append({"epoch": epoch, "train_loss": stats})
            print(json.dumps({"official_refit": refit_history[-1]}), flush=True)
        torch.save({"state_dict": final_model.state_dict(),
                    "epochs": refit_epochs, "args": vars(args)},
                   out / "model_official_refit.pth.tar")
    elif best_epoch >= 0:
        final_model = model
        state = torch.load(out / "model_selection_best.pth.tar",
                           map_location=device, weights_only=False)
        final_model.load_state_dict(state["state_dict"], strict=True)
    else:
        final_model = None

    final_val = evaluate(
        final_model, validation, val_scores, all_val, device, args, args.batch_size
    )
    result = {
        **manifest,
        "best_epoch": best_epoch,
        "best_holdout_metric_mm": best_metric,
        "selection_history": history,
        "official_refit_epochs": refit_epochs,
        "official_refit_history": refit_history,
        "S9_S11_final_once": final_val,
        "gbt_hrnet_target_mm": {"V2": 36.8, "V3": 30.4, "V4": 26.0},
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "COMPLETED").write_text("completed\n")
    print(json.dumps({"S9_S11_final_once": final_val, "best_epoch": best_epoch},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
