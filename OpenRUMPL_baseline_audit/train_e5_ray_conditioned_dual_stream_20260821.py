#!/usr/bin/env python3
"""Ray-conditioned ST/TS dual-stream temporal refinement of frozen K96.

This is a deliberately small, falsifiable adaptation of two published ideas:

* UniCodebook's parallel spatial->temporal and temporal->spatial branches with
  token-wise adaptive fusion;
* DCSA-style cross attention, with the unavailable discrete 2D/3D tokens
  replaced by RUMPL's *observed* per-view rays, confidence, and ray residuals.

The latter is an adaptation, not a reproduction of UniCodebook.  It addresses
the information missing from the failed H19 pose-only temporal residual: the
network can see which view/joint observation supports the frozen K96 anchor at
every frame.  The output heads are zero initialized, hence step zero is exactly
the frozen coordinate-only K96/E2-C2 result.

This file reuses the audited H19 subject split, causal-window construction,
two-stage S8 selection/refit protocol, and metrics.  Only the model and the
batch forward path are replaced.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_e2_causal_temporal_seq2seq_20260818 as h19
from train_e2_v234_universal_20260812 import ORIGINAL_COMBINATIONS


TASKS = tuple(ORIGINAL_COMBINATIONS)
TASK_VIEW_MASK = torch.tensor(
    [[view in task for view in range(4)] for task in TASKS], dtype=torch.bool
)
TASK_CARDINALITY = torch.tensor([len(task) for task in TASKS], dtype=torch.long)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", required=True)
    p.add_argument("--train-fused", required=True)
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--validation-fused", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window-length", type=int, default=9)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--relative-scale-m", type=float, default=0.10)
    p.add_argument("--root-scale-m", type=float, default=0.05)
    p.add_argument("--root-mode", choices=("protected", "learned"),
                   default="learned")
    p.add_argument("--temporal-loss-weight", type=float, default=0.0)
    p.add_argument("--observation-mode", choices=("ray-cross", "none"),
                   default="ray-cross")
    p.add_argument("--cross-gate-init", type=float, default=-2.0)
    p.add_argument("--gate-mm", type=float, default=0.15)
    p.add_argument("--preload-fused", action="store_true")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpu", default="0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-holdout-windows", type=int, default=0)
    p.add_argument("--max-validation-windows", type=int, default=0)
    p.add_argument("--no-refit-all", action="store_true")
    return p.parse_args()


class RayObservationEncoder(nn.Module):
    """Encode a candidate pose's consistency with every selected camera ray."""

    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        # direction(3), root-centred origin(3), confidence(1), signed
        # perpendicular residual(3), residual norm(1), signed depth(1).
        self.projection = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.view_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.view_norm = nn.LayerNorm(hidden_dim)
        self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def forward(self, pose: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
        # pose B,T,K,J,3; rays B,T,J,4,7 -> B,T,K,J,H
        batch, time, tasks, joints, _ = pose.shape
        direction = F.normalize(rays[..., :3], dim=-1)
        origin = rays[..., 3:6]
        confidence = rays[..., 6:7].clamp(0.0, 1.0)
        root = pose[..., :1, :]
        # Vectorize all 11 view tasks.  The previous semantically equivalent
        # Python loop issued 22 tiny MHA kernels per batch and left the 4090
        # mostly idle.
        candidate = pose[..., None, :]                         # B,T,K,J,1,3
        candidate_root = root[..., None, :]                    # B,T,K,1,1,3
        direction = direction[:, :, None]                      # B,T,1,J,V,3
        origin = origin[:, :, None]
        confidence = confidence[:, :, None]
        offset = candidate - origin
        depth = (offset * direction).sum(dim=-1, keepdim=True)
        perpendicular = offset - depth * direction
        features = torch.cat(
            (
                direction.expand(-1, -1, tasks, -1, -1, -1),
                (origin - candidate_root).expand(
                    -1, -1, tasks, joints, -1, -1
                ) / 5.0,
                confidence.expand(-1, -1, tasks, -1, -1, -1),
                perpendicular / 0.10,
                torch.linalg.vector_norm(
                    perpendicular, dim=-1, keepdim=True
                ) / 0.10,
                depth / 5.0,
            ), dim=-1,
        )
        token = self.projection(features).reshape(
            batch * time * tasks * joints, 4, -1
        )
        missing = (~TASK_VIEW_MASK[:tasks]).to(token.device)
        padding = missing[None, None, :, None].expand(
            batch, time, tasks, joints, 4
        ).reshape(batch * time * tasks * joints, 4)
        attended, _ = self.view_attention(
            token, token, token, key_padding_mask=padding, need_weights=False
        )
        token = self.view_norm(token + attended)
        query = self.pool_query.expand(len(token), -1, -1)
        pooled, _ = self.pool_attention(
            query, token, token, key_padding_mask=padding, need_weights=False
        )
        return pooled[:, 0].reshape(batch, time, tasks, joints, -1)


class HybridDualStreamBlock(nn.Module):
    """Parallel ST/TS branches with observation cross-attention and fusion."""

    def __init__(self, hidden_dim: int, heads: int, gate_init: float):
        super().__init__()

        def layer():
            return nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=heads,
                dim_feedforward=2 * hidden_dim, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )

        self.st_spatial = layer()
        self.st_temporal = layer()
        self.ts_temporal = layer()
        self.ts_spatial = layer()
        self.obs_norm = nn.LayerNorm(hidden_dim)
        self.pose_norm = nn.LayerNorm(hidden_dim)
        self.cross = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.0, batch_first=True
        )
        self.cross_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.fusion = nn.Linear(2 * hidden_dim, 2)

    @staticmethod
    def _spatial(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, t, k, j, h = x.shape
        return block(x.reshape(b * t * k, j, h)).reshape(b, t, k, j, h)

    @staticmethod
    def _temporal(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        b, t, k, j, h = x.shape
        y = x.permute(0, 2, 3, 1, 4).reshape(b * k * j, t, h)
        y = block(y)
        return y.reshape(b, k, j, t, h).permute(0, 3, 1, 2, 4)

    def forward(self, pose: torch.Tensor, observation: torch.Tensor,
                use_observation: bool) -> torch.Tensor:
        b, t, k, j, h = pose.shape
        if use_observation:
            query = self.pose_norm(pose).reshape(b * t * k, j, h)
            evidence = self.obs_norm(observation).reshape(b * t * k, j, h)
            injected, _ = self.cross(
                query, evidence, evidence, need_weights=False
            )
            pose = pose + torch.sigmoid(self.cross_gate) * injected.reshape_as(pose)
        st = self._temporal(self.st_temporal,
                            self._spatial(self.st_spatial, pose))
        ts = self._spatial(self.ts_spatial,
                           self._temporal(self.ts_temporal, pose))
        alpha = torch.softmax(self.fusion(torch.cat((st, ts), dim=-1)), dim=-1)
        return alpha[..., :1] * st + alpha[..., 1:] * ts


class RayConditionedDualStream(nn.Module):
    """Identity-safe K96 residual with observation-conditioned ST/TS blocks."""

    def __init__(self, window_length: int, hidden_dim: int, layers: int,
                 relative_scale_m: float, root_scale_m: float,
                 root_mode: str, task_count: int = h19.TASK_COUNT):
        super().__init__()
        args = _ARGS
        if hidden_dim % args.heads:
            raise ValueError("hidden-dim must be divisible by heads")
        self.window_length = window_length
        self.relative_scale_m = relative_scale_m
        self.root_scale_m = root_scale_m
        self.root_mode = root_mode
        self.use_observation = args.observation_mode == "ray-cross"
        self.pose_input = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.observation = RayObservationEncoder(hidden_dim, args.heads)
        self.blocks = nn.ModuleList([
            HybridDualStreamBlock(hidden_dim, args.heads, args.cross_gate_init)
            for _ in range(layers)
        ])
        self.joint_embedding = nn.Parameter(
            torch.zeros(h19.JOINT_COUNT, hidden_dim)
        )
        # Encode only cardinality, never a fixed H36M camera-combination ID.
        # Camera identity/layout must be expressed by its ray geometry so the
        # module remains defined for unseen camera subsets.
        self.cardinality_embedding = nn.Embedding(3, hidden_dim)
        self.time_embedding = nn.Parameter(torch.zeros(window_length, hidden_dim))
        self.relative_output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3)
        )
        self.root_output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3)
        )
        for head in (self.relative_output[-1], self.root_output[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.cardinality_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.time_embedding, std=0.02)

    def forward(self, pose: torch.Tensor, task_ids: torch.Tensor,
                rays: torch.Tensor) -> torch.Tensor:
        root = pose[..., :1, :]
        relative = pose - root
        velocity = torch.diff(pose, dim=1, prepend=pose[:, :1])
        acceleration = torch.diff(velocity, dim=1, prepend=velocity[:, :1])
        feature = torch.cat(
            (relative, root.expand_as(pose), velocity, acceleration), dim=-1
        )
        x = self.pose_input(feature)
        x = x + self.joint_embedding[None, None, None]
        cardinality = TASK_CARDINALITY.to(task_ids.device)[task_ids] - 2
        x = x + self.cardinality_embedding(cardinality)[:, None, :, None]
        x = x + self.time_embedding[None, :, None, None]
        observation = (
            self.observation(pose, rays) if self.use_observation
            else torch.zeros_like(x)
        )
        for block in self.blocks:
            x = block(x, observation, self.use_observation)
        relative_delta = self.relative_scale_m * torch.tanh(
            self.relative_output(x)
        )
        relative_delta = relative_delta.clone()
        relative_delta[..., 0, :] = 0.0
        if self.root_mode == "learned":
            root_delta = self.root_scale_m * torch.tanh(
                self.root_output(x.mean(dim=3))
            )[..., None, :]
        else:
            root_delta = torch.zeros_like(relative_delta[..., :1, :])
        return pose + relative_delta + root_delta


def gather_batch(cache, rows: np.ndarray, fused: np.ndarray,
                 device: torch.device):
    pose = torch.from_numpy(np.asarray(fused[rows])).to(
        device=device, dtype=torch.float32
    )
    target = torch.from_numpy(np.asarray(cache["targets"][rows])).to(
        device=device, dtype=torch.float32
    )
    rays = torch.from_numpy(np.asarray(cache["rays"][rows])).to(
        device=device, dtype=torch.float32
    )
    task_ids = torch.arange(h19.TASK_COUNT, device=device)[None].expand(
        len(rows), -1
    )
    return pose, target, task_ids, rays


@torch.inference_mode()
def evaluate(model, cache, fused, windows, device, batch_size):
    names = ("absolute", "root", "root_relative")
    base_values = {n: {f"V{k}": [] for k in (2, 3, 4)} for n in names}
    pred_values = {n: {f"V{k}": [] for k in (2, 3, 4)} for n in names}
    action_values = {f"V{k}": [] for k in (2, 3, 4)}
    if model is not None:
        model.eval()
    # The identity baseline does not need rays or model activations.  Using the
    # tiny training batch here made every launch spend minutes recomputing an
    # already-known control before the first optimizer step.
    eval_batch_size = batch_size if model is not None else max(batch_size, 512)
    for start in range(0, len(windows), eval_batch_size):
        rows = windows[start:start + eval_batch_size]
        if model is None:
            pose = torch.from_numpy(np.asarray(fused[rows])).to(
                device=device, dtype=torch.float32
            )
            target_sequence = torch.from_numpy(
                np.asarray(cache["targets"][rows])
            ).to(device=device, dtype=torch.float32)
            task_ids = rays = None
        else:
            pose, target_sequence, task_ids, rays = gather_batch(
                cache, rows, fused, device
            )
        baseline = pose[:, -1]
        prediction = baseline if model is None else model(
            pose, task_ids, rays
        )[:, -1]
        target = target_sequence[:, -1, None]
        base_batch = {
            "absolute": torch.linalg.vector_norm(baseline - target, dim=-1),
            "root": torch.linalg.vector_norm(
                baseline[..., 0, :] - target[..., 0, :], dim=-1
            ),
            "root_relative": torch.linalg.vector_norm(
                (baseline - baseline[..., :1, :]) -
                (target - target[..., :1, :]), dim=-1
            ),
        }
        pred_batch = {
            "absolute": torch.linalg.vector_norm(prediction - target, dim=-1),
            "root": torch.linalg.vector_norm(
                prediction[..., 0, :] - target[..., 0, :], dim=-1
            ),
            "root_relative": torch.linalg.vector_norm(
                (prediction - prediction[..., :1, :]) -
                (target - target[..., :1, :]), dim=-1
            ),
        }
        actions = np.asarray(cache["actions"][rows[:, -1]])
        counts = (2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4)
        for task_index, count in enumerate(counts):
            stage = f"V{count}"
            for name in names:
                base_values[name][stage].append(
                    base_batch[name][:, task_index].cpu().numpy() * 1000.0
                )
                pred_values[name][stage].append(
                    pred_batch[name][:, task_index].cpu().numpy() * 1000.0
                )
            action_values[stage].append(actions.copy())
    result, stage_metrics = {}, []
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(action_values[stage])
        row = {}
        for name in names:
            base = np.concatenate(base_values[name][stage])
            pred = np.concatenate(pred_values[name][stage])
            b = h19.action_equal(base, actions)
            value = h19.action_equal(pred, actions)
            row[f"baseline_{name}_mm"] = float(b)
            row[f"temporal_{name}_mm"] = float(value)
            row[f"delta_{name}_mm"] = float(value - b)
        result[stage] = row
        stage_metrics.append(row["temporal_absolute_mm"])
    result["mean_v234_mm"] = float(np.mean(stage_metrics))
    result["windows"] = int(len(windows))
    return result


def train_epoch(model, optimizer, cache, fused, windows, device, batch_size,
                seed, temporal_loss_weight):
    model.train()
    order = np.random.default_rng(seed).permutation(len(windows))
    losses, coordinates, mpjpes, temporals = [], [], [], []
    for offset in range(0, len(order), batch_size):
        rows = windows[order[offset:offset + batch_size]]
        pose, target, task_ids, rays = gather_batch(cache, rows, fused, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pose, task_ids, rays)
        target_expanded = target[:, :, None].expand_as(prediction)
        coordinate = F.smooth_l1_loss(
            prediction, target_expanded, beta=0.01
        )
        mpjpe = torch.linalg.vector_norm(
            prediction - target_expanded, dim=-1
        ).mean()
        temporal = prediction.new_zeros(())
        if temporal_loss_weight > 0:
            temporal = F.smooth_l1_loss(
                torch.diff(prediction, dim=1),
                torch.diff(target_expanded, dim=1), beta=0.005,
            )
        loss = coordinate + 0.10 * mpjpe + temporal_loss_weight * temporal
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        coordinates.append(float(coordinate.detach().cpu()))
        mpjpes.append(float(mpjpe.detach().cpu()))
        temporals.append(float(temporal.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "coordinate": float(np.mean(coordinates)),
        "mpjpe_m": float(np.mean(mpjpes)),
        "temporal": float(np.mean(temporals)),
    }


def rewrite_metadata(output_dir: str, args: argparse.Namespace) -> None:
    for name in ("manifest.json", "result.json"):
        path = h19.Path(output_dir).resolve() / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        data["method"] = (
            "K96 identity anchor + ray-conditioned adaptive ST/TS dual-stream"
        )
        data["adaptation_boundary"] = {
            "paper_backbone": "UniCodebook adaptive spatial-temporal dual stream",
            "paper_interaction": "DCSA-style query/key/value cross attention",
            "our_conditioning": "selected-view RUMPL rays, confidence, and residuals",
            "not_claimed": "not a strict UniCodebook reproduction",
            "observation_mode": args.observation_mode,
            "coordinate_only_input": True,
            "image_features_or_heatmaps": False,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")


_ARGS: argparse.Namespace


def main() -> None:
    global _ARGS
    _ARGS = parse_args()
    # Reuse the audited H19 experiment driver without duplicating its split,
    # selection, refit, and final-evaluation logic.
    h19.parse_args = lambda: _ARGS
    h19.CausalSeq2SeqTemporalModel = RayConditionedDualStream
    h19.evaluate = evaluate
    h19.train_epoch = train_epoch
    original_main = h19.main
    original_main()
    rewrite_metadata(_ARGS.output_dir, _ARGS)


if __name__ == "__main__":
    main()
