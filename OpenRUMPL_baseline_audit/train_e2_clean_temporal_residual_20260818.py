#!/usr/bin/env python3
"""Full clean-data temporal control on the calibrated E2-C2 output.

This is the temporal experiment that the earlier H16 utility screen did not
perform.  The E2-C2 candidate generator and soft fusion are frozen.  A small
MixSTE-style factorized spatial/temporal encoder receives the *sequence of
fused 3-D poses* and predicts a zero-initialized center-frame residual.

The initial function is exactly the E2-C2 center pose.  The residual is
root-protected in this first control, so a gain cannot come from rewriting the
calibrated absolute translation.  Training uses all available clean windows
from S1/S5/S6/S7 and selects on complete S8 subject holdout; S9/S11 are read
only for the final report.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_COUNT = 11
JOINT_COUNT = 17
ACTION_NAMES = tuple(range(2, 17))


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
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--grad-accum-steps", type=int, default=1,
        help="Accumulate this many mini-batches before one optimizer update.",
    )
    p.add_argument(
        "--eval-batch-size", type=int, default=0,
        help="Evaluation batch size; 0 reuses --batch-size.",
    )
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--residual-scale-m", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--gpu", default="1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-holdout-windows", type=int, default=0)
    p.add_argument("--max-validation-windows", type=int, default=0)
    p.add_argument(
        "--train-uncertainty",
        help="Optional N,K,J,F label-free uncertainty feature array.",
    )
    p.add_argument(
        "--validation-uncertainty",
        help="Validation counterpart of --train-uncertainty.",
    )
    p.add_argument(
        "--uncertainty-gate", action="store_true",
        help="Condition and gate residuals with E2/ray uncertainty features.",
    )
    p.add_argument(
        "--stage-balanced-loss", action="store_true",
        help="Give V2, V3 and V4 equal loss weight instead of 6:4:1.",
    )
    p.add_argument(
        "--sequence-loss-weight", type=float, default=0.0,
        help="Additional all-nine-frame supervision weight; 0 keeps center-only training.",
    )
    p.add_argument(
        "--camera-independent", action="store_true",
        help=(
            "Use a center-frame pelvis/shoulder/torso coordinate system, "
            "remove camera-subset task embeddings, and rotate the residual "
            "back to world coordinates."
        ),
    )
    p.add_argument(
        "--continuous-time", action="store_true",
        help=(
            "Encode physical frame offsets in seconds and normalize temporal "
            "differences to --reference-dt-s. This removes the otherwise "
            "implicit dependence on the source dataset frame rate."
        ),
    )
    p.add_argument(
        "--source-fps", type=float, default=50.0,
        help="Frame rate used by image_id before temporal subsampling.",
    )
    p.add_argument(
        "--reference-dt-s", type=float, default=0.1,
        help=(
            "Reference interval for velocity/acceleration feature scale. "
            "0.1 s exactly preserves the legacy H36M stride-5 feature scale."
        ),
    )
    p.add_argument(
        "--max-time-period-s", type=float, default=2.0,
        help="Maximum period used by the continuous sinusoidal time encoder.",
    )
    p.add_argument(
        "--time-scale-min", type=float, default=1.0,
        help="Minimum sequence playback-time scale sampled during training.",
    )
    p.add_argument(
        "--time-scale-max", type=float, default=1.0,
        help="Maximum sequence playback-time scale sampled during training.",
    )
    p.add_argument(
        "--init-checkpoint",
        help=(
            "Optional H18 checkpoint used to initialize compatible weights. "
            "Legacy learned time positions are projected exactly onto the "
            "continuous encoder at the native training timestamps."
        ),
    )
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    keys = list(groups)
    bad = [(key, sorted(cams)) for key, cams in groups.items()
           if cams != {0, 1, 2, 3}]
    if bad:
        raise RuntimeError(f"incomplete four-view groups: {bad[:3]}")
    sequence_ids: list[int] = []
    seq_map: dict[tuple[int, int, int], int] = {}
    for subject, action, subaction, _ in keys:
        seq = (subject, action, subaction)
        seq_map.setdefault(seq, len(seq_map))
        sequence_ids.append(seq_map[seq])
    return {
        "subjects": np.asarray([x[0] for x in keys], dtype=np.int16),
        "actions": np.asarray([x[1] for x in keys], dtype=np.int16),
        "subactions": np.asarray([x[2] for x in keys], dtype=np.int16),
        "frame_ids": np.asarray([x[3] for x in keys], dtype=np.int64),
        "sequence_ids": np.asarray(sequence_ids, dtype=np.int32),
    }


def build_windows(meta: dict[str, np.ndarray], length: int, stride: int) -> np.ndarray:
    if length == 1:
        return np.arange(len(meta["frame_ids"]), dtype=np.int64)[:, None]
    if length < 3 or length % 2 != 1:
        raise ValueError("window length must be an odd integer >= 1")
    by_sequence: dict[int, list[int]] = collections.defaultdict(list)
    for index, sequence in enumerate(meta["sequence_ids"].tolist()):
        by_sequence[int(sequence)].append(index)
    rows: list[list[int]] = []
    for indices in by_sequence.values():
        indices.sort(key=lambda i: int(meta["frame_ids"][i]))
        for start in range(0, len(indices) - length + 1):
            row = indices[start:start + length]
            if np.all(np.diff(meta["frame_ids"][row]) == stride):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no T={length} windows with stride={stride}")
    return np.asarray(rows, dtype=np.int64)


class ContinuousTimeEncoder(nn.Module):
    """Continuous sinusoidal encoding of physical offsets measured in seconds."""

    def __init__(self, hidden_dim: int, max_period_s: float = 2.0):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("continuous time encoding requires an even hidden-dim")
        if max_period_s <= 0.0:
            raise ValueError("max-time-period-s must be positive")
        half = hidden_dim // 2
        # Cover both native video-frame timing and the complete temporal
        # context.  Angular log-spaced frequencies provide enough rank to
        # reproduce a legacy learned T-position table while remaining a
        # continuous function for unseen frame rates and irregular sampling.
        min_period_s = min(0.02, max_period_s)
        periods = torch.exp(torch.linspace(
            math.log(min_period_s), math.log(max_period_s), half,
            dtype=torch.float32,
        ))
        frequencies = (2.0 * math.pi) / periods
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, delta_t_s: torch.Tensor) -> torch.Tensor:
        if delta_t_s.ndim != 2:
            raise ValueError(
                f"expected physical time offsets B,T, got {tuple(delta_t_s.shape)}"
            )
        phase = delta_t_s[..., None] * self.frequencies
        return self.projection(torch.cat((phase.sin(), phase.cos()), dim=-1))


class TemporalPoseModel(nn.Module):
    """MixSTE-style spatial/temporal factorization with identity warm start."""

    def __init__(self, window_length: int, hidden_dim: int, layers: int,
                 residual_scale_m: float, task_count: int = TASK_COUNT,
                 camera_independent: bool = False,
                 continuous_time: bool = False,
                 reference_dt_s: float = 0.1,
                 max_time_period_s: float = 2.0,
                 uncertainty_dim: int = 0,
                 uncertainty_gate: bool = False):
        super().__init__()
        if hidden_dim % 8:
            raise ValueError("hidden-dim must be divisible by 8")
        self.window_length = window_length
        self.residual_scale_m = float(residual_scale_m)
        self.camera_independent = bool(camera_independent)
        self.continuous_time = bool(continuous_time)
        self.uncertainty_dim = int(uncertainty_dim)
        self.uncertainty_gate_enabled = bool(uncertainty_gate)
        if self.uncertainty_gate_enabled and self.uncertainty_dim < 1:
            raise ValueError("uncertainty gate requires uncertainty features")
        self.reference_dt_s = float(reference_dt_s)
        if self.reference_dt_s <= 0.0:
            raise ValueError("reference-dt-s must be positive")
        self.input = nn.Sequential(
            nn.Linear(12 + self.uncertainty_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.joint_embedding = nn.Parameter(torch.zeros(JOINT_COUNT, hidden_dim))
        self.task_embedding = nn.Embedding(task_count, hidden_dim)
        if self.continuous_time:
            self.time_encoder = ContinuousTimeEncoder(hidden_dim, max_time_period_s)
        else:
            # Keep the legacy parameter name and shape so old checkpoints load
            # strictly and the existing formal run remains exactly reproducible.
            self.time_embedding = nn.Parameter(
                torch.zeros(window_length, hidden_dim)
            )
        self.spatial = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim,
                dropout=0.0, activation="gelu", batch_first=True,
                norm_first=True,
            ) for _ in range(layers)
        ])
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 3))
        self.uncertainty_gate = (
            nn.Sequential(
                nn.Linear(self.uncertainty_dim, hidden_dim // 2), nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if self.uncertainty_gate_enabled else None
        )
        # Exact identity around frozen E2-C2 at step zero.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        if self.uncertainty_gate is not None:
            nn.init.zeros_(self.uncertainty_gate[-1].weight)
            nn.init.zeros_(self.uncertainty_gate[-1].bias)
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        nn.init.trunc_normal_(self.task_embedding.weight, std=0.02)

    @staticmethod
    def body_canonical_window(pose: torch.Tensor):
        """Canonicalize B,T,K,J,3 poses with center-frame body axes."""
        center = pose[:, pose.shape[1] // 2]
        origin = center[:, :, 0]
        x_axis = F.normalize(
            center[:, :, 14] - center[:, :, 11], dim=-1, eps=1e-7
        )
        up_hint = center[:, :, 8] - origin
        y_axis = up_hint - (
            up_hint * x_axis
        ).sum(dim=-1, keepdim=True) * x_axis
        y_axis = F.normalize(y_axis, dim=-1, eps=1e-7)
        z_axis = F.normalize(
            torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-7
        )
        y_axis = F.normalize(
            torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=1e-7
        )
        # Columns are canonical basis vectors expressed in world coordinates.
        basis = torch.stack((x_axis, y_axis, z_axis), dim=-1)
        centered = pose - origin[:, None, :, None, :]
        canonical = torch.einsum("btkji,bkic->btkjc", centered, basis)
        return canonical, basis

    def motion_features(
        self, model_pose: torch.Tensor, delta_t_s: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return velocity/acceleration at a dataset-independent time scale."""
        if not self.continuous_time:
            velocity = torch.diff(
                model_pose, dim=1, prepend=model_pose[:, :1]
            )
            acceleration = torch.diff(
                velocity, dim=1, prepend=velocity[:, :1]
            )
            return velocity, acceleration
        if delta_t_s is None:
            raise ValueError("continuous-time model requires physical delta_t_s")
        if delta_t_s.shape != model_pose.shape[:2]:
            raise ValueError(
                "delta_t_s must match the B,T dimensions of the pose window"
            )
        step_s = torch.diff(delta_t_s, dim=1)
        if torch.any(step_s < 0.0):
            raise ValueError(
                "physical timestamps must not decrease within each window"
            )
        # Divide by dt/reference_dt rather than dt itself.  Features therefore
        # represent displacement per reference interval and retain the legacy
        # H36M scale exactly when stride 5 at 50 Hz gives dt=0.1 seconds.
        normalized_step = step_s / self.reference_dt_s
        valid_step = normalized_step > 1e-8
        safe_step = torch.where(
            valid_step, normalized_step, torch.ones_like(normalized_step)
        )
        velocity = torch.zeros_like(model_pose)
        velocity[:, 1:] = torch.where(
            valid_step[:, :, None, None, None],
            torch.diff(model_pose, dim=1)
            / safe_step[:, :, None, None, None],
            torch.zeros_like(model_pose[:, 1:]),
        )
        acceleration = torch.zeros_like(model_pose)
        if model_pose.shape[1] > 1:
            # Keep the legacy left-boundary sentinel, but use the already
            # normalized velocity so it no longer depends on source FPS.
            acceleration[:, 1] = velocity[:, 1]
        if model_pose.shape[1] > 2:
            acceleration[:, 2:] = torch.where(
                valid_step[:, 1:, None, None, None],
                torch.diff(velocity, dim=1)[:, 1:]
                / safe_step[:, 1:, None, None, None],
                torch.zeros_like(model_pose[:, 2:]),
            )
        return velocity, acceleration

    def forward(self, pose: torch.Tensor, task_ids: torch.Tensor,
                delta_t_s: torch.Tensor | None = None,
                uncertainty: torch.Tensor | None = None,
                return_sequence: bool = False) -> torch.Tensor:
        # pose: B,T,K,J,3; task_ids: B,K
        if pose.ndim != 5:
            raise ValueError(f"expected B,T,K,J,3, got {tuple(pose.shape)}")
        batch, time, tasks, joints, channels = pose.shape
        if time != self.window_length or joints != JOINT_COUNT or channels != 3:
            raise ValueError("unexpected temporal pose shape")
        model_pose = pose
        body_basis = None
        if self.camera_independent:
            model_pose, body_basis = self.body_canonical_window(pose)
        root = model_pose[:, :, :, :1]
        relative = model_pose - root
        velocity, acceleration = self.motion_features(model_pose, delta_t_s)
        # Absolute root is retained as a conditioning signal, while the
        # output remains root-protected in this clean control.
        features = torch.cat(
            (relative, root.expand(-1, -1, -1, joints, -1), velocity, acceleration),
            dim=-1,
        )
        if self.uncertainty_dim:
            if uncertainty is None or uncertainty.shape != (
                batch, time, tasks, joints, self.uncertainty_dim
            ):
                raise ValueError(
                    "uncertainty must have shape B,T,K,J,F matching the model"
                )
            features = torch.cat((features, uncertainty), dim=-1)
        x = self.input(features)
        x = x + self.joint_embedding[None, None, None]
        if not self.camera_independent:
            x = x + self.task_embedding(task_ids)[:, None, :, None]
        if self.continuous_time:
            if delta_t_s is None:
                raise ValueError("continuous-time model requires physical delta_t_s")
            time_embedding = self.time_encoder(delta_t_s)
            x = x + time_embedding[:, :, None, None]
        else:
            x = x + self.time_embedding[None, :, None, None]

        # Spatial block (per frame/task), then temporal block (per joint/task).
        x = x.reshape(batch * time * tasks, joints, -1)
        for block in self.spatial:
            x = block(x)
        x = x.reshape(batch, time, tasks, joints, -1)
        x = x.permute(0, 2, 3, 1, 4).reshape(batch * tasks * joints, time, -1)
        for block in self.temporal:
            x = block(x)
        x = x.reshape(batch, tasks, joints, time, -1)
        if return_sequence:
            x = x.permute(0, 3, 1, 2, 4)
            delta = self.residual_scale_m * torch.tanh(self.output(x))
            if self.uncertainty_gate is not None:
                delta = delta * 2.0 * torch.sigmoid(
                    self.uncertainty_gate(uncertainty)
                )
            delta = delta.clone()
            delta[:, :, :, 0] = 0.0
            if body_basis is not None:
                delta = torch.einsum("btkjc,bkic->btkji", delta, body_basis)
            return pose + delta
        x = x[:, :, :, time // 2]
        delta = self.residual_scale_m * torch.tanh(self.output(x))
        if self.uncertainty_gate is not None:
            delta = delta * 2.0 * torch.sigmoid(
                self.uncertainty_gate(uncertainty[:, time // 2])
            )
        delta = delta.clone()
        delta[:, :, 0] = 0.0  # preserve E2-C2 absolute root in this control
        if body_basis is not None:
            delta = torch.einsum("bkjc,bkic->bkji", delta, body_basis)
        return pose[:, time // 2] + delta


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES
        if np.any(actions == action)
    ]))


def gather_batch(cache: np.lib.npyio.NpzFile | dict, rows: np.ndarray,
                 fused: np.ndarray, center: int, device: torch.device):
    pose = torch.from_numpy(np.asarray(fused[rows])).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(np.asarray(cache["targets"][rows[:, center]])).to(
        device=device, dtype=torch.float32
    )
    task_ids = torch.arange(TASK_COUNT, device=device)[None].expand(len(rows), -1)
    return pose, target, task_ids


def gather_uncertainty(uncertainty: np.ndarray | None, rows: np.ndarray,
                       device: torch.device) -> torch.Tensor | None:
    if uncertainty is None:
        return None
    return torch.from_numpy(np.asarray(uncertainty[rows])).to(
        device=device, dtype=torch.float32
    )


def physical_time_offsets(
    frame_ids: np.ndarray, rows: np.ndarray, source_fps: float,
    device: torch.device,
) -> torch.Tensor:
    """Build center-relative physical timestamps directly from source IDs."""
    if source_fps <= 0.0:
        raise ValueError("source-fps must be positive")
    ids = np.asarray(frame_ids[rows], dtype=np.float64)
    ids = ids - ids[:, ids.shape[1] // 2:ids.shape[1] // 2 + 1]
    return torch.from_numpy(ids / source_fps).to(
        device=device, dtype=torch.float32
    )


def load_initial_checkpoint(
    model: TemporalPoseModel, checkpoint_path: str, frame_stride: int,
    source_fps: float, device: torch.device,
) -> dict:
    """Load an H18 checkpoint, converting legacy time positions if needed."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = state["state_dict"]
    compatible = {
        name: value for name, value in source.items()
        if name in model.state_dict()
        and model.state_dict()[name].shape == value.shape
    }
    partial_input = False
    if "input.0.weight" in source and "input.0.weight" in model.state_dict():
        old = source["input.0.weight"]
        new = model.state_dict()["input.0.weight"].clone()
        if old.shape[0] == new.shape[0] and old.shape[1] <= new.shape[1]:
            new.zero_()
            new[:, :old.shape[1]].copy_(old)
            compatible["input.0.weight"] = new
            partial_input = old.shape != new.shape
    incompatible = model.load_state_dict(compatible, strict=False)
    unexpected = [name for name in incompatible.unexpected_keys]
    if unexpected:
        raise RuntimeError(f"unexpected initialization keys: {unexpected}")

    conversion_error = None
    if model.continuous_time and "time_embedding" in source:
        half = model.window_length // 2
        delta_t_s = (
            torch.arange(-half, half + 1, device=device, dtype=torch.float32)
            * (frame_stride / source_fps)
        )
        frequencies = model.time_encoder.frequencies
        phase = delta_t_s[:, None] * frequencies[None]
        raw = torch.cat((phase.sin(), phase.cos()), dim=-1)
        design = torch.cat(
            (raw, torch.ones(len(raw), 1, device=device)), dim=-1
        )
        target = source["time_embedding"].to(device=device, dtype=raw.dtype)
        # Solve in float64 because this is a wide (T x [D+1]) exact
        # interpolation problem; the pseudoinverse gives the stable
        # minimum-norm projection for all legacy temporal positions.
        solution = (
            torch.linalg.pinv(design.double(), rtol=1e-10)
            @ target.double()
        ).to(dtype=raw.dtype)
        with torch.no_grad():
            model.time_encoder.projection.weight.copy_(solution[:-1].T)
            model.time_encoder.projection.bias.copy_(solution[-1])
        reconstructed = model.time_encoder(delta_t_s[None])[0]
        conversion_error = float((reconstructed - target).abs().max().cpu())
        if conversion_error > 1e-4:
            raise RuntimeError(
                f"legacy time-embedding conversion error {conversion_error}"
            )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "source_epoch": int(state.get("epoch", -1)),
        "source_best_holdout_metric_mm": state.get("best_holdout_metric_mm"),
        "loaded_parameter_keys": len(compatible),
        "partially_extended_input_projection": partial_input,
        "missing_parameter_keys": list(incompatible.missing_keys),
        "legacy_time_conversion_max_abs": conversion_error,
    }


@torch.inference_mode()
def evaluate(model: nn.Module | None, cache, fused: np.ndarray,
             windows: np.ndarray, device: torch.device, center: int,
             batch_size: int, frame_ids: np.ndarray | None = None,
             source_fps: float | None = None,
             uncertainty: np.ndarray | None = None) -> dict:
    base_by_stage = {f"V{k}": [] for k in (2, 3, 4)}
    pred_by_stage = {f"V{k}": [] for k in (2, 3, 4)}
    action_by_stage = {f"V{k}": [] for k in (2, 3, 4)}
    if model is not None:
        model.eval()
    for start in range(0, len(windows), batch_size):
        rows = windows[start:start + batch_size]
        pose, target, task_ids = gather_batch(cache, rows, fused, center, device)
        uncertainty_batch = gather_uncertainty(uncertainty, rows, device)
        baseline = pose[:, center]
        delta_t_s = None
        if model is not None and getattr(model, "continuous_time", False):
            if frame_ids is None or source_fps is None:
                raise ValueError(
                    "continuous-time evaluation requires frame_ids and source_fps"
                )
            delta_t_s = physical_time_offsets(
                frame_ids, rows, source_fps, device
            )
        prediction = (
            baseline if model is None else model(
                pose, task_ids, delta_t_s, uncertainty_batch
            )
        )
        base_err = torch.linalg.vector_norm(baseline - target[:, None], dim=-1)
        pred_err = torch.linalg.vector_norm(prediction - target[:, None], dim=-1)
        actions = np.asarray(cache["actions"][rows[:, center]])
        for task_index, count in enumerate((2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4)):
            stage = f"V{count}"
            base_by_stage[stage].append(base_err[:, task_index].cpu().numpy() * 1000.0)
            pred_by_stage[stage].append(pred_err[:, task_index].cpu().numpy() * 1000.0)
            action_by_stage[stage].append(actions.copy())
    result = {}
    stage_metrics = []
    for stage in ("V2", "V3", "V4"):
        base = np.concatenate(base_by_stage[stage])
        pred = np.concatenate(pred_by_stage[stage])
        actions = np.concatenate(action_by_stage[stage])
        b = action_equal(base, actions)
        t = action_equal(pred, actions)
        result[stage] = {
            "baseline_action_equal_all17_mm": float(b),
            "temporal_action_equal_all17_mm": float(t),
            "delta_mm": float(t - b),
            "baseline_frame_weighted_all17_mm": float(base.mean()),
            "temporal_frame_weighted_all17_mm": float(pred.mean()),
        }
        stage_metrics.append(t)
    result["mean_v234_mm"] = float(np.mean(stage_metrics))
    result["windows"] = int(len(windows))
    return result


def select_rows(rows: np.ndarray, max_rows: int) -> np.ndarray:
    if max_rows and len(rows) > max_rows:
        return rows[np.linspace(0, len(rows) - 1, max_rows, dtype=np.int64)]
    return rows


def regression_loss(prediction: torch.Tensor, target: torch.Tensor,
                    stage_balanced: bool) -> torch.Tensor:
    """Matched smooth-L1/MPJPE objective with optional equal stage weight."""
    task_dim = prediction.ndim - 3
    groups = ((0, 6), (6, 10), (10, 11)) if stage_balanced else ((0, 11),)
    losses = []
    for start, stop in groups:
        indices = torch.arange(start, stop, device=prediction.device)
        pred = torch.index_select(prediction, task_dim, indices)
        truth = torch.index_select(target, task_dim, indices)
        coord = F.smooth_l1_loss(pred, truth, beta=0.01)
        mpjpe = torch.linalg.vector_norm(pred - truth, dim=-1).mean()
        losses.append(coord + 0.10 * mpjpe)
    return torch.stack(losses).mean()


def main() -> None:
    args = parse_args()
    if args.window_length % 2 != 1:
        raise ValueError("window-length must be odd")
    if args.grad_accum_steps < 1:
        raise ValueError("grad-accum-steps must be >= 1")
    if args.eval_batch_size < 0:
        raise ValueError("eval-batch-size must be >= 0")
    if args.source_fps <= 0.0:
        raise ValueError("source-fps must be positive")
    if args.reference_dt_s <= 0.0:
        raise ValueError("reference-dt-s must be positive")
    if not 0.0 < args.time_scale_min <= args.time_scale_max:
        raise ValueError("require 0 < time-scale-min <= time-scale-max")
    if not args.continuous_time and (
        args.time_scale_min != 1.0 or args.time_scale_max != 1.0
    ):
        raise ValueError("time-scale augmentation requires --continuous-time")
    if args.sequence_loss_weight < 0.0:
        raise ValueError("sequence-loss-weight must be non-negative")
    if bool(args.train_uncertainty) != bool(args.validation_uncertainty):
        raise ValueError("train/validation uncertainty arrays must be provided together")
    if args.uncertainty_gate and not args.train_uncertainty:
        raise ValueError("--uncertainty-gate requires uncertainty arrays")
    eval_batch_size = args.eval_batch_size or args.batch_size
    seed_everything(args.seed)
    train_cache = np.load(args.train_cache, allow_pickle=False)
    val_cache = np.load(args.validation_cache, allow_pickle=False)
    train_fused = np.load(args.train_fused, mmap_mode="r")
    val_fused = np.load(args.validation_fused, mmap_mode="r")
    train_uncertainty = (
        np.load(args.train_uncertainty, mmap_mode="r")
        if args.train_uncertainty else None
    )
    val_uncertainty = (
        np.load(args.validation_uncertainty, mmap_mode="r")
        if args.validation_uncertainty else None
    )
    if train_fused.shape[1:] != (TASK_COUNT, JOINT_COUNT, 3):
        raise ValueError(f"bad train fused shape {train_fused.shape}")
    if val_fused.shape[1:] != (TASK_COUNT, JOINT_COUNT, 3):
        raise ValueError(f"bad validation fused shape {val_fused.shape}")
    uncertainty_dim = 0
    if train_uncertainty is not None:
        if train_uncertainty.shape[:3] != (
            len(train_fused), TASK_COUNT, JOINT_COUNT
        ):
            raise ValueError(f"bad train uncertainty shape {train_uncertainty.shape}")
        if val_uncertainty.shape[:3] != (
            len(val_fused), TASK_COUNT, JOINT_COUNT
        ):
            raise ValueError(f"bad validation uncertainty shape {val_uncertainty.shape}")
        if train_uncertainty.shape[-1] != val_uncertainty.shape[-1]:
            raise ValueError("train/validation uncertainty dimensions differ")
        uncertainty_dim = int(train_uncertainty.shape[-1])
    train_meta = metadata_from_pkl(args.train_pkl, len(train_cache["targets"]))
    val_meta = metadata_from_pkl(args.validation_pkl, len(val_cache["targets"]))
    for name, cache, meta in (("train", train_cache, train_meta),
                              ("validation", val_cache, val_meta)):
        if not np.array_equal(cache["subjects"], meta["subjects"]):
            raise RuntimeError(f"{name} subject order mismatch")
        if not np.array_equal(cache["actions"], meta["actions"]):
            raise RuntimeError(f"{name} action order mismatch")

    all_train = build_windows(train_meta, args.window_length, args.frame_stride)
    all_val = build_windows(val_meta, args.window_length, args.frame_stride)
    centers = all_train[:, args.window_length // 2]
    train_windows = all_train[np.isin(train_meta["subjects"][centers], [1, 5, 6, 7])]
    holdout_windows = all_train[train_meta["subjects"][centers] == 8]
    train_windows = select_rows(train_windows, args.max_train_windows)
    holdout_windows = select_rows(holdout_windows, args.max_holdout_windows)
    val_windows = select_rows(all_val, args.max_validation_windows)
    if not len(train_windows) or not len(holdout_windows) or not len(val_windows):
        raise RuntimeError("empty train, holdout, or validation windows")

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "full clean MixSTE-style E2-C2 temporal pose residual",
        "train_cache": str(Path(args.train_cache).resolve()),
        "train_fused": str(Path(args.train_fused).resolve()),
        "validation_cache": str(Path(args.validation_cache).resolve()),
        "validation_fused": str(Path(args.validation_fused).resolve()),
        "train_pkl": str(Path(args.train_pkl).resolve()),
        "validation_pkl": str(Path(args.validation_pkl).resolve()),
        "window_length": args.window_length,
        "frame_stride": args.frame_stride,
        "train_subjects": [1, 5, 6, 7],
        "holdout_subjects": [8],
        "train_windows": int(len(train_windows)),
        "holdout_windows": int(len(holdout_windows)),
        "validation_windows": int(len(val_windows)),
        "root_protected": True,
        "camera_independent": args.camera_independent,
        "continuous_time": args.continuous_time,
        "source_fps": args.source_fps,
        "reference_dt_s": args.reference_dt_s,
        "time_scale_range": [args.time_scale_min, args.time_scale_max],
        "uncertainty_dim": uncertainty_dim,
        "uncertainty_gate": args.uncertainty_gate,
        "stage_balanced_loss": args.stage_balanced_loss,
        "sequence_loss_weight": args.sequence_loss_weight,
        "clean_training_only": True,
        "args": vars(args),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    center = args.window_length // 2
    baseline_holdout = evaluate(None, train_cache, train_fused, holdout_windows,
                                device, center, eval_batch_size)
    baseline_val = evaluate(None, val_cache, val_fused, val_windows,
                            device, center, eval_batch_size)
    (out / "baseline_holdout.json").write_text(json.dumps(baseline_holdout, indent=2) + "\n")
    (out / "baseline_validation.json").write_text(json.dumps(baseline_val, indent=2) + "\n")
    print(json.dumps({"baseline_holdout": baseline_holdout,
                      "baseline_validation": baseline_val}, indent=2), flush=True)

    model = TemporalPoseModel(
        args.window_length, args.hidden_dim, args.layers, args.residual_scale_m,
        camera_independent=args.camera_independent,
        continuous_time=args.continuous_time,
        reference_dt_s=args.reference_dt_s,
        max_time_period_s=args.max_time_period_s,
        uncertainty_dim=uncertainty_dim,
        uncertainty_gate=args.uncertainty_gate,
    ).to(device)
    initialization = None
    if args.init_checkpoint:
        initialization = load_initial_checkpoint(
            model, args.init_checkpoint, args.frame_stride, args.source_fps,
            device,
        )
        manifest["initialization"] = initialization
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best_metric = float(baseline_holdout["mean_v234_mm"])
    best_epoch = -1
    best_checkpoint_source = "baseline"
    has_best_model = False
    history = []
    if initialization is not None:
        initial_holdout = evaluate(
            model, train_cache, train_fused, holdout_windows, device, center,
            eval_batch_size, train_meta["frame_ids"], args.source_fps,
            train_uncertainty,
        )
        initial_metric = float(initial_holdout["mean_v234_mm"])
        print(json.dumps({"initialization": initialization,
                          "initial_holdout": initial_holdout}), flush=True)
        if initial_metric < best_metric:
            best_metric = initial_metric
            best_checkpoint_source = "initialization"
            has_best_model = True
            torch.save({"state_dict": model.state_dict(), "epoch": -1,
                        "best_holdout_metric_mm": best_metric,
                        "args": vars(args)}, out / "model_best.pth.tar")
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_windows))
        losses = []
        offsets = list(range(0, len(order), args.batch_size))
        optimizer.zero_grad(set_to_none=True)
        for batch_index, offset in enumerate(offsets):
            rows = train_windows[order[offset:offset + args.batch_size]]
            pose, target, task_ids = gather_batch(train_cache, rows, train_fused,
                                                   center, device)
            uncertainty_batch = gather_uncertainty(
                train_uncertainty, rows, device
            )
            delta_t_s = (
                physical_time_offsets(
                    train_meta["frame_ids"], rows, args.source_fps, device
                )
                if args.continuous_time else None
            )
            if args.continuous_time and (
                args.time_scale_min != 1.0 or args.time_scale_max != 1.0
            ):
                log_min = math.log(args.time_scale_min)
                log_max = math.log(args.time_scale_max)
                time_scale = torch.empty(
                    len(rows), 1, device=device
                ).uniform_(log_min, log_max).exp_()
                delta_t_s = delta_t_s * time_scale
            if batch_index % args.grad_accum_steps == 0:
                group_end = min(batch_index + args.grad_accum_steps, len(offsets))
                group_samples = sum(
                    min(args.batch_size, len(order) - offsets[index])
                    for index in range(batch_index, group_end)
                )
            if args.sequence_loss_weight > 0.0:
                prediction_sequence = model(
                    pose, task_ids, delta_t_s, uncertainty_batch,
                    return_sequence=True,
                )
                prediction = prediction_sequence[:, center]
                target_sequence = torch.from_numpy(np.asarray(
                    train_cache["targets"][rows]
                )).to(device=device, dtype=torch.float32)
                target_sequence = target_sequence[:, :, None].expand_as(
                    prediction_sequence
                )
            else:
                prediction = model(
                    pose, task_ids, delta_t_s, uncertainty_batch
                )
                target_sequence = None
            target_expanded = target[:, None].expand_as(prediction)
            loss = regression_loss(
                prediction, target_expanded, args.stage_balanced_loss
            )
            if target_sequence is not None:
                loss = loss + args.sequence_loss_weight * regression_loss(
                    prediction_sequence, target_sequence,
                    args.stage_balanced_loss,
                )
            # Scale each mini-batch by its sample count so accumulation matches
            # a single mean loss over the effective batch, including the tail.
            (loss * (len(rows) / group_samples)).backward()
            is_group_end = (
                (batch_index + 1) % args.grad_accum_steps == 0
                or batch_index + 1 == len(offsets)
            )
            if is_group_end:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        holdout = evaluate(model, train_cache, train_fused, holdout_windows,
                           device, center, eval_batch_size,
                           train_meta["frame_ids"], args.source_fps,
                           train_uncertainty)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": holdout["mean_v234_mm"],
            "holdout": holdout,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if holdout["mean_v234_mm"] < best_metric:
            best_metric = float(holdout["mean_v234_mm"])
            best_epoch = epoch
            best_checkpoint_source = f"epoch-{epoch}"
            has_best_model = True
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "best_holdout_metric_mm": best_metric,
                        "args": vars(args)}, out / "model_best.pth.tar")

    if has_best_model:
        state = torch.load(out / "model_best.pth.tar", map_location=device,
                           weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        final_val = evaluate(model, val_cache, val_fused, val_windows,
                             device, center, eval_batch_size,
                             val_meta["frame_ids"], args.source_fps,
                             val_uncertainty)
    else:
        final_val = baseline_val
    result = {
        **manifest,
        "baseline_holdout": baseline_holdout,
        "baseline_validation": baseline_val,
        "best_epoch": best_epoch,
        "best_checkpoint_source": best_checkpoint_source,
        "best_holdout_metric_mm": best_metric,
        "history": history,
        "S9_S11_final_once": final_val,
        "decision": (
            "clean temporal branch retained for occlusion-gated follow-up"
            if has_best_model and final_val["mean_v234_mm"] < baseline_val["mean_v234_mm"]
            else "clean temporal branch did not improve; do not add occlusion gate"
        ),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "COMPLETED").write_text("completed\n")
    print(json.dumps({"S9_S11_final_once": final_val,
                      "decision": result["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
