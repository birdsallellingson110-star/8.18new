#!/usr/bin/env python3
"""Train a failure-informed uncertainty-aware ray-MAP decoder.

The frozen K96 limb-proposal model supplies the current best coordinate-only
pose ``X0``.  A permutation-equivariant joint-view encoder predicts a human
pose prior and a selective trust gate.  The final pose is obtained through an
explicit point-to-ray MAP solve, rather than an unconstrained residual MLP.

M1 deliberately keeps HRNet confidence as the observation weight.  Learned
heteroscedastic observation precision is implemented behind an explicit flag
for the later M2 ablation, but is disabled by default.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_limb_utility_20260820 as limb_utility
import train_e2_pose_dsac_20260820 as dsac
import train_e2_v234_universal_20260812 as trainer
from diagnose_e2_structured_candidates_20260820 import train_bone_stats
from train_h76_hypothesis_utility_20260811 import ACTION_NAMES, ArrayDataset


TASKS = tuple(
    combo for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--e2-checkpoint", required=True)
    parser.add_argument("--proposal-checkpoint", required=True)
    parser.add_argument("--k96-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--prior-precision", type=float, default=30.0)
    parser.add_argument("--max-prior-delta-m", type=float, default=0.15)
    parser.add_argument("--trust-bias", type=float, default=-4.0)
    parser.add_argument("--relative-loss-weight", type=float, default=0.25)
    parser.add_argument(
        "--learn-observation-precision", action="store_true",
        help="M2 ablation only; M1 must leave this disabled.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--holdout-subject", type=int, default=0,
        help=(
            "If non-zero, select checkpoints on this entirely unseen training "
            "subject instead of the historical group_index modulo holdout."
        ),
    )
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--smoke-validation-batches", type=int, default=0)
    return parser.parse_args()


def zero_last_linear(module: nn.Sequential, bias: float = 0.0) -> None:
    layer = next(item for item in reversed(module) if isinstance(item, nn.Linear))
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, bias)


class JointViewMAPDecoder(nn.Module):
    """Predict probabilistic factors, then solve the explicit ray-MAP system."""

    def __init__(
        self,
        d_model: int = 96,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.0,
        prior_precision: float = 30.0,
        max_prior_delta_m: float = 0.15,
        trust_bias: float = -4.0,
        learn_observation_precision: bool = False,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by attention heads")
        self.prior_precision = float(prior_precision)
        self.max_prior_delta_m = float(max_prior_delta_m)
        self.learn_observation_precision = learn_observation_precision
        # direction(3), root-centred camera origin(3), confidence(1),
        # anchor-to-ray perpendicular vector(3), root-relative anchor(3),
        # ray depth(1), normalized information eigenvalues(3), view count(1).
        input_size = 18
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        self.joint_embedding = nn.Parameter(torch.zeros(17, d_model))
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.relative_prior_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.root_prior_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 3),
        )
        self.prior_precision_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1),
        )
        self.trust_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 1),
        )
        zero_last_linear(self.relative_prior_head)
        zero_last_linear(self.root_prior_head)
        zero_last_linear(self.prior_precision_head)
        zero_last_linear(self.trust_head, trust_bias)
        if learn_observation_precision:
            self.observation_precision_head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, 1),
            )
            zero_last_linear(self.observation_precision_head)
        else:
            self.observation_precision_head = None

    @staticmethod
    def ray_projectors(direction: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(3, device=direction.device, dtype=direction.dtype)
        return eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)

    def build_features(
        self, anchor: torch.Tensor, selected_rays: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        direction = F.normalize(selected_rays[..., :3], dim=-1)
        origin = selected_rays[..., 3:6]
        confidence = selected_rays[..., 6:7].clamp(0.01, 1.0)
        projectors = self.ray_projectors(direction)
        anchor_root = anchor[:, :1]
        anchor_relative = anchor - anchor_root
        offset = anchor[:, :, None] - origin
        perpendicular = torch.einsum("njvac,njvc->njva", projectors, offset)
        depth = (offset * direction).sum(dim=-1, keepdim=True)
        fixed_weight = confidence.squeeze(-1)
        information = (
            fixed_weight[..., None, None] * projectors
        ).sum(dim=2)
        eigenvalues = torch.linalg.eigvalsh(information).clamp_min(1e-6)
        view_count = selected_rays.shape[2]
        feature = torch.cat(
            (
                direction,
                (origin - anchor_root[:, :, None]) / 5.0,
                confidence,
                perpendicular / 0.10,
                anchor_relative[:, :, None].expand(-1, -1, view_count, -1),
                depth / 5.0,
                (eigenvalues / float(view_count))[:, :, None].expand(
                    -1, -1, view_count, -1
                ),
                anchor.new_full(
                    (anchor.shape[0], 17, view_count, 1), view_count / 4.0
                ),
            ),
            dim=-1,
        )
        return feature, {
            "direction": direction,
            "origin": origin,
            "confidence": confidence,
            "projectors": projectors,
            "perpendicular": perpendicular,
            "information_eigenvalues": eigenvalues,
        }

    def forward(
        self, anchor: torch.Tensor, selected_rays: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature, geometry = self.build_features(anchor, selected_rays)
        batch, joints, views, _ = feature.shape
        token = self.input_projection(feature)
        token = token + self.joint_embedding[None, :, None]
        contextual = self.encoder(token.reshape(batch, joints * views, -1))
        contextual = contextual.reshape(batch, joints, views, -1)
        confidence = geometry["confidence"]
        joint_context = (
            contextual * confidence
        ).sum(dim=2) / confidence.sum(dim=2).clamp_min(1e-6)
        global_context = joint_context.mean(dim=1)

        relative_delta = self.max_prior_delta_m * torch.tanh(
            self.relative_prior_head(joint_context)
        )
        # Pelvis is represented by the separate absolute root prior.
        relative_delta = relative_delta.clone()
        relative_delta[:, 0] = 0.0
        root_delta = self.max_prior_delta_m * torch.tanh(
            self.root_prior_head(global_context)
        )
        anchor_root = anchor[:, :1]
        anchor_relative = anchor - anchor_root
        prior_mean = (
            anchor_root + root_delta[:, None]
            + anchor_relative + relative_delta
        )
        log_prior_scale = 2.0 * torch.tanh(
            self.prior_precision_head(joint_context).squeeze(-1)
        )
        prior_precision = self.prior_precision * torch.exp(log_prior_scale)

        observation_weight = confidence.squeeze(-1)
        if self.observation_precision_head is not None:
            relative_log_weight = 2.0 * torch.tanh(
                self.observation_precision_head(contextual).squeeze(-1)
            )
            observation_weight = observation_weight * torch.exp(relative_log_weight)

        projectors = geometry["projectors"]
        origin = geometry["origin"]
        weighted_projector = observation_weight[..., None, None] * projectors
        normal = weighted_projector.sum(dim=2)
        rhs_observation = torch.einsum(
            "njvac,njvc->nja", weighted_projector, origin
        )
        eye = torch.eye(3, device=anchor.device, dtype=anchor.dtype)
        system = normal + prior_precision[..., None, None] * eye
        rhs = rhs_observation + prior_precision[..., None] * prior_mean
        map_pose = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)

        trust = torch.sigmoid(self.trust_head(joint_context).squeeze(-1))
        output = anchor + trust[..., None] * (map_pose - anchor)
        auxiliary = {
            **geometry,
            "prior_mean": prior_mean,
            "prior_precision": prior_precision,
            "observation_weight": observation_weight,
            "map_pose": map_pose,
            "trust": trust,
            "root_delta": root_delta,
            "relative_delta": relative_delta,
        }
        return output, auxiliary


class FrozenK96Anchor:
    """Exact GT-free inference stack used by the current best K96 experiment."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        self.e2, self.coord_mean, self.coord_std = dsac.load_e2(
            args.e2_checkpoint, device
        )
        self.bone_mean, self.bone_std = train_bone_stats(args.train_cache)
        self.bone_mean = self.bone_mean.to(device)
        self.bone_std = self.bone_std.to(device)
        proposal_state = torch.load(
            args.proposal_checkpoint, map_location=device, weights_only=False
        )
        self.proposal = limb_utility.LimbUtility(
            self.coord_mean, self.coord_std
        ).to(device)
        self.proposal.load_state_dict(proposal_state["state_dict"], strict=True)
        self.proposal.eval().requires_grad_(False)
        k96_state = torch.load(
            args.k96_checkpoint, map_location=device, weights_only=False
        )
        k96_args = k96_state["args"]
        if k96_args["sampling"] != "limb" or k96_args["hypotheses"] != 96:
            raise ValueError("checkpoint is not the frozen K96 limb model")
        self.scorer = dsac.PoseScoreMLP(
            k96_args["features"], k96_args.get("sigmoid_score", False)
        ).to(device)
        self.scorer.load_state_dict(k96_state["state_dict"], strict=True)
        self.scorer.eval().requires_grad_(False)
        self.proposal_temperature = float(k96_args["proposal_temperature"])
        self.score_temperature = float(k96_args["score_temperature"])

    def candidates_without_target(
        self, predictions: torch.Tensor, rays: torch.Tensor,
        combo: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        available, masks, task_mask, baseline_local = trainer.task_spec(
            combo, predictions.device
        )
        candidates = predictions[:, available]
        extras, extra_masks = extra.extra_candidates(
            candidates[:, baseline_local], rays, combo
        )
        candidates = torch.cat((candidates, extras), dim=1)
        masks = torch.cat((masks, extra_masks), dim=0)
        raw = self.e2(candidates, rays, masks, task_mask)
        unary = raw - raw[..., baseline_local:baseline_local + 1]
        return unary, candidates, baseline_local

    @torch.inference_mode()
    def hypothesis_evidence(
        self, predictions: torch.Tensor, rays: torch.Tensor,
        combo: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unary, candidates, baseline_local = self.candidates_without_target(
            predictions, rays, combo
        )
        group_cost = self.proposal(
            candidates, unary, rays, combo, self.bone_mean, self.bone_std
        )
        hypotheses, selected = dsac.sample_pool(
            candidates, unary, baseline_local, 96, "limb",
            self.proposal_temperature, group_cost,
        )
        scores = self.scorer(
            hypotheses, selected, self.coord_mean, self.coord_std,
            self.bone_mean, self.bone_std,
        )
        return hypotheses, selected, scores

    @torch.inference_mode()
    def hypotheses_and_scores(
        self, predictions: torch.Tensor, rays: torch.Tensor,
        combo: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hypotheses, _, scores = self.hypothesis_evidence(
            predictions, rays, combo
        )
        return hypotheses, scores

    @torch.inference_mode()
    def __call__(
        self, predictions: torch.Tensor, rays: torch.Tensor,
        combo: tuple[int, ...],
    ) -> torch.Tensor:
        hypotheses, scores = self.hypotheses_and_scores(
            predictions, rays, combo
        )
        weights = F.softmax(scores / self.score_temperature, dim=-1)
        return torch.einsum("bk,bkjd->bjd", weights, hypotheses)


def selected_rays(rays: torch.Tensor, combo: tuple[int, ...]) -> torch.Tensor:
    return rays[:, :, list(combo)]


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    return float(np.mean([
        values[actions == action].mean()
        for action in ACTION_NAMES if np.any(actions == action)
    ]))


def pose_errors(pose: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    absolute = torch.linalg.vector_norm(pose - target, dim=-1)
    pose_relative = pose - pose[:, :1]
    target_relative = target - target[:, :1]
    relative = torch.linalg.vector_norm(pose_relative - target_relative, dim=-1)
    return {
        "all17": absolute,
        "root": absolute[:, :1],
        "relative_nonroot": relative[:, 1:],
    }


def train_loss(
    output: torch.Tensor, target: torch.Tensor, auxiliary: dict[str, torch.Tensor],
    relative_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    absolute = torch.linalg.vector_norm(output - target, dim=-1).mean()
    output_relative = output - output[:, :1]
    target_relative = target - target[:, :1]
    relative = torch.linalg.vector_norm(
        output_relative[:, 1:] - target_relative[:, 1:], dim=-1
    ).mean()
    loss = absolute / 0.01 + relative_weight * relative / 0.01
    metrics = {
        "absolute_mm": float(absolute.detach() * 1000.0),
        "relative_nonroot_mm": float(relative.detach() * 1000.0),
        "trust_mean": float(auxiliary["trust"].detach().mean()),
        "prior_precision_mean": float(
            auxiliary["prior_precision"].detach().mean()
        ),
    }
    return loss, metrics


def evaluate(
    model: JointViewMAPDecoder,
    anchor_model: FrozenK96Anchor,
    loader: DataLoader,
    device: torch.device,
    seed: int,
) -> dict:
    model.eval()
    store = {
        f"V{count}": defaultdict(list) for count in (2, 3, 4)
    }
    actions_by_stage = {f"V{count}": [] for count in (2, 3, 4)}
    pair_store: dict[str, list[np.ndarray]] = defaultdict(list)
    diagnostics: dict[str, list[float]] = defaultdict(list)
    torch.manual_seed(10000 + seed)
    torch.cuda.manual_seed_all(10000 + seed)
    with torch.inference_mode():
        for predictions, targets, rays, actions in loader:
            predictions = predictions.to(device)
            targets = targets.to(device)
            rays = rays.to(device)
            action_numpy = actions.numpy().copy()
            for combo in TASKS:
                stage = f"V{len(combo)}"
                anchor = anchor_model(predictions, rays, combo)
                output, auxiliary = model(anchor, selected_rays(rays, combo))
                for mode, pose in (("anchor", anchor), ("map", output)):
                    errors = pose_errors(pose, targets)
                    for metric, values in errors.items():
                        store[stage][f"{mode}_{metric}"].append(
                            values.cpu().numpy() * 1000.0
                        )
                if len(combo) == 2:
                    pair_store[f"{combo[0] + 1}-{combo[1] + 1}"].append(
                        pose_errors(output, targets)["all17"].cpu().numpy() * 1000.0
                    )
                actions_by_stage[stage].append(action_numpy)
                diagnostics[f"{stage}_trust"].append(
                    float(auxiliary["trust"].mean())
                )
                diagnostics[f"{stage}_prior_precision"].append(
                    float(auxiliary["prior_precision"].mean())
                )
                diagnostics[f"{stage}_step_mm"].append(float(
                    torch.linalg.vector_norm(output - anchor, dim=-1).mean() * 1000.0
                ))
    result: dict[str, dict] = {}
    for stage in ("V2", "V3", "V4"):
        actions = np.concatenate(actions_by_stage[stage])
        result[stage] = {}
        for key, chunks in store[stage].items():
            values = np.concatenate(chunks)
            result[stage][key] = {
                "action_equal_mm": action_equal(values, actions),
                "frame_weighted_mm": float(values.mean()),
            }
        result[stage]["diagnostics"] = {
            "trust_mean": float(np.mean(diagnostics[f"{stage}_trust"])),
            "prior_precision_mean": float(
                np.mean(diagnostics[f"{stage}_prior_precision"])
            ),
            "accepted_step_mm": float(np.mean(diagnostics[f"{stage}_step_mm"])),
        }
    result["V2_pairs_map_frame_weighted_all17_mm"] = {
        key: float(np.concatenate(value).mean())
        for key, value in pair_store.items()
    }
    return result


def headline(result: dict) -> float:
    return float(np.mean([
        result[stage]["map_all17"]["action_equal_mm"]
        for stage in ("V2", "V3", "V4")
    ]))


def numerical_self_test() -> dict[str, float]:
    torch.manual_seed(123)
    model = JointViewMAPDecoder(
        d_model=32, heads=4, layers=1, dropout=0.0,
        prior_precision=30.0, trust_bias=-4.0,
    )
    batch, joints, views = 2, 17, 3
    anchor = torch.randn(batch, joints, 3) * 0.3
    origin = torch.randn(batch, joints, views, 3) * 2.0
    direction = F.normalize(
        anchor[:, :, None] - origin + 0.01 * torch.randn_like(origin), dim=-1
    )
    confidence = 0.2 + 0.8 * torch.rand(batch, joints, views, 1)
    rays = torch.cat((direction, origin, confidence), dim=-1)
    model.eval()
    output, _ = model(anchor, rays)
    permutation = torch.tensor([2, 0, 1])
    permuted, _ = model(anchor, rays[:, :, permutation])
    permutation_error = float((output - permuted).abs().max())
    identity_error = float((output - anchor).abs().max())
    model.train()
    train_output, _ = model(anchor, rays)
    loss = train_output.square().mean()
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(item).all() for item in gradients):
        raise RuntimeError("non-finite or absent gradient in MAP self-test")
    if permutation_error > 1e-5:
        raise RuntimeError(
            f"view permutation invariance failed: {permutation_error:.3e}"
        )
    return {
        "view_permutation_max_abs_m": permutation_error,
        "initial_identity_max_abs_m": identity_error,
        "gradient_tensor_count": float(len(gradients)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(args.train_cache)
    self_test = numerical_self_test()
    print(json.dumps({"self_test": self_test}), flush=True)

    device = torch.device(f"cuda:{args.gpu}")
    train = trainer.load_arrays([args.train_cache], 22)
    validation = trainer.load_arrays([args.validation_cache], 22)
    if args.holdout_subject:
        holdout_mask = train["subjects"] == args.holdout_subject
        if not np.any(holdout_mask):
            raise ValueError(
                f"holdout subject {args.holdout_subject} is absent from train cache"
            )
    else:
        holdout_mask = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout_mask)
    holdout_indices = np.flatnonzero(holdout_mask)
    if args.smoke_batches:
        train_indices = train_indices[: args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[: args.batch_size]
    if args.smoke_validation_batches:
        keep = args.batch_size * args.smoke_validation_batches
        validation = {key: value[:keep] for key, value in validation.items()}
    train_loader = DataLoader(
        ArrayDataset(train, train_indices), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers,
    )
    holdout_loader = DataLoader(
        ArrayDataset(train, holdout_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
    )
    validation_loader = DataLoader(
        ArrayDataset(validation, np.arange(len(validation["targets"]))),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )
    anchor_model = FrozenK96Anchor(args, device)
    model = JointViewMAPDecoder(
        d_model=args.d_model,
        heads=args.attention_heads,
        layers=args.attention_layers,
        dropout=args.dropout,
        prior_precision=args.prior_precision,
        max_prior_delta_m=args.max_prior_delta_m,
        trust_bias=args.trust_bias,
        learn_observation_precision=args.learn_observation_precision,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_metrics: dict[str, list[float]] = defaultdict(list)
        for batch_index, (predictions, targets, rays, _) in enumerate(train_loader):
            predictions = predictions.to(device)
            targets = targets.to(device)
            rays = rays.to(device)
            combo = TASKS[(batch_index + epoch * 7) % len(TASKS)]
            with torch.inference_mode():
                anchor = anchor_model(predictions, rays, combo)
            optimizer.zero_grad(set_to_none=True)
            output, auxiliary = model(anchor, selected_rays(rays, combo))
            loss, metrics = train_loss(
                output, targets, auxiliary, args.relative_loss_weight
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_metrics["loss"].append(float(loss.detach()))
            epoch_metrics["gradient_norm"].append(float(gradient_norm))
            epoch_metrics["trust_mean"].append(float(auxiliary["trust"].detach().mean()))
            epoch_metrics["accepted_step_mm"].append(float(
                torch.linalg.vector_norm(output.detach() - anchor, dim=-1).mean() * 1000.0
            ))
            epoch_metrics["prior_delta_mm"].append(float(
                torch.linalg.vector_norm(
                    auxiliary["prior_mean"].detach() - anchor, dim=-1
                ).mean() * 1000.0
            ))
        holdout_result = evaluate(
            model, anchor_model, holdout_loader, device, args.seed
        )
        score = headline(holdout_result)
        row = {
            "epoch": epoch + 1,
            "holdout_headline_mm": score,
            "train": {
                key: float(np.mean(value)) for key, value in epoch_metrics.items()
            },
            "holdout": holdout_result,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score < best:
            best = score
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "holdout_headline_mm": score,
                    "self_test": self_test,
                },
                output_dir / "model_best.pth.tar",
            )

    checkpoint = torch.load(
        output_dir / "model_best.pth.tar", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    validation_result = evaluate(
        model, anchor_model, validation_loader, device, args.seed
    )
    payload = {
        "method": "frozen K96 anchor + joint-view probabilistic prior + differentiable ray-MAP",
        "stage": "M2" if args.learn_observation_precision else "M1",
        "input_protocol": "HRNet coordinates/confidence/cameras only; no image or heatmap features",
        "best_epoch": checkpoint["epoch"],
        "best_holdout_headline_mm": checkpoint["holdout_headline_mm"],
        "self_test": self_test,
        "validation": validation_result,
        "history": history,
        "args": vars(args),
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["validation"], indent=2), flush=True)


if __name__ == "__main__":
    main()
