#!/usr/bin/env python3
"""GHT hypothesis utility with Set-Transformer candidate interaction."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_h76_counterfactual_delta_20260811 import (
    evaluate,
    predict_delta,
    training_loss,
)
from train_h76_hypothesis_utility_20260811 import (
    ArrayDataset,
    TASK_COMBINATIONS,
    load_arrays,
    masked_statistics,
)


def canonicalize_candidate_geometry(candidates, rays, task_mask):
    """GHT-style SE(3)-invariant geometry for candidate scoring.

    The frame uses only active-view rays and detector confidence.  Candidate
    coordinates are standardized for scoring; the caller still fuses the
    untouched world-space candidates, so metric output and camera calibration
    are preserved.
    """
    active = task_mask.bool().nonzero(as_tuple=False).flatten()
    selected = rays[:, :, active]
    direction = F.normalize(selected[..., :3], dim=-1, eps=1e-7)
    point = selected[..., 3:6]
    confidence = selected[..., 6:7].clamp(0, 1) + 0.05
    eye = torch.eye(3, device=rays.device, dtype=rays.dtype)
    projection = eye - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    weighted_projection = confidence.unsqueeze(-1) * projection
    lhs = weighted_projection.sum(dim=2)
    rhs = (weighted_projection @ point.unsqueeze(-1)).sum(dim=2)
    centroid = (
        (confidence * point).sum(dim=2)
        / confidence.sum(dim=2).clamp_min(1e-7)
    )
    # The scorer must use the same body-frame construction as the frozen
    # generator.  Earlier E2 runs silently hard-coded reg=1e-4 and disabled
    # the pelvis prior even when the generator used reg=1e-2 + pelvis prior;
    # that made the geometry features describe a different coordinate frame.
    regularization = float(os.environ.get("RUMPL_BODY_CANONICAL_REG", "1e-4"))
    pelvis_prior = os.environ.get("RUMPL_BODY_CANONICAL_PELVIS_PRIOR", "0") == "1"
    if pelvis_prior:
        pelvis_lhs = lhs[:, 0] + regularization * eye
        pelvis_rhs = rhs[:, 0] + regularization * centroid[:, 0].unsqueeze(-1)
        pelvis = torch.linalg.solve(pelvis_lhs, pelvis_rhs).squeeze(-1)
        lhs = lhs + regularization * eye
        rhs = rhs + regularization * pelvis[:, None, :, None]
    else:
        lhs = lhs + regularization * eye
        rhs = rhs + regularization * centroid.unsqueeze(-1)
    anchors = torch.linalg.solve(lhs, rhs).squeeze(-1)
    origin = anchors[:, 0]
    x_axis = F.normalize(anchors[:, 14] - anchors[:, 11], dim=-1, eps=1e-7)
    up_hint = anchors[:, 8] - origin
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
    basis = torch.stack((x_axis, y_axis, z_axis), dim=-1)

    canonical_candidates = torch.einsum(
        "b...i,bij->b...j",
        candidates - origin[:, None, None, :],
        basis,
    )
    canonical_rays = rays.clone()
    canonical_rays[..., :3] = torch.einsum(
        "b...i,bij->b...j", rays[..., :3], basis
    )
    canonical_rays[..., 3:6] = torch.einsum(
        "b...i,bij->b...j",
        rays[..., 3:6] - origin[:, None, None, :],
        basis,
    )
    return canonical_candidates, canonical_rays


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--attention-depth", type=int, choices=(0, 1, 2, 3, 4), required=True
    )
    parser.add_argument("--view-cross-attention", action="store_true")
    parser.add_argument(
        "--joint-attention", choices=("none", "post", "alternating"),
        default="none",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument(
        "--total-epochs", type=int, default=0,
        help="Optional total epoch count. When larger than pretrain+finetune, "
             "extra epochs continue the GHT phase.",
    )
    parser.add_argument(
        "--resume-checkpoint", default="",
        help="Resume model weights from a previous model_best.pth.tar.",
    )
    parser.add_argument(
        "--resume-start-epoch", type=int, default=-1,
        help="First epoch to run when resuming; defaults to checkpoint epoch + 1.",
    )
    parser.add_argument(
        "--resume-best-metric", type=float, default=math.inf,
        help="Existing holdout selection metric for the resumed checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


class SetTransformerJointUtility(nn.Module):
    """Permutation-equivariant cross-candidate joint utility scorer.

    It retains C2's geometry features but replaces independent candidate MLP
    scoring with Set Transformer-style self-attention over the available H76
    hypotheses.  No candidate-index or camera-ID embedding is used.
    """

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor, attention_depth: int,
        view_cross_attention: bool = False, joint_attention: str = "none",
        stage_heads: bool = False, neutralize_subset_penalty: bool = False,
        canonical_geometry: bool = False,
        fixed_metric_normalization: bool = False,
    ):
        super().__init__()
        self.register_buffer("pose_mean", mean)
        self.register_buffer("pose_std", std.clamp_min(1e-6))
        self.view_cross_attention = view_cross_attention
        self.joint_attention = joint_attention
        self.stage_heads = stage_heads
        self.neutralize_subset_penalty = neutralize_subset_penalty
        self.canonical_geometry = canonical_geometry
        self.fixed_metric_normalization = fixed_metric_normalization
        self.pose_encoder = nn.Sequential(
            nn.Linear(51, 64), nn.ReLU6(), nn.Linear(64, 64), nn.ReLU6()
        )
        self.joint_embedding = nn.Parameter(torch.zeros(17, 16))
        self.candidate_encoder = nn.Sequential(
            nn.Linear(106, 96), nn.ReLU6(), nn.Linear(96, 64), nn.ReLU6()
        )
        if view_cross_attention:
            # Direction(3), candidate-to-ray perpendicular vector(3), its
            # norm(1), detector confidence(1), candidate-inclusion flag(1).
            self.view_encoder = nn.Sequential(
                nn.Linear(9, 64), nn.ReLU6(), nn.Linear(64, 64)
            )
            self.view_attention = nn.MultiheadAttention(
                64, 8, dropout=0.0, batch_first=True
            )
            self.view_norm = nn.LayerNorm(64)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=8,
                dim_feedforward=128,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(attention_depth)
        ])
        joint_block_count = attention_depth if joint_attention == "alternating" else (
            1 if joint_attention == "post" else 0
        )
        self.joint_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64, nhead=8, dim_feedforward=128, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(joint_block_count)
        ])
        def make_output_head():
            return nn.Sequential(
                nn.LayerNorm(64), nn.Linear(64, 32), nn.ReLU6(), nn.Linear(32, 1)
            )
        if stage_heads:
            # The geometry/candidate encoder remains shared.  Only the final
            # utility calibration is separated by view count, preventing the
            # V2, V3 and V4 objectives from forcing one common score scale.
            self.output_heads = nn.ModuleDict({
                "V2": make_output_head(),
                "V3": make_output_head(),
                "V4": make_output_head(),
            })
        else:
            self.output = make_output_head()

    def forward(self, candidates, rays, candidate_masks, task_mask):
        batch, count, joints, _ = candidates.shape
        if self.canonical_geometry:
            candidates, rays = canonicalize_candidate_geometry(
                candidates, rays, task_mask
            )
        if self.fixed_metric_normalization:
            # Dataset-independent units: coordinates are metres and the
            # canonical body extent is O(1 m).  Isotropic scaling avoids
            # reintroducing H36M axis statistics after standardization.
            normalized = candidates / 0.5
        else:
            normalized = (candidates - self.pose_mean) / self.pose_std
        root_relative = normalized - normalized[:, :, :1]
        context = self.pose_encoder(root_relative.flatten(2))
        context = context[:, :, None].expand(-1, -1, joints, -1)

        consensus = candidates.mean(dim=1, keepdim=True)
        delta = candidates - consensus
        delta_feature = torch.cat(
            (delta / 0.1, torch.linalg.vector_norm(delta, dim=-1, keepdim=True) / 0.1),
            dim=-1,
        )
        local = torch.cat(
            (root_relative, normalized[:, :, :1].expand(-1, -1, joints, -1)),
            dim=-1,
        )
        direction = F.normalize(rays[..., :3], dim=-1)
        point = rays[..., 3:6]
        offset = candidates[:, :, :, None, :] - point[:, None]
        residual = torch.linalg.vector_norm(
            torch.cross(offset, direction[:, None], dim=-1), dim=-1
        )
        residual = torch.log1p(residual / 0.005)
        included = candidate_masks.bool()
        excluded = (~included) & task_mask[None].bool()
        included_residual = masked_statistics(residual, included)
        excluded_residual = masked_statistics(residual, excluded)

        confidence = rays[..., 6].clamp(0, 1)
        confidence = confidence[:, None].expand(-1, count, -1, -1)
        included_conf = masked_statistics(confidence, included)
        excluded_conf = masked_statistics(confidence, excluded)
        projection = (
            torch.eye(3, device=rays.device, dtype=rays.dtype)
            - direction.unsqueeze(-1) * direction.unsqueeze(-2)
        )
        weight = rays[..., 6:7].clamp(0, 1) + 0.05
        normal_per_view = weight.unsqueeze(-1) * projection
        candidate_normal = torch.einsum(
            "cv,bjvxy->bcjxy", candidate_masks, normal_per_view
        )
        eigenvalues = torch.linalg.eigvalsh(candidate_normal).clamp_min(1e-7)
        spectrum = torch.log(eigenvalues / eigenvalues.sum(dim=-1, keepdim=True))
        view_fraction = (
            candidate_masks.sum(dim=-1)[None, :, None, None] / task_mask.sum()
        ).expand(batch, -1, joints, -1)
        if self.neutralize_subset_penalty:
            # Leave-one-out and single-view hypotheses are the oracle winners,
            # but excluded-residual / view-fraction features punish them.
            excluded_residual = torch.zeros_like(excluded_residual)
            excluded_conf = torch.zeros_like(excluded_conf)
            view_fraction = torch.zeros_like(view_fraction)
        joint = self.joint_embedding[None, None].expand(batch, count, -1, -1)
        features = torch.cat(
            (
                local, delta_feature, included_residual, excluded_residual,
                included_conf, excluded_conf, view_fraction, spectrum,
                context, joint,
            ),
            dim=-1,
        )
        if features.shape[-1] != 106:
            raise RuntimeError(f"set utility feature size {features.shape[-1]}")
        tokens = self.candidate_encoder(features)  # B,C,J,D
        if self.view_cross_attention:
            perpendicular = offset - (offset * direction[:, None]).sum(
                dim=-1, keepdim=True
            ) * direction[:, None]
            perpendicular_norm = torch.linalg.vector_norm(
                perpendicular, dim=-1, keepdim=True
            )
            included_flag = candidate_masks[None, :, None, :, None].expand(
                batch, -1, joints, -1, -1
            )
            view_confidence = rays[..., 6:7][:, None].expand(
                -1, count, -1, -1, -1
            )
            view_features = torch.cat(
                (
                    direction[:, None].expand(-1, count, -1, -1, -1),
                    perpendicular / 0.05,
                    perpendicular_norm / 0.05,
                    view_confidence,
                    included_flag,
                ),
                dim=-1,
            )
            view_tokens = self.view_encoder(view_features).reshape(
                batch * count * joints, rays.shape[2], 64
            )
            candidate_query = tokens.reshape(
                batch * count * joints, 1, 64
            )
            padding_mask = (~task_mask.bool())[None].expand(
                batch * count * joints, -1
            )
            # A 512-frame batch can produce >95k independent candidate/joint
            # sequences, exceeding the CUDA fused-SDPA grid limit although it
            # fits memory.  Chunking is mathematically identical because these
            # sequences never attend to each other.
            view_context_chunks = []
            for start in range(0, len(candidate_query), 32768):
                stop = min(start + 32768, len(candidate_query))
                chunk, _ = self.view_attention(
                    candidate_query[start:stop],
                    view_tokens[start:stop], view_tokens[start:stop],
                    key_padding_mask=padding_mask[start:stop],
                    need_weights=False,
                )
                view_context_chunks.append(chunk)
            view_context = torch.cat(view_context_chunks, dim=0)
            tokens = self.view_norm(
                candidate_query + view_context
            ).reshape(batch, count, joints, 64)
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch * joints, count, 64)
        for block_index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if self.joint_attention == "alternating":
                joint_tokens = tokens.reshape(batch, joints, count, 64)
                joint_tokens = joint_tokens.permute(0, 2, 1, 3).reshape(
                    batch * count, joints, 64
                )
                joint_tokens = self.joint_blocks[block_index](joint_tokens)
                tokens = joint_tokens.reshape(batch, count, joints, 64)
                tokens = tokens.permute(0, 2, 1, 3).reshape(
                    batch * joints, count, 64
                )
        if self.joint_attention == "post":
            joint_tokens = tokens.reshape(batch, joints, count, 64)
            joint_tokens = joint_tokens.permute(0, 2, 1, 3).reshape(
                batch * count, joints, 64
            )
            joint_tokens = self.joint_blocks[0](joint_tokens)
            tokens = joint_tokens.reshape(batch, count, joints, 64)
            tokens = tokens.permute(0, 2, 1, 3).reshape(
                batch * joints, count, 64
            )
        if self.stage_heads:
            view_count = int(task_mask.sum().detach().item())
            try:
                output_head = self.output_heads[f"V{view_count}"]
            except KeyError as exc:
                raise ValueError(f"unsupported view count for stage heads: {view_count}") from exc
        else:
            output_head = self.output
        score = output_head(tokens).reshape(batch, joints, count)
        return score


def task_loss(model, predictions, targets, rays, phase):
    direct_losses = []
    ght_losses = []
    for combo in TASK_COMBINATIONS:
        predicted, true_delta, true_error, candidates, _ = predict_delta(
            model, predictions, targets, rays, combo
        )
        direct_losses.append(training_loss(predicted, true_delta, "balanced_rank"))
        if phase == "ght":
            weights = F.softmax(-predicted, dim=-1)
            expected = (weights * true_error).sum(dim=-1).mean()
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            ght_losses.append((expected + 0.05 * fused_error) / 0.01)
    loss = torch.stack(direct_losses).mean()
    if ght_losses:
        loss = loss + torch.stack(ght_losses).mean()
    return loss


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    train = load_arrays(args.train_shards)
    validation_npz = np.load(args.validation_cache)
    validation = {key: validation_npz[key] for key in validation_npz.files}
    holdout = train["group_indices"] % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    if args.smoke_batches:
        train_indices = train_indices[: args.batch_size * args.smoke_batches]
        holdout_indices = holdout_indices[:args.batch_size]
    mean = torch.from_numpy(train["targets"][train_indices].mean(axis=(0, 1)))
    std = torch.from_numpy(train["targets"][train_indices].std(axis=(0, 1)))
    model = SetTransformerJointUtility(
        mean, std, args.attention_depth, args.view_cross_attention,
        args.joint_attention,
    ).to(device)
    resume_checkpoint = None
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(resume_checkpoint["state_dict"], strict=True)
        if int(resume_checkpoint.get("attention_depth", args.attention_depth)) != args.attention_depth:
            raise ValueError("resume checkpoint attention depth does not match")
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model_best.pth.tar"
    best_metric = math.inf
    best_epoch = -1
    if resume_checkpoint is not None:
        best_epoch = int(resume_checkpoint.get("epoch", -1))
        # The existing checkpoint is already the best checkpoint selected on
        # the original holdout.  Preserve it if all extension epochs are
        # worse; the exact metric is supplied by the caller from result.json.
        best_metric = float(args.resume_best_metric)
        torch.save(resume_checkpoint, checkpoint_path)
    history = []
    base_phases = (
        [("direct", 5e-4)] * args.pretrain_epochs
        + [("ght", 1e-4)] * args.finetune_epochs
    )
    total_epochs = args.total_epochs or len(base_phases)
    if total_epochs < len(base_phases):
        raise ValueError("total-epochs cannot be smaller than pretrain+finetune")
    phases = base_phases + [("ght", 1e-4)] * (total_epochs - len(base_phases))
    start_epoch = 0
    if resume_checkpoint is not None:
        start_epoch = (
            args.resume_start_epoch
            if args.resume_start_epoch >= 0
            else int(resume_checkpoint.get("epoch", -1)) + 1
        )
        if not (0 <= start_epoch < total_epochs):
            raise ValueError(f"invalid resume start epoch {start_epoch} for {total_epochs} epochs")
    optimizer = None
    previous_phase = None
    for epoch in range(start_epoch, total_epochs):
        phase, lr = phases[epoch]
        if phase != previous_phase:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=1e-4
            )
            previous_phase = phase
        model.train()
        losses = []
        for predictions, targets, rays, _ in train_loader:
            predictions = predictions.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(model, predictions, targets, rays, phase)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate(model, holdout_loader, device, 1.0)
        metric = 0.5 * (
            holdout_result["V3"]["soft"]["action_equal_all17_mm"]
            + holdout_result["V4"]["soft"]["action_equal_all17_mm"]
        )
        record = {
            "epoch": epoch,
            "phase": phase,
            "train_loss": float(np.mean(losses)),
            "holdout_selection_metric_mm": float(metric),
            "holdout": holdout_result,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric = float(metric)
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(), "mean": mean, "std": std,
                "attention_depth": args.attention_depth, "epoch": epoch,
                "view_cross_attention": args.view_cross_attention,
                "joint_attention": args.joint_attention,
                "phase": phase,
            }, checkpoint_path)

    best = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(best["state_dict"], strict=True)
    test_result = None if args.smoke_batches else evaluate(
        model, test_loader, device, 1.0
    )
    payload = {
        "method": (
            "GHT hypotheses + "
            + ("ray-view cross-attention + " if args.view_cross_attention else "")
            + "Set Transformer candidate interaction"
            + (f" + {args.joint_attention} joint-axis attention"
               if args.joint_attention != "none" else "")
        ),
        "attention_depth": args.attention_depth,
        "view_cross_attention": args.view_cross_attention,
        "joint_attention": args.joint_attention,
        "best_epoch": best_epoch,
        "best_phase": best["phase"],
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
