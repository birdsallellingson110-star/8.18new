#!/usr/bin/env python3
"""Train/evaluate the HRNet feature-level RIGR probe.

This is the first experiment that uses image evidence rather than heatmap
top-K coordinates.  The detector and H76 are frozen; only a small
view-shared/joint-shared refiner is trained.  The input contains a 5x5 patch
sampled from HRNet's high-resolution feature map at the calibrated projection
of the frozen H76 3-D query.  The zero-initialized output makes epoch zero an
exact H76 baseline, which prevents a hidden trunk or protocol change.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diagnose_rigr_heatmap_oracle_20260812 import build_four_view_groups


COMBINATIONS = tuple(combo for n in (2, 3, 4) for combo in itertools.combinations(range(4), n))


class GeometryViewBlock(nn.Module):
    """Small view-axis Transformer block with a per-sample logit bias.

    ``nn.TransformerEncoder`` accepts a 3-D mask, but expands a separate
    ``[batch*joint*head, V, V]`` mask through a slow generic path.  Since the
    present experiment has only four views, an explicit qkv implementation is
    both clearer and substantially cheaper while keeping the bias at the
    attention-logit location.
    """

    def __init__(self, d_model: int, nhead: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        # x: [N,V,D], bias: [N,H,V,V], valid: [N,V].
        n, v, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            n, v, 3, self.nhead, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, value = qkv.unbind(0)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        logits = logits + bias
        logits = logits.masked_fill(~valid[:, None, None, :], -1e4)
        weights = torch.softmax(logits, dim=-1)
        output = torch.matmul(weights, value).transpose(1, 2).reshape(n, v, d)
        x = x + self.drop1(self.proj(output))
        x = x + self.mlp(self.norm2(x))
        return x * valid[..., None].to(x.dtype)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", nargs="+", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--train-tokens", required=True)
    p.add_argument("--validation-tokens", required=True)
    p.add_argument("--train-aux", default="",
                   help="Optional subset-specific detector geometry auxiliaries .npy")
    p.add_argument("--validation-aux", default="",
                   help="Optional validation detector geometry auxiliaries .npy")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-train-groups", type=int, default=10000)
    p.add_argument("--max-validation-groups", type=int, default=0,
                   help="Optional validation prefix for implementation smoke tests.")
    p.add_argument("--train-group-indices-file", default="",
                   help="Optional .npy/.txt group IDs for a coverage-balanced train subset.")
    p.add_argument("--holdout-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gated-residual", action="store_true",
                   help="multiply the bounded 3-D residual by a learned joint trust gate")
    p.add_argument("--attention-pooling", action="store_true",
                   help="replace equal view averaging with learned quality attention pooling")
    p.add_argument("--cross-view-relation", action="store_true",
                   help="add an explicit current-view versus other-view relation token")
    p.add_argument("--patch-attention", action="store_true",
                   help="learn spatial weights inside each sampled feature patch")
    p.add_argument("--geometry-biased-attention", action="store_true",
                   help="add a learned pairwise ray-geometry bias to view attention logits")
    p.add_argument("--explicit-view-block", action="store_true",
                   help="use the explicit four-view qkv block without geometry bias (control)")
    p.add_argument("--correspondence-attention", action="store_true",
                   help="keep patch tokens and perform cross-view query-to-patch attention")
    return p.parse_args()


def load_cache(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    order = np.argsort(arrays["group_indices"])
    return {key: value[order] for key, value in arrays.items()}


class TokenDataset(Dataset):
    def __init__(self, cache: dict[str, np.ndarray], tokens_path: str,
                 group_ids: np.ndarray, combos: list[int], aux_path: str = "") -> None:
        self.cache = cache
        self.tokens = np.load(tokens_path, mmap_mode="r")
        self.subset_specific = self.tokens.ndim == 7
        self.aux = np.load(aux_path, mmap_mode="r") if aux_path else None
        if self.aux is not None and self.aux.ndim != 5:
            raise ValueError(f"aux must be [G,11,V,J,D], got {self.aux.shape}")
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.combos = np.asarray(combos, dtype=np.int64)
        if self.tokens.shape[0] < len(cache["targets"]):
            raise ValueError(f"tokens/cache mismatch {self.tokens.shape} vs {len(cache['targets'])}")

    def __len__(self) -> int:
        return len(self.group_ids) * len(self.combos)

    def __getitem__(self, index: int):
        local = index // len(self.combos)
        combo_id = int(self.combos[index % len(self.combos)])
        group_id = int(self.group_ids[local])
        views = np.asarray(COMBINATIONS[combo_id], dtype=np.int64)
        # Cache rows are already in group order; group_id is local to the
        # cache passed to this Dataset (train/holdout views are sliced first).
        prediction = self.cache["predictions"][local, combo_id].astype(np.float32)
        # Slice the group row first: mixing a basic slice and a NumPy advanced
        # index in one expression would transpose the joint/view axes.
        rays = self.cache["rays"][local][:, views].astype(np.float32)
        target = self.cache["targets"][local].astype(np.float32)
        # Convert memmapped float16 evidence before torch LayerNorm; the
        # model itself is trained in float32 for stable residual updates.
        if self.subset_specific:
            features = np.asarray(self.tokens[group_id, combo_id, views], dtype=np.float32)
        else:
            features = np.asarray(self.tokens[group_id, views], dtype=np.float32)
        aux = None
        if self.aux is not None:
            aux = np.asarray(self.aux[group_id, combo_id, views], dtype=np.float32)
        return prediction, rays, features, aux, target, np.int64(combo_id)


def collate(batch):
    prediction, rays, features, aux, target, combo = zip(*batch)
    b = len(batch)
    j, d, p = 17, features[0].shape[2], features[0].shape[3]
    padded_rays = np.zeros((b, j, 4, 7), dtype=np.float32)
    padded_features = np.zeros((b, 4, j, d, p, p), dtype=np.float32)
    aux_dim = 0 if aux[0] is None else int(aux[0].shape[-1])
    padded_aux = np.zeros((b, 4, j, aux_dim), dtype=np.float32) if aux_dim else None
    mask = np.zeros((b, 4), dtype=np.bool_)
    for row, (r, f, a) in enumerate(zip(rays, features, aux)):
        n = r.shape[1]
        padded_rays[row, :, :n] = r
        padded_features[row, :n] = f
        if padded_aux is not None:
            padded_aux[row, :n] = a
        mask[row, :n] = True
    return (
        torch.from_numpy(np.stack(prediction)), torch.from_numpy(padded_rays),
        torch.from_numpy(padded_features),
        None if padded_aux is None else torch.from_numpy(padded_aux),
        torch.from_numpy(np.stack(target)), torch.from_numpy(np.stack(combo)),
        torch.from_numpy(mask),
    )


class RIGRHRNetFeature(nn.Module):
    def __init__(self, channels: int, patch: int, d_model: int = 128,
                 gated_residual: bool = False, aux_dim: int = 0,
                 attention_pooling: bool = False,
                 cross_view_relation: bool = False,
                 patch_attention: bool = False,
                 geometry_biased_attention: bool = False,
                 explicit_view_block: bool = False,
                 correspondence_attention: bool = False) -> None:
        super().__init__()
        self.gated_residual = bool(gated_residual)
        self.aux_dim = int(aux_dim)
        self.attention_pooling = bool(attention_pooling)
        self.cross_view_relation = bool(cross_view_relation)
        self.patch_attention = bool(patch_attention)
        self.geometry_biased_attention = bool(geometry_biased_attention)
        self.explicit_view_block = bool(explicit_view_block)
        self.correspondence_attention = bool(correspondence_attention)
        self.patch_size = int(patch)
        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(channels * patch * patch),
            nn.Linear(channels * patch * patch, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        if self.correspondence_attention:
            # Preserve the spatial 5x5 HRNet evidence instead of flattening it
            # into one view token.  The query is the centre feature at the
            # projected H76 point; keys/values are all patch locations from
            # every available view.  No camera-ID embedding is used.
            self.patch_token_encoder = nn.Sequential(
                nn.LayerNorm(channels), nn.Linear(channels, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.corr_patch_position = nn.Parameter(torch.zeros(patch * patch, d_model))
            self.correspondence_attention_layer = nn.MultiheadAttention(
                d_model, 4, dropout=0.0, batch_first=True,
            )
            self.correspondence_norm = nn.LayerNorm(d_model)
            self.correspondence_gate = nn.Parameter(torch.zeros(()))
        if self.patch_attention:
            self.patch_encoder = nn.Sequential(
                nn.LayerNorm(channels), nn.Linear(channels, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.patch_position = nn.Parameter(torch.zeros(patch * patch, d_model))
            self.patch_score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
            self.patch_output = nn.Linear(d_model, d_model)
            # Equal spatial pooling at initialization; the zero-initialized
            # patch residual makes epoch zero exactly the established feature
            # encoder, while the final 3-D head remains H76 as before.
            nn.init.zeros_(self.patch_score[-1].weight)
            nn.init.zeros_(self.patch_score[-1].bias)
            nn.init.zeros_(self.patch_output.weight)
            nn.init.zeros_(self.patch_output.bias)
        # Ray geometry and the current 3-D query are camera-order agnostic.
        geometry_dim = 7 + 3 + self.aux_dim
        self.geometry_encoder = nn.Sequential(
            nn.LayerNorm(geometry_dim), nn.Linear(geometry_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        view_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.view_transformer = nn.TransformerEncoder(view_layer, num_layers=2)
        if self.geometry_biased_attention or self.explicit_view_block:
            # The ray features are converted into a pairwise, camera-order
            # agnostic bias at the actual attention-logit location.  A zero
            # final layer makes this branch an exact no-bias initialization;
            # it cannot win merely by changing the epoch-zero representation.
            self.geometry_bias = nn.Sequential(
                nn.LayerNorm(5), nn.Linear(5, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 4),
            )
            nn.init.zeros_(self.geometry_bias[-1].weight)
            nn.init.zeros_(self.geometry_bias[-1].bias)
            self.geometry_view_blocks = nn.ModuleList(
                [GeometryViewBlock(d_model, nhead=4, dropout=0.1) for _ in range(2)]
            )
            # The explicit block replaces the generic view encoder only for
            # this ablation; all other variants retain the established path.
        if self.cross_view_relation:
            self.relation_encoder = nn.Sequential(
                nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            # Start with exactly the established view-token stream.  The
            # relation branch is learned only from the training signal.
            nn.init.zeros_(self.relation_encoder[-1].weight)
            nn.init.zeros_(self.relation_encoder[-1].bias)
        if self.attention_pooling:
            self.view_quality = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
            # Uniform pooling at initialization makes this an isolated change
            # from the established equal-view feature model.
            nn.init.zeros_(self.view_quality[-1].weight)
            nn.init.zeros_(self.view_quality[-1].bias)
        joint_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.joint_transformer = nn.TransformerEncoder(joint_layer, num_layers=2)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.joint_embedding = nn.Parameter(torch.zeros(17, d_model))
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        if self.gated_residual:
            self.trust_gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
            # Start close to identity while leaving the residual head exactly
            # zero.  The gate is per joint, not a camera-ID lookup, so it can
            # be transferred to unseen camera layouts.
            nn.init.zeros_(self.trust_gate[-1].weight)
            nn.init.constant_(self.trust_gate[-1].bias, -2.0)

    def forward(self, prediction: torch.Tensor, rays: torch.Tensor,
                features: torch.Tensor, view_mask: torch.Tensor,
                aux: torch.Tensor | None = None) -> torch.Tensor:
        # prediction B,J,3; rays B,J,V,7; features B,V,J,C,P,P.
        b, j, v, _ = rays.shape
        feat = features.permute(0, 2, 1, 3, 4, 5).contiguous()
        base_feat = self.feature_encoder(feat.reshape(b * j * v, -1)).reshape(b * j, v, -1)
        if self.correspondence_attention:
            _, _, _, channels, height, width = feat.shape
            patch_tokens = feat.reshape(
                b * j * v, channels, height * width
            ).transpose(1, 2)
            patch_tokens = self.patch_token_encoder(patch_tokens)
            patch_tokens = patch_tokens + self.corr_patch_position[None, :height * width]
            patch_tokens = patch_tokens.reshape(b * j, v * height * width, -1)

            # Each current-view query attends to all patch locations in all
            # available views.  Padding views are masked at key level; the
            # subsequent view mask also removes their query contribution.
            query = base_feat.reshape(b * j * v, 1, -1)
            key_value = patch_tokens[:, None].expand(
                -1, v, -1, -1
            ).reshape(b * j * v, v * height * width, -1)
            valid_views = view_mask[:, None, :].expand(
                b, j, v
            ).reshape(b * j, v)
            valid_patches = valid_views.repeat_interleave(height * width, dim=1)
            key_padding = (~valid_patches[:, None].expand(
                b * j, v, v * height * width
            ).reshape(b * j * v, v * height * width))
            attended, _ = self.correspondence_attention_layer(
                query, key_value, key_value,
                key_padding_mask=key_padding,
                need_weights=False,
            )
            attended = attended.reshape(b * j, v, -1)
            mixed = self.correspondence_norm(base_feat + attended)
            # Identity at initialization: the new correspondence path is
            # enabled gradually instead of changing the old feature stream
            # merely because LayerNorm was added.
            feat = base_feat + torch.tanh(self.correspondence_gate) * (mixed - base_feat)
        elif self.patch_attention:
            _, _, _, channels, height, width = feat.shape
            patch_tokens = feat.reshape(b * j * v, channels, height * width).transpose(1, 2)
            patch_tokens = self.patch_encoder(patch_tokens)
            patch_tokens = patch_tokens + self.patch_position[None, :height * width]
            patch_weights = torch.softmax(self.patch_score(patch_tokens).squeeze(-1), dim=1)
            patch_fused = (patch_tokens * patch_weights[..., None]).sum(dim=1)
            patch_delta = self.patch_output(patch_fused).reshape(b * j, v, -1)
            feat = base_feat + patch_delta
        else:
            feat = base_feat
        query = prediction[:, :, None, :].expand(-1, -1, v, -1)
        geometry_input = torch.cat((rays, query), dim=-1)
        if self.aux_dim:
            if aux is None or aux.shape[-1] != self.aux_dim:
                raise ValueError(f"model expects aux_dim={self.aux_dim}, got {None if aux is None else aux.shape}")
            geometry_input = torch.cat((geometry_input, aux.permute(0, 2, 1, 3)), dim=-1)
        geometry = self.geometry_encoder(geometry_input.reshape(b * j * v, -1))
        tokens = feat + geometry.reshape(b * j, v, -1)
        if self.cross_view_relation:
            valid_flat = view_mask[:, None, :].expand(b, j, v).reshape(b * j, v, 1).to(tokens.dtype)
            context = (tokens * valid_flat).sum(dim=1, keepdim=True) / valid_flat.sum(dim=1, keepdim=True).clamp_min(1)
            relation = tokens - context
            tokens = tokens + self.relation_encoder(torch.cat((tokens, relation), dim=-1))
        key_padding = (~view_mask.bool())[:, None, :].expand(b, j, v).reshape(b * j, v)
        if self.geometry_biased_attention or self.explicit_view_block:
            # Pairwise geometry for each joint and view pair.  Rays contain
            # normalized direction, camera center and detector confidence.
            direction = torch.nn.functional.normalize(rays[..., :3], dim=-1)
            center = rays[..., 3:6]
            confidence = rays[..., 6].clamp(0.0, 1.0)
            first_direction = direction.unsqueeze(-2)
            second_direction = direction.unsqueeze(-3)
            cross = torch.cross(first_direction.expand(-1, -1, v, v, -1),
                                second_direction.expand(-1, -1, v, v, -1), dim=-1)
            sine = torch.linalg.vector_norm(cross, dim=-1)
            cosine = (first_direction * second_direction).sum(dim=-1).abs().clamp(0.0, 1.0)
            baseline = center.unsqueeze(-2) - center.unsqueeze(-3)
            skew = (baseline * cross).sum(dim=-1).abs() / sine.clamp_min(1e-5)
            point_line = torch.linalg.vector_norm(
                torch.cross(baseline, first_direction.expand(-1, -1, v, v, -1), dim=-1), dim=-1
            )
            ray_distance = torch.where(sine > 1e-5, skew, point_line)
            pair = ~torch.eye(v, dtype=torch.bool, device=rays.device)[None, None]
            pair = pair & view_mask[:, None, :, None] & view_mask[:, None, None, :]
            scale = (ray_distance * pair).sum(dim=(-1, -2), keepdim=True)
            scale = scale / pair.sum(dim=(-1, -2), keepdim=True).clamp_min(1)
            ray_distance = (ray_distance / scale.clamp_min(1e-4)).clamp(0.0, 10.0)
            if self.geometry_biased_attention:
                pair_features = torch.stack(
                    (cosine, sine, ray_distance,
                     confidence.unsqueeze(-1).expand(-1, -1, v, v),
                     confidence.unsqueeze(-2).expand(-1, -1, v, v)), dim=-1
                )
                pair_features = pair_features * pair[..., None].to(pair_features.dtype)
                # [B,J,V,V,heads] -> [B*J,heads,V,V].
                bias = self.geometry_bias(pair_features).permute(0, 1, 4, 2, 3)
            else:
                bias = torch.zeros(
                    b, j, 4, v, v, dtype=tokens.dtype, device=tokens.device
                )
            view_valid = view_mask[:, None, :].expand(b, j, v).reshape(b * j, v)
            tokens = tokens.reshape(b * j, v, -1)
            for block in self.geometry_view_blocks:
                tokens = block(tokens, bias.reshape(b * j, 4, v, v), view_valid)
        else:
            tokens = self.view_transformer(tokens, src_key_padding_mask=key_padding)
        tokens = tokens.reshape(b, j, v, -1)
        valid = view_mask[:, None, :].bool().expand(b, j, v)
        if self.attention_pooling:
            quality = self.view_quality(tokens).squeeze(-1)
            quality = quality.masked_fill(~valid, torch.finfo(quality.dtype).min)
            weights = torch.softmax(quality, dim=2).masked_fill(~valid, 0.0)
            fused = (tokens * weights[..., None]).sum(dim=2)
        else:
            valid_float = valid[..., None].to(tokens.dtype)
            fused = (tokens * valid_float).sum(dim=2) / valid_float.sum(dim=2).clamp_min(1)
        fused = fused + self.query_encoder(torch.cat((prediction, prediction - prediction[:, :1]), dim=-1))
        fused = fused + self.joint_embedding[None]
        fused = self.joint_transformer(fused)
        delta = 0.25 * torch.tanh(self.output(fused))
        if self.gated_residual:
            delta = delta * torch.sigmoid(self.trust_gate(fused))
        return prediction + delta


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    per_action = []
    for action in sorted(set(int(x) for x in actions)):
        selected = values[actions == action]
        if len(selected):
            per_action.append(float(selected.mean()))
    return float(np.mean(per_action))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             actions: np.ndarray) -> dict:
    model.eval()
    errors, combo_ids = [], []
    for prediction, rays, features, aux, target, combo, mask in loader:
        prediction = prediction.to(device)
        rays = rays.to(device)
        features = features.to(device)
        mask = mask.to(device)
        target = target.to(device)
        if aux is not None:
            aux = aux.to(device)
        output = model(prediction, rays, features, mask, aux)
        errors.append(torch.linalg.vector_norm(output - target.to(device), dim=-1).mean(dim=-1).cpu().numpy())
        combo_ids.append(combo.numpy())
    values = np.concatenate(errors) * 1000.0
    combo_ids = np.concatenate(combo_ids)
    result = {}
    for n in (2, 3, 4):
        ids = [i for i, c in enumerate(COMBINATIONS) if len(c) == n]
        selected = np.isin(combo_ids, ids)
        n_values = values[selected]
        n_actions = np.repeat(actions, len(ids))
        result[f"V{n}"] = {
            "frame_weighted_all17_mm": float(n_values.mean()),
            "action_equal_all17_mm": action_equal(n_values, n_actions),
        }
    return result


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_full = load_cache(args.train_cache)
    val = load_cache([args.validation_cache])
    with open(args.train_pkl, "rb") as handle:
        train_records = pickle.load(handle)
    with open(args.validation_pkl, "rb") as handle:
        val_records = pickle.load(handle)
    train_groups = build_four_view_groups(train_records)
    val_groups = build_four_view_groups(val_records)
    if len(train_groups) != len(train_full["targets"]):
        raise ValueError(f"train group/cache mismatch {len(train_groups)} vs {len(train_full['targets'])}")
    if len(val_groups) != len(val["targets"]):
        raise ValueError(f"val group/cache mismatch {len(val_groups)} vs {len(val['targets'])}")
    validation_limit = (
        min(args.max_validation_groups, len(val_groups))
        if args.max_validation_groups else len(val_groups)
    )
    val_groups = val_groups[:validation_limit]
    val = {key: value[:validation_limit] for key, value in val.items()}
    if args.train_group_indices_file:
        path = Path(args.train_group_indices_file)
        if path.suffix == ".npy":
            selected_ids = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
        else:
            selected_ids = np.asarray(
                [int(line.strip()) for line in path.read_text().splitlines() if line.strip()],
                dtype=np.int64,
            )
        cache_rows = {int(group_id): row for row, group_id in enumerate(train_full["group_indices"])}
        if len(selected_ids) == 0 or np.any(selected_ids < 0) or np.any(selected_ids >= len(train_groups)):
            raise ValueError(f"invalid train-group-indices-file: {path}")
        selected_rows = np.asarray([cache_rows[int(group_id)] for group_id in selected_ids], dtype=np.int64)
        train_full = {key: value[selected_rows] for key, value in train_full.items()}
        limit = len(selected_ids)
    else:
        limit = min(args.max_train_groups, len(train_groups)) if args.max_train_groups else len(train_groups)
        train_full = {key: value[:limit] for key, value in train_full.items()}
    holdout = np.arange(limit) % args.holdout_every == 0
    train_ids, holdout_ids = np.flatnonzero(~holdout), np.flatnonzero(holdout)
    train_cache = {key: value[train_ids] for key, value in train_full.items()}
    holdout_cache = {key: value[holdout_ids] for key, value in train_full.items()}
    # Token files are always in original prefix group order.  Dataset-local
    # group IDs therefore map to the sliced cache positions explicitly.
    # The caches are already sliced into the train/holdout rows; dataset-local
    # token IDs refer to the same selected balanced order, not raw pkl order.
    train_ds = TokenDataset(train_cache, args.train_tokens, train_ids,
                            list(range(len(COMBINATIONS))), args.train_aux)
    holdout_ds = TokenDataset(holdout_cache, args.train_tokens, holdout_ids,
                              list(range(len(COMBINATIONS))), args.train_aux)
    val_ds = TokenDataset(val, args.validation_tokens, np.arange(len(val_groups)),
                          list(range(len(COMBINATIONS))), args.validation_aux)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True, collate_fn=collate)
    holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, collate_fn=collate)
    sample = np.load(args.train_tokens, mmap_mode="r")
    # [G,11,V,J,C,P,P] for subset-specific tokens, or [G,V,J,C,P,P]
    # for the earlier full-view probe.
    channel_axis = -3
    patch_axis = -2
    aux_dim = 0
    if args.train_aux:
        aux_sample = np.load(args.train_aux, mmap_mode="r")
        if aux_sample.ndim != 5:
            raise ValueError(f"train aux must be [G,11,V,J,D], got {aux_sample.shape}")
        aux_dim = int(aux_sample.shape[-1])
    if args.validation_aux:
        val_aux_sample = np.load(args.validation_aux, mmap_mode="r")
        if val_aux_sample.ndim != 5 or int(val_aux_sample.shape[-1]) != aux_dim:
            raise ValueError(f"validation aux shape incompatible: {val_aux_sample.shape}, aux_dim={aux_dim}")
    model = RIGRHRNetFeature(
        int(sample.shape[channel_axis]), int(sample.shape[patch_axis]),
        gated_residual=args.gated_residual,
        aux_dim=aux_dim,
        attention_pooling=args.attention_pooling,
        cross_view_relation=args.cross_view_relation,
        patch_attention=args.patch_attention,
                        geometry_biased_attention=args.geometry_biased_attention,
        explicit_view_block=args.explicit_view_block,
        correspondence_attention=args.correspondence_attention,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best.pth"
    best_metric, best_epoch, history = float("inf"), -1, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for prediction, rays, features, aux, target, _, mask in train_loader:
            prediction, rays = prediction.to(device), rays.to(device)
            features, target, mask = features.to(device), target.to(device), mask.to(device)
            if aux is not None:
                aux = aux.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(prediction, rays, features, mask, aux)
            loss = torch.nn.functional.smooth_l1_loss(output, target, beta=0.025)
            loss = loss + 0.002 * (output - prediction).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate(model, holdout_loader, device, holdout_cache["actions"])
        metric = 0.5 * (holdout_result["V3"]["action_equal_all17_mm"] + holdout_result["V4"]["action_equal_all17_mm"])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "holdout_metric_mm": float(metric), "holdout": holdout_result}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "history": history,
                        "channels": int(sample.shape[channel_axis]), "patch": int(sample.shape[patch_axis]),
                        "gated_residual": bool(args.gated_residual), "aux_dim": aux_dim,
                        "attention_pooling": bool(args.attention_pooling),
                        "cross_view_relation": bool(args.cross_view_relation),
                        "patch_attention": bool(args.patch_attention),
                        "geometry_biased_attention": bool(args.geometry_biased_attention),
                        "explicit_view_block": bool(args.explicit_view_block),
                        "correspondence_attention": bool(args.correspondence_attention)}, best_path)
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    final = evaluate(model, val_loader, device, val["actions"])
    method = "RIGR HRNet intermediate feature refiner"
    if args.attention_pooling:
        method += " + learned view-quality pooling"
    if args.cross_view_relation:
        method += " + explicit cross-view relation"
    if args.patch_attention:
        method += " + spatial patch attention"
    if args.geometry_biased_attention:
        method += " + pairwise ray-geometry attention bias"
    if args.explicit_view_block:
        method += " + explicit view qkv control"
    if args.correspondence_attention:
        method += " + cross-view patch correspondence attention"
    if args.gated_residual:
        method += " + selective trust gate"
    result = {"method": method,
              "paper_basis": ["MVGFormer (CVPR 2024)", "Epipolar Transformer (CVPR 2020)"],
              "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
              "history": history, "S9_S11_final_once": final, "args": vars(args),
              "combinations": [list(c) for c in COMBINATIONS]}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"S9_S11_final_once": final}, indent=2), flush=True)


if __name__ == "__main__":
    main()
