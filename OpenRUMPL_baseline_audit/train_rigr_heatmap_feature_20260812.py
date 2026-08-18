#!/usr/bin/env python3
"""Train a small geometry--heatmap refiner on the frozen H76 predictions.

This is the first executable P2 probe after the P0 local-mode ceiling.  It is
deliberately an *offline* implementation: the H76 prediction and calibrated
rays are frozen, while the compact HRNet top-K heatmap evidence is read from
the already exported detector outputs.  The model uses view-shared attention,
so camera IDs and a fixed camera layout never enter the network.

It follows the useful part of MVGFormer/Epipolar Transformer (a 3-D query is
conditioned on per-view image evidence) without pretending that the missing
raw image feature maps are available.  If this compact evidence version does
not beat H76 on an internal holdout, the next experiment must export and use
real HRNet intermediate feature maps rather than adding more ray-only blocks.
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

from diagnose_h76_multiview_bottleneck_20260808 import build_four_view_groups


COMBINATIONS = tuple(
    combo for n in (2, 3, 4) for combo in itertools.combinations(range(4), n)
)
LOWER_SWAP = np.asarray([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
H36M_TO_COCO = {
    1: 12, 2: 14, 3: 16, 4: 11, 5: 13, 6: 15,
    9: 0, 11: 5, 12: 7, 13: 9, 14: 6, 15: 8, 16: 10,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", nargs="+", required=True)
    p.add_argument("--validation-cache", required=True)
    p.add_argument("--train-pkl", required=True)
    p.add_argument("--validation-pkl", required=True)
    p.add_argument("--train-topk-shards", nargs="+", required=True)
    p.add_argument("--validation-topk-shards", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-train-groups", type=int, default=20000)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_cache(paths: list[str]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in paths]
    keys = ("group_indices", "actions", "subjects", "predictions", "targets", "rays")
    arrays = {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}
    order = np.argsort(arrays["group_indices"])
    return {key: value[order] for key, value in arrays.items()}


class TopKStore:
    """Load compact top-K heatmap records into direct record-index arrays."""

    def __init__(self, paths: list[str], topk: int) -> None:
        records = []
        for path in paths:
            with np.load(path) as source:
                records.append({
                    "indices": source["record_indices"].copy(),
                    "xy": source["candidate_xy"][:, :, :topk].copy(),
                    "scores": source["candidate_scores"][:, :, :topk].copy(),
                })
        max_index = max(int(item["indices"].max()) for item in records)
        self.xy = np.zeros((max_index + 1, 17, topk, 2), dtype=np.float32)
        self.scores = np.zeros((max_index + 1, 17, topk), dtype=np.float32)
        self.present = np.zeros(max_index + 1, dtype=np.bool_)
        for item in records:
            self.xy[item["indices"]] = item["xy"]
            self.scores[item["indices"]] = item["scores"]
            self.present[item["indices"]] = True
        if not self.present.all():
            missing = np.flatnonzero(~self.present)
            raise RuntimeError(f"top-K store has missing records, first={missing[:8].tolist()}")
        self.topk = topk

    def h36m(self, record_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return Vx17xKx2 and Vx17xK in RUMPL H36M order."""
        xy = self.xy[np.asarray(record_indices)]
        scores = self.scores[np.asarray(record_indices)]
        out_xy = np.zeros_like(xy)
        out_scores = np.zeros_like(scores)
        for h36m_joint, coco_joint in H36M_TO_COCO.items():
            out_xy[:, h36m_joint] = xy[:, coco_joint]
            out_scores[:, h36m_joint] = scores[:, coco_joint]
        out_xy[:, 0] = xy[:, [11, 12]].mean(axis=1)
        out_scores[:, 0] = scores[:, [11, 12]].mean(axis=1)
        out_xy[:, 8] = xy[:, [3, 4, 5, 6]].mean(axis=1)
        out_scores[:, 8] = scores[:, [3, 4, 5, 6]].mean(axis=1)
        out_xy[:, 10] = xy[:, [0, 1, 2, 3, 4]].mean(axis=1)
        out_scores[:, 10] = scores[:, [0, 1, 2, 3, 4]].mean(axis=1)
        out_xy[:, 7] = 0.5 * (out_xy[:, 0] + out_xy[:, 8])
        out_scores[:, 7] = 0.5 * (out_scores[:, 0] + out_scores[:, 8])
        return out_xy, out_scores


class CandidateDataset(Dataset):
    def __init__(self, cache: dict[str, np.ndarray], groups: list[list[int]], store: TopKStore,
                 group_indices: np.ndarray, combo_ids: list[int]) -> None:
        self.cache = cache
        self.groups = groups
        self.store = store
        self.group_indices = np.asarray(group_indices, dtype=np.int64)
        self.combo_ids = np.asarray(combo_ids, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.group_indices) * len(self.combo_ids)

    def __getitem__(self, index: int):
        local_group = index // len(self.combo_ids)
        combo_id = int(self.combo_ids[index % len(self.combo_ids)])
        group_id = int(self.group_indices[local_group])
        combo = COMBINATIONS[combo_id]
        record_ids = self.groups[group_id]
        xy, scores = self.store.h36m(record_ids)
        prediction = self.cache["predictions"][local_group, combo_id].astype(np.float32)
        # `local_group` indexes the selected cache view, while group_id is the
        # original pkl group.  The cache is subsetted in main with the same
        # selected order, so this mapping is intentional and explicit.
        rays = self.cache["rays"][local_group].astype(np.float32)
        target = self.cache["targets"][local_group].astype(np.float32)
        views = np.asarray(combo, dtype=np.int64)
        return (
            prediction, rays[:, views], xy[views], scores[views], target,
            np.int64(combo_id),
        )


def collate_pad_views(batch):
    """Pad mixed V2/V3/V4 samples to four views and return a validity mask."""
    batch_size = len(batch)
    prediction, rays, xy, scores, target, combo = zip(*batch)
    max_views = 4
    padded_rays = np.zeros((batch_size, 17, max_views, 7), dtype=np.float32)
    padded_xy = np.zeros((batch_size, max_views, 17, xy[0].shape[2], 2), dtype=np.float32)
    padded_scores = np.zeros((batch_size, max_views, 17, scores[0].shape[2]), dtype=np.float32)
    view_mask = np.zeros((batch_size, max_views), dtype=np.bool_)
    for row, (r, x, s) in enumerate(zip(rays, xy, scores)):
        n_views = r.shape[1]
        padded_rays[row, :, :n_views] = r
        padded_xy[row, :n_views] = x
        padded_scores[row, :n_views] = s
        view_mask[row, :n_views] = True
    return (
        torch.from_numpy(np.stack(prediction)), torch.from_numpy(padded_rays),
        torch.from_numpy(padded_xy), torch.from_numpy(padded_scores),
        torch.from_numpy(np.stack(target)), torch.from_numpy(np.stack(combo)),
        torch.from_numpy(view_mask),
    )


class RIGRHeatmapFeature(nn.Module):
    """View-shared heatmap evidence encoder and joint geometry refiner."""

    def __init__(self, topk: int = 8, d_model: int = 128) -> None:
        super().__init__()
        # Per view/joint: ray direction, anchor point, confidence, candidate
        # offsets (K*2), candidate scores (K), and 3-D query (3).
        input_dim = 3 + 3 + 1 + topk * 2 + topk + 3
        self.joint_embedding = nn.Parameter(torch.zeros(17, d_model))
        self.view_encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        layer_view = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.view_transformer = nn.TransformerEncoder(layer_view, num_layers=2)
        layer_joint = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.joint_transformer = nn.TransformerEncoder(layer_joint, num_layers=2)
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(6), nn.Linear(6, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 3))
        # Identity at initialization: this is a refiner around frozen H76,
        # not an unconstrained replacement with a hidden baseline change.
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, prediction: torch.Tensor, rays: torch.Tensor,
                candidate_xy: torch.Tensor, candidate_scores: torch.Tensor,
                view_mask: torch.Tensor | None = None) -> torch.Tensor:
        # prediction B,J,3; rays B,J,V,7; candidates B,V,J,K,2.
        b, j, v, _ = rays.shape
        xy = candidate_xy.permute(0, 2, 1, 3, 4)  # B,J,V,K,2
        scores = candidate_scores.permute(0, 2, 1, 3)  # B,J,V,K
        offsets = (xy - xy[..., :1, :]) / 500.0
        direction = rays[..., :3]
        anchor = rays[..., 3:6]
        conf = rays[..., 6:7].clamp(0, 1)
        query = prediction[:, :, None, :].expand(-1, -1, v, -1)
        per_view = torch.cat(
            (direction, anchor, conf, offsets.flatten(-2), scores, query), dim=-1
        )
        tokens = self.view_encoder(per_view.reshape(b * j, v, -1))
        if view_mask is None:
            tokens = self.view_transformer(tokens)
            fused = tokens.mean(dim=1)
        else:
            token_mask = (~view_mask.bool())[:, None, :].expand(b, j, v).reshape(b * j, v)
            tokens = self.view_transformer(tokens, src_key_padding_mask=token_mask)
            valid = view_mask[:, None, :, None].to(tokens.dtype)
            fused = (tokens.reshape(b, j, v, -1) * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)
            fused = fused.reshape(b * j, -1)
        fused = fused.reshape(b, j, -1)
        root_relative = prediction - prediction[:, :1]
        fused = fused + self.query_encoder(
            torch.cat((prediction, root_relative), dim=-1)
        ) + self.joint_embedding[None]
        fused = self.joint_transformer(fused)
        delta = 0.25 * torch.tanh(self.output(fused))
        return prediction + delta


def action_equal(values: np.ndarray, actions: np.ndarray) -> float:
    names = []
    for action in sorted(set(int(item) for item in actions)):
        selected = values[actions == action]
        if len(selected):
            names.append(float(selected.mean()))
    return float(np.mean(names))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             actions: np.ndarray, n_groups: int) -> dict:
    model.eval()
    errors, combo_ids = [], []
    for prediction, rays, xy, scores, target, combo, view_mask in loader:
        output = model(
            prediction.to(device, non_blocking=True),
            rays.to(device, non_blocking=True),
            xy.to(device, non_blocking=True),
            scores.to(device, non_blocking=True),
            view_mask.to(device, non_blocking=True),
        )
        errors.append(torch.linalg.vector_norm(output - target.to(device), dim=-1).mean(dim=-1).cpu().numpy())
        combo_ids.append(combo.numpy())
    values = np.concatenate(errors) * 1000.0
    combo_ids = np.concatenate(combo_ids)
    result = {}
    for n in (2, 3, 4):
        selected = np.isin(combo_ids, [i for i, c in enumerate(COMBINATIONS) if len(c) == n])
        n_values = values[selected]
        n_actions = np.repeat(actions, len([c for c in COMBINATIONS if len(c) == n]))
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

    train_cache_full = load_cache(args.train_cache)
    val_cache = load_cache([args.validation_cache])
    with open(args.train_pkl, "rb") as handle:
        train_records = pickle.load(handle)
    with open(args.validation_pkl, "rb") as handle:
        val_records = pickle.load(handle)
    train_groups = build_four_view_groups(train_records)
    val_groups = build_four_view_groups(val_records)
    if len(train_groups) != len(train_cache_full["targets"]):
        raise ValueError(f"train group/cache mismatch {len(train_groups)} vs {len(train_cache_full['targets'])}")
    if len(val_groups) != len(val_cache["targets"]):
        raise ValueError(f"validation group/cache mismatch {len(val_groups)} vs {len(val_cache['targets'])}")

    limit = min(args.max_train_groups, len(train_groups)) if args.max_train_groups else len(train_groups)
    selected_groups = np.arange(limit, dtype=np.int64)
    # Keep cache rows and pkl groups aligned after truncation.
    train_cache = {key: value[selected_groups] for key, value in train_cache_full.items()}
    train_group_list = [train_groups[int(i)] for i in selected_groups]
    train_store = TopKStore(args.train_topk_shards, args.topk)
    val_store = TopKStore(args.validation_topk_shards, args.topk)

    holdout = np.arange(limit) % 10 == 0
    train_indices = np.flatnonzero(~holdout)
    holdout_indices = np.flatnonzero(holdout)
    train_view = {key: value[train_indices] for key, value in train_cache.items()}
    holdout_view = {key: value[holdout_indices] for key, value in train_cache.items()}
    train_groups_view = [train_group_list[int(i)] for i in train_indices]
    holdout_groups_view = [train_group_list[int(i)] for i in holdout_indices]
    all_combo_ids = list(range(len(COMBINATIONS)))
    train_ds = CandidateDataset(train_view, train_groups_view, train_store,
                                np.arange(len(train_indices)), all_combo_ids)
    holdout_ds = CandidateDataset(holdout_view, holdout_groups_view, train_store,
                                  np.arange(len(holdout_indices)), all_combo_ids)
    val_ds = CandidateDataset(val_cache, val_groups, val_store,
                              np.arange(len(val_groups)), all_combo_ids)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True,
                              collate_fn=collate_pad_views)
    holdout_loader = DataLoader(holdout_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True,
                                collate_fn=collate_pad_views)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=collate_pad_views)

    model = RIGRHeatmapFeature(args.topk).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model_best.pth"
    best_metric, best_epoch, history = float("inf"), -1, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for prediction, rays, xy, scores, target, _, view_mask in train_loader:
            prediction = prediction.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            xy = xy.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            view_mask = view_mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(prediction, rays, xy, scores, view_mask)
            loss = torch.nn.functional.smooth_l1_loss(output, target, beta=0.025)
            loss = loss + 0.002 * (output - prediction).abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        holdout_result = evaluate(model, holdout_loader, device,
                                  holdout_view["actions"], len(holdout_indices))
        metric = 0.5 * (holdout_result["V3"]["action_equal_all17_mm"]
                        + holdout_result["V4"]["action_equal_all17_mm"])
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "holdout_metric_mm": float(metric), "holdout": holdout_result}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            torch.save({"state_dict": model.state_dict(), "topk": args.topk,
                        "epoch": epoch, "history": history}, best_path)

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    validation_result = evaluate(model, val_loader, device, val_cache["actions"], len(val_groups))
    result = {
        "method": "RIGR compact heatmap feature refiner (P2 probe)",
        "paper_basis": ["MVGFormer (CVPR 2024)", "Epipolar Transformer (CVPR 2020)"],
        "best_epoch": best_epoch, "best_holdout_metric_mm": best_metric,
        "history": history, "S9_S11_final_once": validation_result,
        "args": vars(args), "combinations": [list(c) for c in COMBINATIONS],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"S9_S11_final_once": validation_result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
