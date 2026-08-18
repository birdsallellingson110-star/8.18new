#!/usr/bin/env python3
"""AdaFuse-inspired adaptive joint gate for H76 and Volumetric LT.

The gate only observes prediction disagreement, pelvis-relative poses, local
bone consistency, LT confidence, optional official volumetric-posterior
uncertainty, joint identity and camera-combination identity.  It
never observes GT, action or test-subject identity at inference.  Architecture
and regularization are selected by leave-one-training-subject-out CV, then one
model is fitted on S1/S5/S6/S7/S8 and evaluated once on S9/S11.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


ACTION_IDS = tuple(range(2, 17))
VOL_PREFIX = "backbone_prediction_lt_to_rumpl_control"
PARENTS = np.asarray([0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-h76-pkl", required=True)
    parser.add_argument("--train-vol-npz", required=True)
    parser.add_argument("--test-h76-pkl", required=True)
    parser.add_argument("--test-vol-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights-output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument(
        "--feature-set",
        choices=(
            "full", "no_pair_id", "no_pair_id_uncertainty",
            "no_confidence", "no_bone", "disagreement",
        ),
        default="full",
    )
    return parser.parse_args()


def parse_subject(fname: str) -> int:
    fields = fname.split("_")
    return int(fields[fields.index("s") + 1])


def load_pair(
    h76_path: str,
    vol_path: str,
    views: int,
    combos: tuple[str, ...],
) -> dict[str, np.ndarray]:
    with open(h76_path, "rb") as stream:
        h76_data = pickle.load(stream)
    vol_data = np.load(vol_path)
    h76 = np.asarray(h76_data["pred"], dtype=np.float32) * 1000.0
    h76_gt = np.asarray(h76_data["gt"], dtype=np.float32) * 1000.0
    confidence = np.asarray(h76_data["confs_2d"], dtype=np.float32)
    if len(h76) % len(combos):
        raise ValueError("H76 records are not complete camera-combination chunks")
    groups = len(h76) // len(combos)
    h76 = h76.reshape(groups, len(combos), 17, 3)
    h76_gt = h76_gt.reshape(groups, len(combos), 17, 3)
    confidence = confidence.reshape(groups, len(combos), 17)
    subjects = np.asarray([
        parse_subject(fname) for fname in h76_data["fnames"][::len(combos)]
    ], dtype=np.int64)
    vol = np.stack([
        np.asarray(vol_data[f"{VOL_PREFIX}_V{views}_{combo.replace('-', '_')}"])
        for combo in combos
    ], axis=1).astype(np.float32)
    uncertainty = {}
    for statistic in ("variance_mm2", "entropy", "peak_probability"):
        keys = [
            f"{VOL_PREFIX}_{statistic}_V{views}_{combo.replace('-', '_')}"
            for combo in combos
        ]
        if all(key in vol_data for key in keys):
            uncertainty[statistic] = np.stack(
                [np.asarray(vol_data[key]) for key in keys], axis=1
            ).astype(np.float32)
    target = np.asarray(vol_data["targets"], dtype=np.float32)
    actions = np.asarray(vol_data["actions"], dtype=np.int64)
    if h76.shape != vol.shape or target.shape != h76_gt[:, 0].shape:
        raise ValueError(
            f"shape mismatch H76={h76.shape} Vol={vol.shape} "
            f"target={target.shape} H76-GT={h76_gt.shape}"
        )
    gt_cross = np.linalg.norm(h76_gt[:, 0] - target, axis=-1)
    if np.abs(h76_gt - h76_gt[:, :1]).max() > 1e-3 or gt_cross.max() > 1e-2:
        raise ValueError("H76 and Volumetric GT alignment failed")
    result = {
        "h76": h76,
        "vol": vol,
        "target": target,
        "actions": actions,
        "subjects": subjects,
        "confidence": confidence,
        "gt_cross_mean": np.asarray(gt_cross.mean()),
        "gt_cross_max": np.asarray(gt_cross.max()),
    }
    result.update(uncertainty)
    return result


def make_features(
    data: dict[str, np.ndarray],
    canonical_bones: np.ndarray,
    feature_set: str,
    pair_count: int,
) -> np.ndarray:
    h76 = data["h76"]
    vol = data["vol"]
    groups = len(h76)
    disagreement = (h76 - vol) / 1000.0
    disagreement_norm = np.linalg.norm(disagreement, axis=-1, keepdims=True)
    h76_relative = (h76 - h76[:, :, :1]) / 1000.0
    vol_relative = (vol - vol[:, :, :1]) / 1000.0

    joint_indices = np.arange(17)
    h76_bones = h76 - h76[:, :, PARENTS]
    vol_bones = vol - vol[:, :, PARENTS]
    h76_bone_error = (
        np.linalg.norm(h76_bones, axis=-1) - canonical_bones[None, None]
    )[..., None] / 1000.0
    vol_bone_error = (
        np.linalg.norm(vol_bones, axis=-1) - canonical_bones[None, None]
    )[..., None] / 1000.0
    confidence = np.log1p(data["confidence"] * 100.0)[..., None]
    uncertainty_blocks = []
    if feature_set == "no_pair_id_uncertainty":
        required = ("variance_mm2", "entropy", "peak_probability")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(
                "volumetric uncertainty features requested but missing: "
                + ", ".join(missing)
            )
        uncertainty_blocks = [
            np.log1p(data["variance_mm2"])[..., None],
            data["entropy"][..., None],
            np.log(data["peak_probability"].clip(1e-12))[..., None],
        ]

    joint_one_hot = np.eye(17, dtype=np.float32)[None, None]
    joint_one_hot = np.broadcast_to(
        joint_one_hot, (groups, pair_count, 17, 17)
    )
    pair_one_hot = np.eye(pair_count, dtype=np.float32)[None, :, None]
    pair_one_hot = np.broadcast_to(
        pair_one_hot, (groups, pair_count, 17, pair_count)
    )
    blocks = {
        "disagreement": [disagreement, disagreement_norm, joint_one_hot],
        "no_pair_id": [
            disagreement, disagreement_norm, h76_relative, vol_relative,
            h76_bone_error, vol_bone_error, confidence, joint_one_hot,
        ],
        "no_pair_id_uncertainty": [
            disagreement, disagreement_norm, h76_relative, vol_relative,
            h76_bone_error, vol_bone_error, confidence, *uncertainty_blocks,
            joint_one_hot,
        ],
        "no_confidence": [
            disagreement, disagreement_norm, h76_relative, vol_relative,
            h76_bone_error, vol_bone_error, joint_one_hot, pair_one_hot,
        ],
        "no_bone": [
            disagreement, disagreement_norm, h76_relative, vol_relative,
            confidence, joint_one_hot, pair_one_hot,
        ],
        "full": [
            disagreement, disagreement_norm, h76_relative, vol_relative,
            h76_bone_error, vol_bone_error, confidence, joint_one_hot,
            pair_one_hot,
        ],
    }
    features = np.concatenate(blocks[feature_set], axis=-1)
    return features.astype(np.float32)


def fit_static_logits(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    pair_specific: bool,
    pair_count: int,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 101)
    h76 = data["h76"][mask]
    vol = data["vol"][mask]
    target = data["target"][mask]
    static_pair_count = pair_count if pair_specific else 1
    alphas = np.zeros((static_pair_count, 17), dtype=np.float32)
    for pair in range(static_pair_count):
        for joint in range(17):
            h76_selected = (
                h76[:, pair, joint] if pair_specific else h76[:, :, joint]
            )
            vol_selected = (
                vol[:, pair, joint] if pair_specific else vol[:, :, joint]
            )
            target_selected = (
                target[:, joint] if pair_specific else target[:, None, joint]
            )
            losses = [
                np.linalg.norm(
                    alpha * h76_selected
                    + (1.0 - alpha) * vol_selected
                    - target_selected,
                    axis=-1,
                ).mean()
                for alpha in grid
            ]
            alphas[pair, joint] = grid[int(np.argmin(losses))]
    clipped = np.clip(alphas, 0.01, 0.99)
    logits = np.log(clipped / (1.0 - clipped)).astype(np.float32)
    return logits if pair_specific else logits[0]


class AdaptiveJointGate(nn.Module):
    def __init__(
        self, feature_dim: int, hidden: int, max_delta: float,
        logits: np.ndarray, depth: int = 1, hidden2: int | None = None,
    ):
        super().__init__()
        if depth not in (1, 2):
            raise ValueError("AdaptiveJointGate depth must be 1 or 2")
        self.max_delta = max_delta
        self.register_buffer("static_logits", torch.from_numpy(logits))
        layers: list[nn.Module] = [nn.Linear(feature_dim, hidden), nn.ReLU(inplace=True)]
        if depth == 2:
            second_hidden = hidden if hidden2 is None else hidden2
            layers.extend((nn.Linear(hidden, second_hidden), nn.ReLU(inplace=True)))
        else:
            second_hidden = hidden
        layers.append(nn.Linear(second_hidden, 1))
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic = self.max_delta * torch.tanh(self.network(features).squeeze(-1))
        alpha = torch.sigmoid(self.static_logits[None] + dynamic)
        return alpha, dynamic


def standardize(
    train_features: np.ndarray, apply_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = train_features.reshape(-1, train_features.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0)
    # One-hot features are already well scaled; constant dimensions stay zero.
    std[std < 1e-6] = 1.0
    return (
        (train_features - mean) / std,
        (apply_features - mean) / std,
        np.stack([mean, std]),
    )


def train_one(
    data: dict[str, np.ndarray],
    features: np.ndarray,
    train_mask: np.ndarray,
    eval_mask: np.ndarray,
    hidden: int,
    max_delta: float,
    regularization: float,
    pair_specific_static: bool,
    pair_count: int,
    combos: tuple[str, ...],
    epochs: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor], np.ndarray]:
    torch.manual_seed(seed)
    static_logits = fit_static_logits(
        data, train_mask, pair_specific=pair_specific_static,
        pair_count=pair_count,
    )
    train_features, eval_features, normalization = standardize(
        features[train_mask], features[eval_mask]
    )
    model = AdaptiveJointGate(
        features.shape[-1], hidden, max_delta, static_logits
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_x = torch.from_numpy(train_features).to(device)
    train_h = torch.from_numpy(data["h76"][train_mask]).to(device)
    train_v = torch.from_numpy(data["vol"][train_mask]).to(device)
    train_t = torch.from_numpy(data["target"][train_mask]).to(device)
    for _ in range(epochs):
        model.train()
        alpha, dynamic = model(train_x)
        prediction = alpha[..., None] * train_h + (1.0 - alpha[..., None]) * train_v
        loss = torch.linalg.vector_norm(
            prediction - train_t[:, None], dim=-1
        ).mean() + regularization * dynamic.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_x = torch.from_numpy(eval_features).to(device)
        eval_h = torch.from_numpy(data["h76"][eval_mask]).to(device)
        eval_v = torch.from_numpy(data["vol"][eval_mask]).to(device)
        eval_t = torch.from_numpy(data["target"][eval_mask]).to(device)
        alpha, dynamic = model(eval_x)
        prediction = alpha[..., None] * eval_h + (1.0 - alpha[..., None]) * eval_v
        frame_error = torch.linalg.vector_norm(
            prediction - eval_t[:, None], dim=-1
        ).mean(-1).cpu().numpy()
    actions = data["actions"][eval_mask]
    metrics = {
        "frame_weighted_mm": float(frame_error.mean()),
        "action_equal_mm": float(np.mean([
            frame_error[actions == action].mean()
            for action in ACTION_IDS if np.any(actions == action)
        ])),
        "dynamic_abs_mean": float(dynamic.abs().mean().cpu()),
        "per_pair_action_equal_mm": {
            combo: float(np.mean([
                frame_error[actions == action, pair_index].mean()
                for action in ACTION_IDS if np.any(actions == action)
            ]))
            for pair_index, combo in enumerate(combos)
        },
        "per_action_mm": {
            str(action): float(frame_error[actions == action].mean())
            for action in ACTION_IDS if np.any(actions == action)
        },
    }
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return metrics, state, normalization


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    combos = tuple(
        "-".join(str(index + 1) for index in combo)
        for combo in itertools.combinations(range(4), args.views)
    )
    pair_count = len(combos)
    train_data = load_pair(
        args.train_h76_pkl, args.train_vol_npz, args.views, combos
    )
    test_data = load_pair(
        args.test_h76_pkl, args.test_vol_npz, args.views, combos
    )
    target_bones = train_data["target"] - train_data["target"][:, PARENTS]
    canonical_bones = np.linalg.norm(target_bones, axis=-1).mean(0)
    train_features = make_features(
        train_data, canonical_bones, args.feature_set, pair_count
    )
    test_features = make_features(
        test_data, canonical_bones, args.feature_set, pair_count
    )

    candidates = [
        {"hidden": 8, "max_delta": 1.0, "regularization": 1e-3},
        {"hidden": 16, "max_delta": 1.0, "regularization": 1e-3},
        {"hidden": 16, "max_delta": 2.0, "regularization": 1e-3},
        {"hidden": 32, "max_delta": 1.0, "regularization": 1e-3},
        {"hidden": 16, "max_delta": 1.0, "regularization": 1e-2},
    ]
    pair_specific_static = args.feature_set not in (
        "no_pair_id", "no_pair_id_uncertainty", "disagreement"
    )
    subjects = sorted(int(value) for value in np.unique(train_data["subjects"]))
    cv_results = []
    for candidate_index, candidate in enumerate(candidates):
        folds = []
        for held_out in subjects:
            metrics, _, _ = train_one(
                train_data,
                train_features,
                train_data["subjects"] != held_out,
                train_data["subjects"] == held_out,
                epochs=args.epochs,
                device=device,
                seed=args.seed + candidate_index,
                pair_specific_static=pair_specific_static,
                pair_count=pair_count,
                combos=combos,
                **candidate,
            )
            folds.append({"held_out_subject": held_out, **metrics})
        cv_results.append({
            **candidate,
            "folds": folds,
            "mean_frame_weighted_mm": float(np.mean([
                fold["frame_weighted_mm"] for fold in folds
            ])),
        })
    best = min(cv_results, key=lambda entry: entry["mean_frame_weighted_mm"])
    best_candidate = {
        key: best[key] for key in ("hidden", "max_delta", "regularization")
    }
    combined = {
        key: np.concatenate([train_data[key], test_data[key]], axis=0)
        for key in ("h76", "vol", "target", "actions", "subjects", "confidence")
    }
    combined_features = np.concatenate([train_features, test_features], axis=0)
    train_mask = np.zeros(len(combined["h76"]), dtype=bool)
    train_mask[:len(train_data["h76"])] = True
    test_mask = ~train_mask
    test_metrics, state, normalization = train_one(
        combined,
        combined_features,
        train_mask,
        test_mask,
        epochs=args.epochs,
        device=device,
        seed=args.seed + 100,
        pair_specific_static=pair_specific_static,
        pair_count=pair_count,
        combos=combos,
        **best_candidate,
    )
    result = {
        "protocol": {
            "paper_basis": "AdaFuse IJCV joint-level adaptive quality weighting",
            "fit_subjects": subjects,
            "test_subjects": sorted(
                int(value) for value in np.unique(test_data["subjects"])
            ),
            "features_exclude_gt_action_test_identity": True,
            "train_frames": int(len(train_data["h76"])),
            "test_frames": int(len(test_data["h76"])),
            "epochs": args.epochs,
            "seed": args.seed,
            "feature_set": args.feature_set,
            "feature_dim": int(train_features.shape[-1]),
            "pair_specific_static": pair_specific_static,
            "views": args.views,
            "camera_combinations": list(combos),
        },
        "candidates": cv_results,
        "selected": best_candidate,
        "test": test_metrics,
        "canonical_bone_lengths_mm": canonical_bones.tolist(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    torch.save({
        "model": state,
        "normalization": torch.from_numpy(normalization),
        "canonical_bone_lengths_mm": torch.from_numpy(canonical_bones),
        "selected": best_candidate,
        "feature_dim": int(train_features.shape[-1]),
    }, args.weights_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
