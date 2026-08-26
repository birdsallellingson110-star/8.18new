#!/usr/bin/env python3
"""Evaluate a clean-trained frozen H18 checkpoint on dense H36M-Occl.

This script performs no optimization or checkpoint selection.  It preserves
the original centered T=9 definition and labels it non-causal explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import train_e2_clean_temporal_residual_20260818 as h18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--validation-fused", required=True)
    parser.add_argument(
        "--validation-uncertainty",
        help="Optional N,K,J,F features required by uncertainty-aware H18.",
    )
    parser.add_argument("--validation-pkl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-centers-pkl",
        help=(
            "Optional sparse benchmark PKL whose group keys define scored "
            "centers. Dense neighboring rows still provide temporal context; "
            "sequence boundaries use replicated edge frames."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_center_windows(
    dense_meta: dict[str, np.ndarray], target_pkl: str, length: int
) -> tuple[np.ndarray, int]:
    target_meta = h18.metadata_from_pkl(target_pkl, 2021)
    dense_keys = list(
        zip(
            dense_meta["subjects"].tolist(),
            dense_meta["actions"].tolist(),
            dense_meta["subactions"].tolist(),
            dense_meta["frame_ids"].tolist(),
        )
    )
    dense_index = {tuple(map(int, key)): index for index, key in enumerate(dense_keys)}
    by_sequence: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(dense_keys):
        by_sequence.setdefault(tuple(map(int, key[:3])), []).append(index)
    positions = {}
    for sequence, indices in by_sequence.items():
        indices.sort(key=lambda index: int(dense_meta["frame_ids"][index]))
        positions[sequence] = ({index: pos for pos, index in enumerate(indices)}, indices)

    target_keys = zip(
        target_meta["subjects"].tolist(),
        target_meta["actions"].tolist(),
        target_meta["subactions"].tolist(),
        target_meta["frame_ids"].tolist(),
    )
    half = length // 2
    rows = []
    padded = 0
    for raw_key in target_keys:
        key = tuple(map(int, raw_key))
        if key not in dense_index:
            raise KeyError(f"target center absent from dense cache: {key}")
        center_index = dense_index[key]
        position_by_index, sequence_rows = positions[key[:3]]
        center_position = position_by_index[center_index]
        positions_raw = list(range(center_position - half, center_position + half + 1))
        clipped = [min(max(pos, 0), len(sequence_rows) - 1) for pos in positions_raw]
        padded += int(clipped != positions_raw)
        row = [sequence_rows[pos] for pos in clipped]
        if row[half] != center_index:
            raise RuntimeError(f"bad temporal center construction for {key}")
        rows.append(row)
    return np.asarray(rows, dtype=np.int64), padded


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = state["args"]
    window_length = int(saved["window_length"])
    hidden_dim = int(saved["hidden_dim"])
    layers = int(saved["layers"])
    residual_scale_m = float(saved["residual_scale_m"])
    if window_length != 9:
        raise ValueError(f"expected frozen H18 T=9, got T={window_length}")

    cache_path = Path(args.validation_cache).resolve()
    fused_path = Path(args.validation_fused).resolve()
    pkl_path = Path(args.validation_pkl).resolve()
    cache = np.load(cache_path, allow_pickle=False)
    fused = np.load(fused_path, mmap_mode="r")
    if fused.shape != (len(cache["targets"]), 11, 17, 3):
        raise ValueError(f"cache/fused shape mismatch: {fused.shape}")
    uncertainty_dim = int(saved.get("uncertainty_dim", 0))
    # Early uncertainty-aware checkpoints stored the gate flag and weights but
    # did not copy the runtime-derived feature dimension into checkpoint args.
    # Recover it from the gate input projection so those frozen checkpoints can
    # be evaluated without mutating them.
    if uncertainty_dim == 0 and bool(saved.get("uncertainty_gate", False)):
        gate_weight = state["state_dict"].get("uncertainty_gate.0.weight")
        if gate_weight is None:
            raise ValueError("uncertainty gate checkpoint is missing its input weight")
        uncertainty_dim = int(gate_weight.shape[1])
    uncertainty = None
    uncertainty_path = None
    if uncertainty_dim:
        if not args.validation_uncertainty:
            raise ValueError("checkpoint requires --validation-uncertainty")
        uncertainty_path = Path(args.validation_uncertainty).resolve()
        uncertainty = np.load(uncertainty_path, mmap_mode="r")
        if uncertainty.shape != (len(cache["targets"]), 11, 17, uncertainty_dim):
            raise ValueError(f"bad uncertainty shape {uncertainty.shape}")
    meta = h18.metadata_from_pkl(str(pkl_path), len(cache["targets"]))
    if not np.array_equal(cache["subjects"], meta["subjects"]):
        raise RuntimeError("validation subject order mismatch")
    if not np.array_equal(cache["actions"], meta["actions"]):
        raise RuntimeError("validation action order mismatch")
    if args.target_centers_pkl:
        windows, boundary_padded_windows = target_center_windows(
            meta, args.target_centers_pkl, window_length
        )
    else:
        windows = h18.build_windows(meta, window_length, args.frame_stride)
        boundary_padded_windows = 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = h18.TemporalPoseModel(
        window_length, hidden_dim, layers, residual_scale_m,
        camera_independent=bool(saved.get("camera_independent", False)),
        continuous_time=bool(saved.get("continuous_time", False)),
        reference_dt_s=float(saved.get("reference_dt_s", 0.1)),
        max_time_period_s=float(saved.get("max_time_period_s", 2.0)),
        uncertainty_dim=uncertainty_dim,
        uncertainty_gate=bool(saved.get("uncertainty_gate", False)),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    result = h18.evaluate(
        model, cache, fused, windows, device, window_length // 2, args.batch_size,
        meta["frame_ids"], float(saved.get("source_fps", 50.0)),
        uncertainty,
    )
    payload = {
        "method": "frozen clean-trained H18 centered temporal residual",
        "protocol": "H36M-Occl S9/S11, T=9 centered, stride=5, action-equal All-17 absolute MPJPE",
        "causal": False,
        "reporting_boundary": (
            "uses four past and four future frames; internal robustness result, "
            "not a strict comparison to GBT's latest-frame causal inference"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_clean_training_only": True,
        "checkpoint_selected_before_occlusion_evaluation": True,
        "validation_cache": str(cache_path),
        "validation_fused": str(fused_path),
        "validation_uncertainty": (
            str(uncertainty_path) if uncertainty_path else None
        ),
        "validation_pkl": str(pkl_path),
        "target_centers_pkl": (
            str(Path(args.target_centers_pkl).resolve())
            if args.target_centers_pkl else None
        ),
        "window_length": window_length,
        "frame_stride": args.frame_stride,
        "camera_independent": model.camera_independent,
        "continuous_time": model.continuous_time,
        "source_fps": float(saved.get("source_fps", 50.0)),
        "windows": int(len(windows)),
        "boundary_padded_windows": int(boundary_padded_windows),
        "result": result,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
