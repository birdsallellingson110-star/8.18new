#!/usr/bin/env python3
"""Evaluate a frozen H36M E2 scorer on a four-camera CMU hypothesis cache."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch


AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(AUDIT))

import train_e2_v234_universal_20260812 as trainer  # noqa: E402
from train_h76_set_transformer_utility_20260811 import (  # noqa: E402
    SetTransformerJointUtility,
)


KP_STAR_H36M = (2, 3, 5, 6, 11, 12, 13, 14, 15, 16)
COMBINATIONS = tuple(
    combo
    for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-output")
    parser.add_argument("--temperature-v2", type=float, default=0.4)
    parser.add_argument("--temperature-v3", type=float, default=1.8)
    parser.add_argument("--temperature-v4", type=float, default=1.8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def metrics(pose: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = torch.linalg.vector_norm(pose - target, dim=-1) * 1000.0
    return {
        "all17_mm": float(error.mean().item()),
        "kpstar_mm": float(error[:, KP_STAR_H36M].mean().item()),
        "median_frame_all17_mm": float(error.mean(dim=-1).median().item()),
        "p90_frame_all17_mm": float(
            torch.quantile(error.mean(dim=-1), 0.9).item()
        ),
    }


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache).resolve()
    manifest_path = cache_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    camera_views = manifest["camera_views"]
    if len(camera_views) != 4:
        raise ValueError(f"expected four camera views, got {camera_views}")

    source = np.load(cache_path, allow_pickle=False)
    predictions_np = source["predictions"]
    if predictions_np.shape[1:] != (22, 17, 3):
        raise ValueError(f"expected a 22-candidate cache, got {predictions_np.shape}")
    device = torch.device(f"cuda:{args.gpu}")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = SetTransformerJointUtility(
        state["mean"], state["std"], state["attention_depth"],
        stage_heads=state.get("stage_heads", False),
    ).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval()

    # The current C2 cache is exactly two copies of the 11 legal subsets:
    # learned RUMPL candidates followed by confidence/IRLS candidates.
    trainer.ALL_CANDIDATE_COMBINATIONS = COMBINATIONS + COMBINATIONS
    temperatures = {
        2: args.temperature_v2,
        3: args.temperature_v3,
        4: args.temperature_v4,
    }
    stores = {
        combo: {"baseline": [], "soft": [], "hard": [], "target": []}
        for combo in COMBINATIONS
    }
    with torch.inference_mode():
        for start in range(0, len(predictions_np), args.batch_size):
            stop = min(start + args.batch_size, len(predictions_np))
            predictions = torch.from_numpy(predictions_np[start:stop]).to(device)
            targets = torch.from_numpy(source["targets"][start:stop]).to(device)
            rays = torch.from_numpy(source["rays"][start:stop]).to(device)
            for combo in COMBINATIONS:
                scores, _, _, candidates, baseline_local = trainer.predict_task(
                    model, predictions, targets, rays, combo
                )
                weights = torch.softmax(-scores / temperatures[len(combo)], dim=-1)
                soft = torch.einsum("bjc,bcjd->bjd", weights, candidates)
                hard_index = scores.argmin(dim=-1)
                hard = candidates.permute(0, 2, 1, 3).gather(
                    2, hard_index[..., None, None].expand(-1, -1, 1, 3)
                ).squeeze(2)
                stores[combo]["baseline"].append(candidates[:, baseline_local].cpu())
                stores[combo]["soft"].append(soft.cpu())
                stores[combo]["hard"].append(hard.cpu())
                stores[combo]["target"].append(targets.cpu())

    rows = []
    prediction_payload = {
        "frame_keys": source["frame_keys"],
        "camera_views": np.asarray(camera_views, dtype=np.int16),
    }
    for combo in COMBINATIONS:
        physical = tuple(camera_views[index] for index in combo)
        values = {
            key: torch.cat(chunks) for key, chunks in stores[combo].items()
        }
        row = {
            "local_slots": list(combo),
            "physical_cameras": list(physical),
            "view_count": len(combo),
            "n_frames": len(values["target"]),
            "baseline": metrics(values["baseline"], values["target"]),
            "soft": metrics(values["soft"], values["target"]),
            "hard": metrics(values["hard"], values["target"]),
        }
        rows.append(row)
        key = "cams_" + "_".join(str(item) for item in physical)
        prediction_payload[f"{key}_baseline"] = values["baseline"].numpy()
        prediction_payload[f"{key}_soft"] = values["soft"].numpy()
        prediction_payload[f"{key}_hard"] = values["hard"].numpy()
        prediction_payload[f"{key}_target"] = values["target"].numpy()

    summary = {}
    for count in (2, 3, 4):
        active = [row for row in rows if row["view_count"] == count]
        summary[f"V{count}"] = {
            mode: {
                metric_name: float(np.mean([
                    row[mode][metric_name] for row in active
                ]))
                for metric_name in ("all17_mm", "kpstar_mm")
            }
            for mode in ("baseline", "soft", "hard")
        }

    payload = {
        "protocol": "H36M-only training -> zero-shot CMU pose5/pose6",
        "metric": "absolute MPJPE without alignment; mean over camera combinations",
        "uses_cmu_targets_for_scoring_or_selection": False,
        "cache": str(cache_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "camera_views": camera_views,
        "temperatures": {f"V{k}": value for k, value in temperatures.items()},
        "summary": summary,
        "per_combination": rows,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    prediction_output = Path(
        args.prediction_output or output.with_suffix(".predictions.npz")
    ).resolve()
    np.savez_compressed(prediction_output, **prediction_payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
