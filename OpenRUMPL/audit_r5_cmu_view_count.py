#!/usr/bin/env python3
"""Evaluate the exact R5 snapshot on CMU with different view counts."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()

    root = args.repo / "RUMPL"
    os.chdir(root)
    sys.path.insert(0, str(root / "lib"))
    from core.config import config, update_config
    import dataset

    update_config(str(args.config))
    spec = importlib.util.spec_from_file_location("r5_model_snapshot", args.model_snapshot)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model = model_module.get_multiview_rumpl_net(config, is_train=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    validation = dataset.multiview_cmu_panoptic_rumpl(
        config, config.DATASET.TEST_SUBSET, False, None, is_mmpose=True
    )
    count = min(args.samples, len(validation))
    with torch.no_grad():
        for n_views in (2, 3, 4, 5):
            validation.n_views = n_views
            np.random.seed(0)
            predictions = []
            targets = []
            ray_batches = []
            target_batches = []
            for index in range(count):
                _, _, target, rays, _, _ = validation[index]
                ray_batches.append(rays)
                target_batches.append(target)
                if len(ray_batches) == 64 or index == count - 1:
                    batch = torch.stack(ray_batches).to(device)
                    predictions.append(model(batch, is_training=False).cpu().numpy())
                    targets.append(torch.stack(target_batches).numpy())
                    ray_batches.clear()
                    target_batches.clear()
            prediction = np.concatenate(predictions)
            target = np.concatenate(targets)
            error = np.linalg.norm(prediction - target, axis=-1) * 1000.0
            relative_prediction = prediction - prediction[:, :1]
            relative_target = target - target[:, :1]
            relative_error = np.linalg.norm(
                relative_prediction - relative_target, axis=-1
            ) * 1000.0
            print(
                n_views,
                "views absolute mean/median mm",
                round(float(error.mean()), 3),
                round(float(np.median(error)), 3),
                "relative mean/median mm",
                round(float(relative_error.mean()), 3),
                round(float(np.median(relative_error)), 3),
            )


if __name__ == "__main__":
    main()
