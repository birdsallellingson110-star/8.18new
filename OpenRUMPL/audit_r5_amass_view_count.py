#!/usr/bin/env python3
"""Evaluate R5 on its synthetic AMASS distribution with 2 to 5 views."""

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

    validation = dataset.multiview_amass_rumpl(
        config, "validation", True, None
    )
    count = min(args.samples, len(validation))
    np.random.seed(0)
    rays_all = []
    targets = []
    for index in range(count):
        _, _, target, rays, _, _ = validation[index]
        rays_all.append(rays)
        targets.append(target)
    rays_all = torch.stack(rays_all)
    target = torch.stack(targets).numpy()

    with torch.no_grad():
        for n_views in (2, 3, 4, 5):
            predictions = []
            for start in range(0, count, 64):
                batch = rays_all[start : start + 64, :, :n_views].to(device)
                predictions.append(model(batch, is_training=False).cpu().numpy())
            prediction = np.concatenate(predictions)
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
