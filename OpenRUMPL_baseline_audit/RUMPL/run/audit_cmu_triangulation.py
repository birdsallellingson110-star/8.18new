#!/usr/bin/env python3
"""Audit CMU two-view triangulation with the public RUMPL data loader."""

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import _init_paths  # noqa: F401
import dataset
from core.config import config, update_config
from multiviews.triangulate import triangulate_poses


KP_STAR = (5, 6, 7, 8, 9, 10, 13, 14, 15, 16)


def camera_for_triangulation(camera):
    camera = copy.deepcopy(camera)
    if "fx" not in camera:
        camera["fx"] = camera["K"][0, 0]
        camera["fy"] = camera["K"][1, 1]
        camera["cx"] = camera["K"][0, 2]
        camera["cy"] = camera["K"][1, 2]
    if "k" not in camera:
        distortion = np.asarray(camera["distCoef"]).reshape(-1)
        camera["k"] = [distortion[0], distortion[1], distortion[4]]
        camera["p"] = [distortion[2], distortion[3]]
    return camera


def summarize(errors):
    errors = np.asarray(errors, dtype=np.float64)
    return {
        "all_kp_mm": float(np.nanmean(errors)),
        "kp_star_mm": float(np.nanmean(errors[:, KP_STAR])),
        "samples": int(errors.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    update_config(args.cfg)

    dataset_class = getattr(dataset, config.DATASET.TEST_DATASET)
    gt_dataset = dataset_class(config, config.DATASET.TEST_SUBSET, False)
    mmpose_dataset = dataset_class(
        config, config.DATASET.TEST_SUBSET, False, is_mmpose=True
    )
    if len(gt_dataset.grouping) != len(mmpose_dataset.grouping):
        raise RuntimeError(
            f"GT/MMPose grouping mismatch: {len(gt_dataset.grouping)} vs "
            f"{len(mmpose_dataset.grouping)}"
        )

    all_errors = []
    gt2d_errors = []
    by_pair = defaultdict(list)
    for gt_items, pred_items in zip(gt_dataset.grouping, mmpose_dataset.grouping):
        gt_records = [gt_dataset.db[index] for index in gt_items]
        pred_records = [mmpose_dataset.db[index] for index in pred_items]
        gt_images = [record["image"] for record in gt_records]
        pred_images = [record["image"] for record in pred_records]
        if gt_images != pred_images:
            raise RuntimeError(f"GT/MMPose image mismatch: {gt_images} vs {pred_images}")

        cameras = [camera_for_triangulation(record["camera"]) for record in gt_records]
        points = np.asarray([record["joints_2d"] for record in pred_records])
        confidences = np.asarray(
            [record["joints_2d_conf"] for record in pred_records]
        ).squeeze(-1)
        prediction = triangulate_poses(cameras, points, confidences)[0]

        ground_truth = np.asarray(gt_records[0]["joints_3d"], dtype=np.float64).copy()
        visibility = np.asarray(gt_records[0]["joints_3d_conf"]).reshape(-1) > 0
        error_mm = np.linalg.norm(prediction - ground_truth, axis=-1) * 10.0
        error_mm[~visibility] = np.nan
        all_errors.append(error_mm)

        gt_points = np.asarray([record["joints_2d"] for record in gt_records])
        gt2d_errors.append(np.linalg.norm(points - gt_points, axis=-1))
        pair = tuple(record["camera_id"] for record in gt_records)
        by_pair[pair].append(error_mm)

    report = {
        "dataset": config.DATASET.TEST_CMU_DATASET_NAME,
        "mmpose_type": config.DATASET.TEST_MMPOSE_TYPE,
        "groups": len(gt_dataset.grouping),
        "overall": summarize(all_errors),
        "mmpose_vs_gt_2d_px": {
            "mean": float(np.nanmean(gt2d_errors)),
            "median": float(np.nanmedian(gt2d_errors)),
            "kp_star_mean": float(np.nanmean(np.asarray(gt2d_errors)[:, :, KP_STAR])),
            "kp_star_median": float(
                np.nanmedian(np.asarray(gt2d_errors)[:, :, KP_STAR])
            ),
        },
        "pairs": {
            ",".join(map(str, pair)): summarize(errors)
            for pair, errors in sorted(by_pair.items())
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp.json")
        temporary.write_text(rendered + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
