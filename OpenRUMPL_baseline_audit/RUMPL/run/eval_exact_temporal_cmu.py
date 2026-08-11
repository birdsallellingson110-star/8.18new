#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.exact_temporal_rumpl import ExactTemporalRUMPL


CAMERAS = (3, 6, 12, 13, 23)
KP_STAR = (5, 6, 7, 8, 9, 10, 13, 14, 15, 16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-checkpoint", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def camera_rays(record):
    joints_2d = np.asarray(record["joints_2d"], dtype=np.float32)
    camera = record["camera"]
    rotation = np.asarray(camera["R"], dtype=np.float32)
    center = np.asarray(camera["T"], dtype=np.float32).reshape(3) / 100.0
    x = (joints_2d[:, 0] - float(camera["cx"])) / float(camera["fx"])
    y = (joints_2d[:, 1] - float(camera["cy"])) / float(camera["fy"])
    camera_points = np.stack([x, y, np.ones_like(x)], axis=-1)
    world_points = (rotation.T @ camera_points.T).T + center
    centers = np.broadcast_to(center, world_points.shape)
    return centers - world_points, centers


class CMUTemporalWindowDataset(Dataset):
    def __init__(self, annotation, num_frames=9):
        with open(annotation, "rb") as handle:
            records = pickle.load(handle)
        self.num_frames = num_frames
        self.half = num_frames // 2
        self.frames = {}
        for record in records:
            key = (record["pose_id"], int(record["image_id"]))
            self.frames.setdefault(key, {})[int(record["camera_id"])] = record

        self.centers = []
        for pose_id, frame_id in sorted(self.frames):
            window_keys = [
                (pose_id, frame_id + offset)
                for offset in range(-self.half, self.half + 1)
            ]
            if all(
                key in self.frames
                and all(camera in self.frames[key] for camera in CAMERAS)
                for key in window_keys
            ):
                self.centers.append((pose_id, frame_id))

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, index):
        pose_id, center_frame = self.centers[index]
        rays = np.empty(
            (17, len(CAMERAS), self.num_frames, 6), dtype=np.float32
        )
        confidence = np.empty(
            (17, len(CAMERAS), self.num_frames, 1), dtype=np.float32
        )
        for local_frame, offset in enumerate(
            range(-self.half, self.half + 1)
        ):
            frame = self.frames[(pose_id, center_frame + offset)]
            for view_index, camera_id in enumerate(CAMERAS):
                record = frame[camera_id]
                direction, point = camera_rays(record)
                rays[:, view_index, local_frame, :3] = direction
                rays[:, view_index, local_frame, 3:] = point
                confidence[:, view_index, local_frame] = np.asarray(
                    record["joints_2d_conf"], dtype=np.float32
                ).reshape(17, 1)

        center_record = self.frames[(pose_id, center_frame)][CAMERAS[0]]
        target = (
            np.asarray(center_record["joints_3d"], dtype=np.float32) / 100.0
        )
        delta_t = (
            np.arange(-self.half, self.half + 1, dtype=np.float32) / 30.0
        )
        return {
            "rays": torch.from_numpy(rays),
            "confidence": torch.from_numpy(confidence),
            "delta_t": torch.from_numpy(delta_t),
            "target": torch.from_numpy(target),
        }


@torch.no_grad()
def evaluate(model, loader, view_indexes, device, no_temporal):
    predictions = []
    targets = []
    for batch in loader:
        prediction = model(
            batch["rays"][:, :, view_indexes].to(device),
            batch["confidence"][:, :, view_indexes].to(device),
            batch["delta_t"].to(device),
            no_temporal=no_temporal,
        )
        predictions.append(prediction.cpu())
        targets.append(batch["target"])
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    error = torch.linalg.vector_norm(prediction - target, dim=-1) * 1000.0
    return {
        "all17_mm": float(error.mean()),
        "kp_star_mm": float(error[:, KP_STAR].mean()),
        "per_joint_mm": error.mean(dim=0).tolist(),
    }


def main():
    args = parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.temporal_checkpoint, map_location="cpu")
    train_args = checkpoint["args"]
    model = ExactTemporalRUMPL(
        train_args["config"],
        train_args["checkpoint"],
        temporal_depth=train_args["depth"],
        freeze_backbone=True,
        motion_only=train_args.get("motion_only", False),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    dataset = CMUTemporalWindowDataset(
        args.annotation, num_frames=train_args["frames"]
    )
    if not len(dataset):
        raise ValueError("No complete temporal windows found")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    result = {
        "annotation": args.annotation,
        "checkpoint": args.temporal_checkpoint,
        "frames": train_args["frames"],
        "samples": len(dataset),
        "combinations": {},
    }
    for num_views in range(2, len(CAMERAS) + 1):
        for combination in itertools.combinations(range(len(CAMERAS)), num_views):
            camera_ids = tuple(CAMERAS[index] for index in combination)
            key = "_".join(map(str, camera_ids))
            baseline = evaluate(model, loader, combination, device, True)
            temporal = evaluate(model, loader, combination, device, False)
            result["combinations"][key] = {
                "views": num_views,
                "cameras": camera_ids,
                "baseline": baseline,
                "temporal": temporal,
                "delta_all17_mm": temporal["all17_mm"] - baseline["all17_mm"],
                "delta_kp_star_mm": (
                    temporal["kp_star_mm"] - baseline["kp_star_mm"]
                ),
            }
            print(key, json.dumps(result["combinations"][key]), flush=True)

    summary = {}
    for num_views in range(2, len(CAMERAS) + 1):
        rows = [
            row
            for row in result["combinations"].values()
            if row["views"] == num_views
        ]
        summary[str(num_views)] = {
            "combinations": len(rows),
            "baseline_all17_mm": float(
                np.mean([row["baseline"]["all17_mm"] for row in rows])
            ),
            "temporal_all17_mm": float(
                np.mean([row["temporal"]["all17_mm"] for row in rows])
            ),
            "delta_all17_mm": float(
                np.mean([row["delta_all17_mm"] for row in rows])
            ),
            "baseline_kp_star_mm": float(
                np.mean([row["baseline"]["kp_star_mm"] for row in rows])
            ),
            "temporal_kp_star_mm": float(
                np.mean([row["temporal"]["kp_star_mm"] for row in rows])
            ),
            "delta_kp_star_mm": float(
                np.mean([row["delta_kp_star_mm"] for row in rows])
            ),
            "improved_all17": sum(row["delta_all17_mm"] < 0 for row in rows),
            "improved_kp_star": sum(
                row["delta_kp_star_mm"] < 0 for row in rows
            ),
        }
    result["summary"] = summary
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="ascii") as handle:
        json.dump(result, handle, indent=2)
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
