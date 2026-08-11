"""AMASS temporal clips encoded with the audited RUMPL ray convention."""

import glob
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def audited_axis_transform(points):
    """Apply the official-like RUMPL [x, y, z] -> [x, -z, y] transform."""
    transformed = np.asarray(points)[..., [0, 2, 1]].copy()
    transformed[..., 1] *= -1
    return transformed


def camera_rays(joints_2d, camera):
    """Build unnormalized rays whose point is the world-space camera center."""
    joints_2d = np.asarray(joints_2d, dtype=np.float32)
    rotation = np.asarray(camera["R"], dtype=np.float32)
    center = np.asarray(camera["T"], dtype=np.float32).reshape(3)
    x = (joints_2d[:, 0] - float(camera["cx"])) / float(camera["fx"])
    y = (joints_2d[:, 1] - float(camera["cy"])) / float(camera["fy"])
    camera_points = np.stack([x, y, np.ones_like(x)], axis=-1)
    world_points = (rotation.T @ camera_points.T).T + center
    centers = np.broadcast_to(center, world_points.shape)
    direction = centers - world_points
    return (
        audited_axis_transform(direction).astype(np.float32),
        audited_axis_transform(centers).astype(np.float32),
    )


class ExactTemporalClipDataset(Dataset):
    COCO_SIGMAS = np.array(
        [.026, .025, .025, .035, .035, .079, .079, .072, .072,
         .062, .062, .107, .107, .087, .087, .089, .089]
    )

    def __init__(self, pattern, num_frames=9, fixed_window=False, min_center_oks=0.0):
        self.num_frames = num_frames
        self.center_index = num_frames // 2
        self.fixed_window = fixed_window
        self.min_center_oks = min_center_oks
        self.clips = []
        files = sorted(glob.glob(pattern))
        if not files:
            raise ValueError(f"No temporal clips matched {pattern}")
        for path in files:
            with open(path, "rb") as handle:
                data = pickle.load(handle)
            for index in range(data["joints_3d"].shape[0]):
                clip = {
                        "joints_3d": data["joints_3d"][index],
                        "joints_2d": data["joints_2d_mmpose"][index],
                        "confidence": data["confs_2d_mmpose"][index],
                        "cameras": data["camera_parameters_all"][index],
                        "frame_rate": float(data["frame_rate"][index]),
                        "source": str(data["source_npz"][index]),
                        "start_frame": int(data["start_frame"][index]),
                    }
                if min_center_oks > 0:
                    detected = data["joints_2d_mmpose"][index].astype(np.float64)
                    ground_truth = data["joints_2d_amass"][index].astype(np.float64)
                    width = ground_truth[..., 0].max(-1) - ground_truth[..., 0].min(-1)
                    height = ground_truth[..., 1].max(-1) - ground_truth[..., 1].min(-1)
                    area = width * height
                    squared_distance = ((detected - ground_truth) ** 2).sum(-1)
                    oks = np.exp(
                        -squared_distance
                        / ((self.COCO_SIGMAS * 2) ** 2)
                        / (area[..., None] + 1e-9)
                        / 2
                    ).mean(-1)
                    frame_oks = np.median(oks, axis=1)
                    half = num_frames // 2
                    valid = np.flatnonzero(frame_oks >= min_center_oks)
                    valid = valid[(valid >= half) & (valid < len(frame_oks) - half)]
                    if not len(valid):
                        continue
                    clip["valid_centers"] = valid
                self.clips.append(clip)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        clip = self.clips[index]
        total_frames, num_views, num_joints, _ = clip["joints_2d"].shape
        if total_frames < self.num_frames:
            raise ValueError("Clip is shorter than the requested temporal window")
        valid_centers = clip.get("valid_centers")
        if valid_centers is not None:
            if self.fixed_window:
                center = int(valid_centers[len(valid_centers) // 2])
            else:
                center = int(np.random.choice(valid_centers))
            start = center - self.center_index
        elif self.fixed_window:
            start = (total_frames - self.num_frames) // 2
        else:
            start = np.random.randint(0, total_frames - self.num_frames + 1)
        frame_indexes = np.arange(start, start + self.num_frames)

        rays = np.empty(
            (num_joints, num_views, self.num_frames, 6), dtype=np.float32
        )
        confidence = np.empty(
            (num_joints, num_views, self.num_frames, 1), dtype=np.float32
        )
        for local_frame, frame_index in enumerate(frame_indexes):
            for view_index in range(num_views):
                direction, point = camera_rays(
                    clip["joints_2d"][frame_index, view_index],
                    clip["cameras"][view_index],
                )
                rays[:, view_index, local_frame, :3] = direction
                rays[:, view_index, local_frame, 3:6] = point
                confidence[:, view_index, local_frame] = clip["confidence"][
                    frame_index, view_index
                ].astype(np.float32)

        frame_rate = clip["frame_rate"]
        delta_t = (
            np.arange(self.num_frames, dtype=np.float32) - self.center_index
        ) / frame_rate
        target = audited_axis_transform(
            clip["joints_3d"][frame_indexes].astype(np.float32)
        )
        return {
            "rays": torch.from_numpy(rays),
            "confidence": torch.from_numpy(confidence),
            "delta_t": torch.from_numpy(delta_t),
            "target": torch.from_numpy(target),
            "clip_index": index,
        }


def random_view_collate(min_views=2, max_views=5, seed=0):
    generator = np.random.default_rng(seed)

    def collate(batch):
        available = batch[0]["rays"].shape[1]
        num_views = min(int(generator.integers(min_views, max_views + 1)), available)
        indexes = torch.from_numpy(
            np.sort(generator.choice(available, size=num_views, replace=False))
        )
        return {
            "rays": torch.stack([sample["rays"][:, indexes] for sample in batch]),
            "confidence": torch.stack(
                [sample["confidence"][:, indexes] for sample in batch]
            ),
            "delta_t": torch.stack([sample["delta_t"] for sample in batch]),
            "target": torch.stack([sample["target"] for sample in batch]),
        }

    return collate


def fixed_view_collate(num_views, seed=0):
    generator = np.random.default_rng(seed)

    def collate(batch):
        available = batch[0]["rays"].shape[1]
        indexes = torch.from_numpy(
            np.sort(generator.choice(available, size=min(num_views, available), replace=False))
        )
        return {
            "rays": torch.stack([sample["rays"][:, indexes] for sample in batch]),
            "confidence": torch.stack(
                [sample["confidence"][:, indexes] for sample in batch]
            ),
            "delta_t": torch.stack([sample["delta_t"] for sample in batch]),
            "target": torch.stack([sample["target"] for sample in batch]),
        }

    return collate
