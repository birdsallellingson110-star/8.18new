"""Temporal-window adapter for the real-H36M RUMPL dataset.

The underlying :class:`MultiViewH36M_RUMPL` dataset stores one synchronized
multi-camera instant per item.  This adapter groups those instants by H36M
sequence and returns windows without crossing subject/action/subaction
boundaries.  Camera selection is deliberately *not* performed here: all
frames keep the same camera order, so the temporal trainer can sample one
camera subset and apply it to the complete sequence.

The official bundle used by this project is already sampled every five raw
H36M frames.  Therefore ``frame_stride=5`` means adjacent observations on the
available training grid, not five-window subsampling performed by this class.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TemporalWindow:
    """Indices of synchronized groups forming one temporal example."""

    group_indices: Tuple[int, ...]
    frame_ids: Tuple[int, ...]
    sequence_key: Tuple[int, int, int, Tuple[int, ...]]


class _FrameRayCacheBuilder(Dataset):
    """Worker-side extraction of only the tensors used by temporal RUMPL."""

    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base.grouping)

    def __getitem__(self, index):
        sample = self.base[index]
        return index, sample[2], sample[3]


def _group_metadata(base_dataset, group_index: int):
    group = base_dataset.grouping[group_index]
    valid = [record_index for record_index in group if record_index >= 0]
    if not valid:
        raise ValueError(f"empty H36M group at index {group_index}")
    record = base_dataset.db[valid[0]]
    camera_ids = tuple(int(base_dataset.db[index]["camera_id"]) for index in valid)
    sequence_key = (
        int(record["subject"]),
        int(record["action"]),
        int(record["subaction"]),
        camera_ids,
    )
    return sequence_key, int(record["image_id"])


def build_temporal_windows(
    base_dataset,
    window_length: int = 9,
    frame_stride: int = 5,
    window_step: int = 1,
) -> List[TemporalWindow]:
    """Build strictly consecutive windows on the processed H36M grid.

    A window is retained only when every neighboring pair differs by exactly
    ``frame_stride``.  This prevents accidental windows across missing frames
    or the sparse 65-raw-frame validation protocol.
    """

    if window_length < 1:
        raise ValueError("window_length must be positive")
    if frame_stride < 1 or window_step < 1:
        raise ValueError("frame_stride and window_step must be positive")

    sequences: Dict[
        Tuple[int, int, int, Tuple[int, ...]], List[Tuple[int, int]]
    ] = defaultdict(list)
    for group_index in range(len(base_dataset.grouping)):
        key, frame_id = _group_metadata(base_dataset, group_index)
        sequences[key].append((frame_id, group_index))

    windows: List[TemporalWindow] = []
    for sequence_key, entries in sequences.items():
        entries.sort()
        for start in range(0, len(entries) - window_length + 1, window_step):
            candidate = entries[start : start + window_length]
            frame_ids = tuple(frame_id for frame_id, _ in candidate)
            if any(
                right - left != frame_stride
                for left, right in zip(frame_ids, frame_ids[1:])
            ):
                continue
            windows.append(
                TemporalWindow(
                    group_indices=tuple(group_index for _, group_index in candidate),
                    frame_ids=frame_ids,
                    sequence_key=sequence_key,
                )
            )
    return windows


class TemporalH36MRUMPL(Dataset):
    """Wrap a frame-level RUMPL dataset and return ``T``-frame tensors.

    Returned tensor layout is time-major before DataLoader batching:

    - ``middle_points``: ``(T, J, 1, 3)``
    - ``closest_points``: ``(T, J, V, C)``
    - ``target``: ``(T, J, 3)``
    - ``rays``: ``(T, J, V, C)``
    - ``joints_2d``: ``(T, J, V, C)``

    ``metadata`` is a dictionary with the per-frame metadata list plus stable
    sequence and frame identifiers.
    """

    def __init__(
        self,
        base_dataset,
        window_length: int = 9,
        frame_stride: int = 5,
        window_step: int = 1,
        cache_frames: bool = False,
        cache_workers: int = 0,
    ):
        self.base = base_dataset
        # The frame-level dataset otherwise shuffles cameras independently in
        # every __getitem__.  Temporal camera sampling must happen once per
        # sequence in the trainer/model, so preserve canonical camera order.
        self.base.max_random_n_views = None
        self.windows = build_temporal_windows(
            base_dataset,
            window_length=window_length,
            frame_stride=frame_stride,
            window_step=window_step,
        )
        self.window_length = window_length
        self.frame_stride = frame_stride
        self.cache_frames = bool(cache_frames)
        self.cache_workers = int(cache_workers)
        if self.cache_workers < 0:
            raise ValueError("cache_workers must be non-negative")
        self._frame_cache = None
        if self.cache_frames:
            self._build_frame_cache()
        if not self.windows:
            observed = []
            by_sequence = defaultdict(list)
            for index in range(len(base_dataset.grouping)):
                key, frame_id = _group_metadata(base_dataset, index)
                by_sequence[key].append(frame_id)
            for frame_ids in by_sequence.values():
                frame_ids.sort()
                observed.extend(b - a for a, b in zip(frame_ids, frame_ids[1:]))
            common = Counter(observed).most_common(5)
            raise ValueError(
                "no temporal windows were found; "
                f"requested stride={frame_stride}, common observed strides={common}"
            )

    def _build_frame_cache(self):
        """Precompute each synchronized frame once before DataLoader forking.

        ``MultiViewH36M_RUMPL.__getitem__`` performs camera projection,
        closest-point-on-skew-lines and tensor conversion.  A sliding T=9
        window otherwise repeats that work nine times for the same frame.  The
        cache keeps the exact tensors/targets and discards the large per-frame
        diagnostic metadata, which is not consumed by either temporal
        trainer or evaluator.  Building before worker processes are forked
        lets them share these read-only pages via copy-on-write.
        """

        count = len(self.base.grouping)
        print(
            f"[temporal-cache] precomputing {count} synchronized frames "
            f"with workers={self.cache_workers}",
            flush=True,
        )

        builder = _FrameRayCacheBuilder(self.base)
        loader = DataLoader(
            builder,
            batch_size=64,
            shuffle=False,
            num_workers=self.cache_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        targets = None
        rays = None
        done = 0
        for indices, batch_targets, batch_rays in loader:
            if targets is None:
                targets = torch.empty(
                    (count,) + tuple(batch_targets.shape[1:]),
                    dtype=batch_targets.dtype,
                )
                rays = torch.empty(
                    (count,) + tuple(batch_rays.shape[1:]),
                    dtype=batch_rays.dtype,
                )
            targets[indices] = batch_targets
            rays[indices] = batch_rays
            done += int(indices.shape[0])
            if done == count or done % 8192 < int(indices.shape[0]):
                print(f"[temporal-cache] {done}/{count}", flush=True)
        if targets is None or rays is None:
            raise RuntimeError("cannot build an empty temporal frame cache")
        self._frame_cache = (targets, rays)
        print("[temporal-cache] ready", flush=True)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        if self._frame_cache is None:
            frame_samples = [self.base[i] for i in window.group_indices]
        else:
            frame_samples = [
                (
                    torch.empty(0),
                    torch.empty(0),
                    self._frame_cache[0][i],
                    self._frame_cache[1][i],
                    {},
                    torch.empty(0),
                )
                for i in window.group_indices
            ]
        middle, closest, target, rays, metadata, joints_2d = zip(*frame_samples)
        return (
            torch.stack(middle, dim=0),
            torch.stack(closest, dim=0),
            torch.stack(target, dim=0),
            torch.stack(rays, dim=0),
            {
                "frames": list(metadata),
                "sequence_key": window.sequence_key,
                "frame_ids": window.frame_ids,
                "frame_stride": self.frame_stride,
            },
            torch.stack(joints_2d, dim=0),
        )


def collate_temporal_h36m(batch):
    """Collate temporal tensors while keeping rich RUMPL metadata as a list."""

    middle, closest, target, rays, metadata, joints_2d = zip(*batch)
    return (
        torch.stack(middle, dim=0),
        torch.stack(closest, dim=0),
        torch.stack(target, dim=0),
        torch.stack(rays, dim=0),
        list(metadata),
        torch.stack(joints_2d, dim=0),
    )
