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
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TemporalWindow:
    """Indices of synchronized groups forming one temporal example."""

    group_indices: Tuple[int, ...]
    frame_ids: Tuple[int, ...]
    sequence_key: Tuple[int, int, int, Tuple[int, ...]]


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

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        window = self.windows[index]
        frame_samples = [self.base[i] for i in window.group_indices]
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
