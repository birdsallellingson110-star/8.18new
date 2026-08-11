#!/usr/bin/env python3
"""Fast unit test for the real-H36M temporal-window adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO = Path("/home/lixiaob/cjy/OpenRUMPL/RUMPL")
sys.path.insert(0, str(REPO / "lib"))

from dataset.temporal_h36m_rumpl import TemporalH36MRUMPL, build_temporal_windows


class FakeFrameDataset:
    def __init__(self):
        self.max_random_n_views = 4
        self.db = []
        self.grouping = []
        # Two sequences.  The first contains a gap and must not bridge it.
        for sequence, frames in [((1, 2, 1), [1, 6, 11, 21]), ((1, 2, 2), [1, 6, 11])]:
            for frame in frames:
                group = []
                for camera in range(4):
                    group.append(len(self.db))
                    self.db.append(
                        {
                            "subject": sequence[0],
                            "action": sequence[1],
                            "subaction": sequence[2],
                            "image_id": frame,
                            "camera_id": camera,
                        }
                    )
                self.grouping.append(group)

    def __getitem__(self, index):
        value = float(index)
        return (
            torch.full((17, 1, 3), value),
            torch.full((17, 4, 4), value),
            torch.full((17, 3), value),
            torch.full((17, 4, 7), value),
            {"index": index},
            torch.full((17, 4, 20), value),
        )


def main():
    base = FakeFrameDataset()
    windows = build_temporal_windows(base, window_length=3, frame_stride=5)
    assert len(windows) == 2, windows
    assert windows[0].frame_ids == (1, 6, 11)
    assert windows[1].sequence_key == (1, 2, 2, (0, 1, 2, 3))
    wrapped = TemporalH36MRUMPL(base, window_length=3, frame_stride=5)
    sample = wrapped[0]
    assert sample[3].shape == (3, 17, 4, 7)
    assert sample[2].shape == (3, 17, 3)
    assert base.max_random_n_views is None
    print("TemporalH36MRUMPL PASS windows=2 rays=(3,17,4,7)")


if __name__ == "__main__":
    main()
