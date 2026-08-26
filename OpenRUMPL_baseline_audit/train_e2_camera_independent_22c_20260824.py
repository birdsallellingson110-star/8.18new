#!/usr/bin/env python3
"""Train the camera-independent E2 variant on the audited 22-candidate pool.

The historical experiment source was later extended to a 44-candidate pool.
This small entry point pins the current paper baseline layout to the original
11 frozen-generator plus 11 confidence-triangulation candidates without
duplicating the training implementation.
"""

from __future__ import annotations

import train_e2_v234_universal_20260812 as trainer


trainer.ALL_CANDIDATE_COMBINATIONS = (
    trainer.ORIGINAL_COMBINATIONS + trainer.ORIGINAL_COMBINATIONS
)


if __name__ == "__main__":
    trainer.main()
