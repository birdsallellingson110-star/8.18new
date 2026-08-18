#!/usr/bin/env python3
"""Run the audited V2/V3/V4 E2 trainer on the current 22-candidate cache.

The upstream universal trainer was written for a 44-candidate historical cache
(RIGR, pairwise, learned, confidence and IRLS). The current fair GBT-style
line has only frozen H76 candidates plus confidence-weighted candidates. This
thin adapter changes only the candidate manifest; the model, losses, holdout
selection and evaluation remain the audited upstream implementation.
"""

from __future__ import annotations

import itertools

import train_e2_v234_universal_20260812 as trainer


ORIGINAL = tuple(
    combo
    for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)

# First 11 entries are frozen H76; second 11 are confidence-weighted solutions
# for the identical subsets. Duplicate masks are intentional: the scorer must
# learn whether the alternative hypothesis is useful for each task/joint.
trainer.ALL_CANDIDATE_COMBINATIONS = ORIGINAL + ORIGINAL


if __name__ == "__main__":
    trainer.main()
