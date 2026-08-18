#!/usr/bin/env python3
"""Run the audited E2 scorer on H76 + confidence + IRLS candidates.

The candidate order is deliberately explicit: the first 11 candidates are the
frozen H76 outputs, the next 11 are confidence-weighted ray intersections,
and the last 11 are robust IRLS intersections.  The scorer and losses are
otherwise inherited unchanged from the universal E2 implementation.
"""

from __future__ import annotations

import itertools

import train_e2_v234_universal_20260812 as trainer


ORIGINAL = tuple(
    combo
    for count in (2, 3, 4)
    for combo in itertools.combinations(range(4), count)
)

# One frozen model output plus two deterministic geometric alternatives per
# identical view subset.  Duplicate masks are intentional and let the utility
# scorer learn which solver is reliable for each joint/task.
trainer.ALL_CANDIDATE_COMBINATIONS = ORIGINAL + ORIGINAL + ORIGINAL


if __name__ == "__main__":
    trainer.main()
