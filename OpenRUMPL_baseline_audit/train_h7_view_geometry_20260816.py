#!/usr/bin/env python3
"""Current-protocol H7 candidate verifier.

This adapter keeps the audited 22-candidate E2 trainer and changes only its
candidate scorer to the official-source-compatible ray/view cross-attention
variant.  The input remains coordinates, confidence and camera rays; no
heatmap or image feature is introduced.  ``H7_JOINT_ATTENTION`` is optional
and defaults to ``none`` so the first experiment isolates view-wise geometric
verification.
"""

from __future__ import annotations

import os

import train_current_e2_confidence_20260815 as current
from train_h76_set_transformer_utility_20260811 import (
    SetTransformerJointUtility as _BaseUtility,
)


class ViewGeometryUtility(_BaseUtility):
    """E2 scorer with per-candidate ray/view verification tokens."""

    def __init__(self, mean, std, depth, stage_heads=False):
        joint_attention = os.environ.get("H7_JOINT_ATTENTION", "none").strip()
        if joint_attention not in {"none", "post", "alternating"}:
            raise ValueError(
                "H7_JOINT_ATTENTION must be none, post or alternating"
            )
        super().__init__(
            mean,
            std,
            depth,
            view_cross_attention=True,
            joint_attention=joint_attention,
            stage_heads=stage_heads,
        )


# The universal trainer resolves this symbol at runtime from its module
# namespace.  Replacing it here preserves all split, holdout, loss and output
# conventions of the audited current-protocol trainer.
current.trainer.SetTransformerJointUtility = ViewGeometryUtility


if __name__ == "__main__":
    current.trainer.main()
