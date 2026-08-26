#!/usr/bin/env python3
"""Retrain E2 with subset-penalty features removed so oracle winners can rank.

Geometric reprojection selection failed (V2 ~90 mm): 2D residual is not 3D
error.  The learned scorer still cannot pick leave-one-out / snap / bone
because view_fraction and excluded-view residual mark them as bad.  This run
zeros those channels, keeps listwise oracle CE, cardinality-balanced loss,
and the task-local extra candidates.  Frozen H76, clean H36M only.
"""
from __future__ import annotations

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_oracle_listwise_20260819 as listwise
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from train_h76_set_transformer_utility_20260811 import SetTransformerJointUtility


def UnbiasedSetTransformer(*args, **kwargs):
    kwargs["neutralize_subset_penalty"] = True
    return SetTransformerJointUtility(*args, **kwargs)


def main() -> None:
    import json
    import sys
    from pathlib import Path

    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(
        next(sys.argv[i + 1] for i, token in enumerate(sys.argv) if token == "--train-shards")
    )
    trainer.predict_task = extra.predict_task
    trainer.task_loss = listwise.task_loss
    trainer.SetTransformerJointUtility = UnbiasedSetTransformer
    trainer.main()
    output_dir = None
    for index, token in enumerate(sys.argv):
        if token == "--output-dir":
            output_dir = Path(sys.argv[index + 1])
            break
    if output_dir is not None:
        payload_path = output_dir / "result.json"
        if payload_path.exists():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["method"] = (
                "E2-C2 unbiased scorer: no view-fraction/excluded-residual "
                "penalty, listwise oracle CE, cardinality-balanced loss, "
                "view-snap/bone extras"
            )
            payload["neutralize_subset_penalty"] = True
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
