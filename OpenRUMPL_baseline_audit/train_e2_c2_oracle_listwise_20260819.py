#!/usr/bin/env python3
"""Retrain E2 view-snap/bone with listwise oracle CE and cardinality-balanced loss.

Why the previous run did not use the better candidates:
- balanced_rank only asks "better than H76?", and pairwise terms are dominated
  by easy pairs, so V3/V4 oracle-hit stays ~7-23%;
- the GHT oracle-softmax CE was explicitly zeroed (`+ 0.0 * target_temperature`);
- the 11 camera tasks are averaged equally, so V4 is 1/11 of the loss.

This run keeps the same frozen H76 cache and task-local extra candidates, but
trains the scorer to match the per-joint oracle distribution and weights
V2/V3/V4 equally.  No occlusion training.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import train_current_e2_confidence_20260815 as wrapper
import train_e2_c2_viewsnap_bone_20260819 as extra
import train_e2_v234_universal_20260812 as trainer
from train_h76_counterfactual_delta_20260811 import training_loss


def _balanced(terms):
    buckets = {2: [], 3: [], 4: []}
    for term, combo in zip(terms, trainer.TASK_COMBINATIONS):
        buckets[len(combo)].append(term)
    return torch.stack(
        [torch.stack(buckets[count]).mean() for count in (2, 3, 4)]
    ).mean()


def task_loss(
    model, predictions, targets, rays, phase, temperature, target_temperature,
    oracle_weight, identity_hinge=0.0, identity_v2_weight=1.0,
):
    direct_losses = []
    oracle_losses = []
    ght_losses = []
    identity_losses = []
    for task_combo in trainer.TASK_COMBINATIONS:
        predicted, true_delta, true_error, candidates, baseline_local = trainer.predict_task(
            model, predictions, targets, rays, task_combo
        )
        direct_losses.append(training_loss(predicted, true_delta, "balanced_rank"))
        target = F.softmax(-true_error / max(float(target_temperature), 1e-6), dim=-1)
        log_prob = F.log_softmax(-predicted / temperature, dim=-1)
        oracle_losses.append(-(target * log_prob).sum(dim=-1).mean())
        if phase == "ght" or identity_hinge > 0:
            weights = F.softmax(-predicted / temperature, dim=-1)
            expected = (weights * true_error).sum(dim=-1).mean()
            fused = torch.einsum("bjc,bcjd->bjd", weights, candidates)
            fused_error = torch.linalg.vector_norm(fused - targets, dim=-1).mean()
            if identity_hinge > 0:
                baseline_error = true_error[..., baseline_local]
                violation = F.relu(fused_error - baseline_error).mean()
                stage_weight = identity_v2_weight if len(task_combo) == 2 else 1.0
                identity_losses.append(
                    identity_hinge * stage_weight * violation / 0.01
                )
            if phase == "ght":
                ght_losses.append((expected + 0.05 * fused_error) / 0.01)
    loss = _balanced(direct_losses) + float(oracle_weight) * _balanced(oracle_losses)
    if ght_losses:
        loss = loss + _balanced(ght_losses)
    if identity_losses:
        loss = loss + _balanced(identity_losses)
    return loss


def main() -> None:
    import json
    import sys
    from pathlib import Path

    wrapper.trainer.ALL_CANDIDATE_COMBINATIONS = wrapper.ORIGINAL + wrapper.ORIGINAL
    extra.BONE_LENGTHS = extra.train_bone_lengths(
        next(sys.argv[i + 1] for i, token in enumerate(sys.argv) if token == "--train-shards")
    )
    trainer.predict_task = extra.predict_task
    trainer.task_loss = task_loss
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
                "E2-C2 view-snap/bone with listwise oracle CE and "
                "cardinality-balanced V2/V3/V4 loss"
            )
            payload["scoring_fix"] = [
                "listwise CE to per-joint oracle softmax (5 mm), both phases",
                "mean(V2, V3, V4) instead of mean over 11 camera tasks",
                "GHT expected-risk kept in the finetune phase, not zeroing the CE",
            ]
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
