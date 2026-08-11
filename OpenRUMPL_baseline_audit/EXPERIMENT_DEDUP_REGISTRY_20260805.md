# Experiment dedup registry (2026-08-05)

## GBT beat targets (H36M Table-I style, All-17 action-equal mm)

| View | GBT-HRNet (T=9 ref) | H81 single-frame |
|------|---------------------|------------------|
| V2 | 36.8 | **34.61** |
| V3 | 30.4 | 30.81 |
| V4 | 26.0 | 29.99 |

## Invalid / do not repeat

- **H112 on H76**: GBT VFT bias finetune **regressed** (~39.7/38.1/35.1) — do not queue again on H76.
- Scratch H124/H127 without `ftH76`/`ftH81`.
- H129, duplicate H114 always-3/4-view-only training.

## Phase-2 queue (`queue_beat_gbt_phase2_20260805.sh`, GPU0)

| ID | Variable | Stack |
|----|----------|-------|
| H135 | T=9 temporal, frozen H81, GBT-biased JVT, token-dropout 0.2 | H81 |
| H136 | T=9 temporal, unfreeze VFT on H81 | H81 |
| H137 | DePro λ=0.1 + CAA λ=0.1 | ft H81 |
| H138 | View-count weights 1:2:4 (H114) | H76 scratch — pending |
| H139 | H117 temporal H76 JVT | H76 |
| H140 | GBT conf/geom bias finetune | ft H81 only |

## In flight (other queues)

- GPU1: `H127_H81_perJointGate_mono005_w322_ftH81` (stack queue)

## Auto skip list (train + eval complete, or duplicate ckpt)

Regenerate after new runs:

```bash
python3 OpenRUMPL_baseline_audit/scan_experiment_skip_registry_20260805.py
```

See `EXPERIMENT_SKIP_REGISTRY_20260805.md` on the output volume. Queues source `experiment_should_skip.sh`.

## Architecture sprint (H141-H152, 2026-08-05)

Fixed input (HRNet→A1D→H21). Module-only ablations; finetune H81 (or H76 for set-decoder).

```bash
bash OpenRUMPL_baseline_audit/queue_H141_H152_arch_sprint_parallel_20260805.sh
```

Eval fix: `launch_H59` passes `--n-views-combinations` for valid V2/V3/V4 Table-2.
