#!/usr/bin/env python3
"""Collect protocol-matched H0/H32-H38 Table-II results into one scoreboard."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path('/mnt/data/cjyoutput/open_source_fusion_audit_20260731')
OUTPUT_JSON = ROOT / 'H32_H38_ACCURACY_SCOREBOARD.json'
OUTPUT_MD = ROOT / 'H32_H38_ACCURACY_SCOREBOARD.md'
BASELINE_TAG = (
    'H0_H22_CUR_a1dRefined2D_originalRUMPL_triAnchor_'
    'fixedK2First8_thenWeighted3to1to1_seed0_20260731'
)
TARGET = {2: 40.0, 4: 30.0}


def read_metric(path: Path) -> float:
    with path.open() as handle:
        return float(json.load(handle)['table2_action_equal']['all17_mm'])


def main() -> None:
    results: dict[str, dict[int, dict[str, object]]] = {}
    for path in ROOT.rglob('table2.json'):
        if not re.fullmatch(r'V[234]', path.parent.name):
            continue
        tag = path.parent.parent.name
        if tag != BASELINE_TAG and not re.match(r'H(?:3[2-8])_', tag):
            continue
        views = int(path.parent.name[1:])
        results.setdefault(tag, {})[views] = {
            'all17_mm': read_metric(path),
            'path': str(path),
        }

    if BASELINE_TAG not in results:
        raise FileNotFoundError(f'Baseline result not found: {BASELINE_TAG}')
    baseline = {
        views: results[BASELINE_TAG][views]['all17_mm']
        for views in (2, 3, 4)
    }
    rows = []
    for tag, metrics in sorted(results.items()):
        row = {'tag': tag, 'metrics': metrics}
        row['improvement_vs_h0_mm'] = {
            views: baseline[views] - metrics[views]['all17_mm']
            for views in metrics
        }
        row['meets_target'] = {
            views: metrics.get(views, {}).get('all17_mm', float('inf')) <= limit
            for views, limit in TARGET.items()
        }
        rows.append(row)

    payload = {
        'metric': 'Human3.6M Table-II action-equal All-17 MPJPE (mm)',
        'baseline_tag': BASELINE_TAG,
        'target_mm': TARGET,
        'rows': rows,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + '\n')

    lines = [
        '# H36M RUMPL accuracy scoreboard',
        '',
        'All values are protocol-matched Table-II action-equal All-17 MPJPE (mm).',
        '',
        '| experiment | V2 | V3 | V4 | ΔV2 vs H0 | ΔV4 vs H0 | target |',
        '|---|---:|---:|---:|---:|---:|:---:|',
    ]
    for row in rows:
        metrics = row['metrics']
        improvement = row['improvement_vs_h0_mm']
        value = lambda views: (
            f"{metrics[views]['all17_mm']:.3f}" if views in metrics else '—'
        )
        delta = lambda views: (
            f"{improvement[views]:+.3f}" if views in improvement else '—'
        )
        meets = row['meets_target']
        target_text = 'yes' if meets[2] and meets[4] else 'no'
        lines.append(
            f"| {row['tag']} | {value(2)} | {value(3)} | {value(4)} | "
            f"{delta(2)} | {delta(4)} | {target_text} |"
        )
    OUTPUT_MD.write_text('\n'.join(lines) + '\n')
    print(OUTPUT_MD)


if __name__ == '__main__':
    main()
