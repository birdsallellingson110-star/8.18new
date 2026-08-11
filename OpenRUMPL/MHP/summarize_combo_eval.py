#!/usr/bin/env python3
"""Summarize final KP* MPJPE values from CMU camera-combination logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MPJPE_RE = re.compile(r"3D MPJPE \(cm\):\s*([0-9.eE+-]+)")


def read_results(root: Path) -> dict[str, float]:
    results = {}
    for path in sorted(root.glob("views_*.log")):
        values = MPJPE_RE.findall(path.read_text(errors="replace"))
        if values:
            results[path.stem.removeprefix("views_")] = float(values[-1]) * 10.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rumpl", type=Path)
    parser.add_argument(
        "--rumpl-values",
        help="Comma-separated fallback values in mm, for example 3_6=40.37,3_12=46.95",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--name", default="candidate")
    args = parser.parse_args()

    candidate = read_results(args.candidate)
    rumpl = read_results(args.rumpl) if args.rumpl else {}
    if args.rumpl_values:
        rumpl.update(
            (key, float(value))
            for key, value in (item.split("=", 1) for item in args.rumpl_values.split(","))
        )
    baseline = read_results(args.baseline)
    keys = sorted(set(candidate) & set(rumpl) & set(baseline))
    if not keys:
        raise SystemExit("No common completed camera combinations found")

    print("combo\trumpl\tbaseline\t%s\td_rumpl\td_baseline" % args.name)
    delta_rumpl = []
    delta_baseline = []
    for key in keys:
        dr = candidate[key] - rumpl[key]
        db = candidate[key] - baseline[key]
        delta_rumpl.append(dr)
        delta_baseline.append(db)
        print(
            f"{key}\t{rumpl[key]:.3f}\t{baseline[key]:.3f}\t"
            f"{candidate[key]:.3f}\t{dr:+.3f}\t{db:+.3f}"
        )

    print(
        "SUMMARY vs_rumpl: "
        f"mean={sum(delta_rumpl) / len(keys):+.3f} "
        f"worst={max(delta_rumpl):+.3f} improved={sum(x < 0 for x in delta_rumpl)}/{len(keys)}"
    )
    print(
        "SUMMARY vs_baseline: "
        f"mean={sum(delta_baseline) / len(keys):+.3f} "
        f"worst={max(delta_baseline):+.3f} improved={sum(x < 0 for x in delta_baseline)}/{len(keys)}"
    )
    best_candidate_key = min(keys, key=lambda key: candidate[key])
    best_rumpl_key = min(keys, key=lambda key: rumpl[key])
    print(
        "BEST candidate: "
        f"combo={best_candidate_key} value={candidate[best_candidate_key]:.3f} "
        f"same_combo_rumpl={rumpl[best_candidate_key]:.3f} "
        f"delta={candidate[best_candidate_key] - rumpl[best_candidate_key]:+.3f}"
    )
    print(
        "BEST rumpl: "
        f"combo={best_rumpl_key} value={rumpl[best_rumpl_key]:.3f} "
        f"candidate_at_combo={candidate[best_rumpl_key]:.3f} "
        f"delta={candidate[best_rumpl_key] - rumpl[best_rumpl_key]:+.3f}"
    )


if __name__ == "__main__":
    main()
