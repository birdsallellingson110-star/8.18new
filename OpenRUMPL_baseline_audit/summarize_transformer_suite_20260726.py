#!/usr/bin/env python
"""Collect the decisive Transformer ablation metrics into one JSON table."""

import argparse
import json
from pathlib import Path


def load_summary(path):
    with path.open() as stream:
        return json.load(stream)


def find_metric(payload, candidates):
    for key in candidates:
        if key in payload:
            return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            found = find_metric(value, candidates)
            if found is not None:
                return found
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args()

    rows = {}
    for tag in args.tags:
        sources = {
            f"mmpose_v{n_views}": args.root
            / "multiview_model_best_eval"
            / f"{tag}_mmpose_v{n_views}_summary.json"
            for n_views in (2, 3, 4, 5)
        }
        row = {}
        for protocol, path in sources.items():
            if not path.exists():
                continue
            payload = load_summary(path)
            row[protocol] = {
                "all17": find_metric(
                    payload,
                    [
                        "all17_mm",
                        "mpjpe_all_mm",
                        "all_mpjpe_mm",
                        "mpjpe_all",
                        "all",
                    ],
                ),
                "kp_star": find_metric(
                    payload,
                    [
                        "kpstar_mm",
                        "mpjpe_kpstar_mm",
                        "kpstar_mpjpe_mm",
                        "mpjpe_kpstar",
                        "kp_star",
                    ],
                ),
                "summary": str(path),
            }
        rows[tag] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(rows, stream, indent=2, ensure_ascii=False)
    print(args.output)


if __name__ == "__main__":
    main()
