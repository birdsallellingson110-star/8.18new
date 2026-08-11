#!/usr/bin/env python3
"""Print sibling ablation table vs R5 / S2a / A2 / B1."""
import json
import os

root = "/mnt/data/cjyoutput/baseline_reaudit_20260722/occlusion_eval"


def load(tag, v, occ):
    p = f"{root}/{tag}_v{v}_occ{occ}_summary.json"
    if not os.path.isfile(p):
        return None
    return json.load(open(p))["overall"]["all17_mm"]


variants = [
    ("R5", "R5"),
    ("S2a", "S2a"),
    ("A2", "A2"),
    ("B1", "B1"),
    ("B1only", "B1only"),
    ("B1occ02", "B1occ02"),
    ("B1A2", "B1A2"),
    ("C", "C"),
]

print("############ Sibling ablations vs R5 (All-17 mm) ############")
hdr = f"{'':12}" + "".join(f"{n:>9}" for _, n in variants) + f"{'Δbest-R5':>10}"
print(hdr)
for v in (2, 3, 4, 5):
    for occ in ("0.0", "0.3", "0.6"):
        vals = []
        for tag, _ in variants:
            vals.append(load(tag, v, occ))
        if vals[0] is None:
            print(f"V{v} occ{occ}: R5 missing")
            continue
        cells = []
        for x in vals:
            cells.append(f"{x:9.2f}" if x is not None else f"{'MISS':>9}")
        known = [x for x in vals[1:] if x is not None]
        best = min(known) if known else None
        d = (best - vals[0]) if best is not None else float("nan")
        print(f"V{v} occ{occ:3}: {''.join(cells)} {d:+10.2f}")

print("\n--- Δ vs R5 (negative = better) ---")
for tag, name in variants[1:]:
    print(f"\n[{name}]")
    for v in (2, 3, 4, 5):
        parts = []
        for occ in ("0.0", "0.3", "0.6"):
            r, x = load("R5", v, occ), load(tag, v, occ)
            if r is None or x is None:
                parts.append(f"occ{occ}:MISS")
            else:
                parts.append(f"occ{occ}:{x-r:+.2f}")
        print(f"  V{v}  " + "  ".join(parts))
print("=== summarize done ===")
