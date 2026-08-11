#!/usr/bin/env python3
"""Scan disk for experiments that should not be re-queued (train/eval complete)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

AUDIT_ROOT = Path("/mnt/data/cjyoutput/open_source_fusion_audit_20260731")
PAPER_RUMPL = Path(
    "/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/multiview_h36m_rumpl/multiview_rumpl_999"
)

OUT_JSON = AUDIT_ROOT / "EXPERIMENT_SKIP_REGISTRY_20260805.json"
OUT_MD = AUDIT_ROOT / "EXPERIMENT_SKIP_REGISTRY_20260805.md"

TAG_RE = re.compile(r"^(H\d{2,3}_[A-Za-z0-9_.-]+_\d{8})(?:_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})?$")

# Queue variant -> completed experiment tag (same hypothesis, different launcher name)
LAUNCH_ALIASES: dict[str, str] = {
    "h138_h114_v4w": "H114_H76_viewWeights124_workers12_seed0_20260805",
    "h114_v4_train_weight": "H114_H76_viewWeights124_workers12_seed0_20260805",
    "h127_mono_h81": "H127_H81_perJointGate_mono005_w322_ftH81_workers8_seed0_20260805",
    "h128_relview_h81": "H128_H81_relViewFusion_w322_ftH81_workers8_seed0_20260805",
    "h123_gjv2": "H123_H76_globalJV2_rezero_w322_workers8_seed0_20260805",
}


@dataclass
class TagRecord:
    tag: str
    skip_train: bool = False
    reasons: list[str] = field(default_factory=list)
    done_files: list[str] = field(default_factory=list)
    model_best_paths: list[str] = field(default_factory=list)
    table2_paths: list[str] = field(default_factory=list)
    table2_v234_mm: dict[str, float] = field(default_factory=dict)


def normalize_tag(name: str) -> str | None:
    """Strip timestamp suffix from run directory names."""
    m = TAG_RE.match(name)
    if m:
        return m.group(1)
    if re.match(r"^H\d{2,3}_", name) and name.endswith("_20260805"):
        return name
    return None


def find_done_markers(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in root.rglob("*.done"):
        if "node_modules" in path.parts:
            continue
        stem = path.name[:-5] if path.name.endswith(".done") else path.name
        if stem.startswith("completed_"):
            stem = stem[len("completed_") :]
        tag = canonical_train_tag(stem) if stem.startswith("H") else None
        if not tag:
            tag = normalize_tag(stem) or (stem if stem.startswith("H") else None)
        if not tag:
            continue
        out.setdefault(tag, []).append(str(path))
    return out


def canonical_train_tag(name: str) -> str:
    """Merge run dirs that differ only by known naming typos (e.g. missing ftH81)."""
    t = normalize_tag(name) or name
    t = re.sub(
        r"_w322_workers8_seed",
        "_w322_ftH81_workers8_seed",
        t,
        count=1,
    )
    return t


def find_model_bests(*roots: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for ckpt in root.rglob("model_best.pth.tar"):
            parent = ckpt.parent.name
            tag = canonical_train_tag(parent)
            out.setdefault(tag, []).append(str(ckpt))
    return out


def find_table2_evals(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in root.rglob("table2.json"):
        if "/eval/" not in str(path) and "/eval_fast/" not in str(path):
            continue
        parts = path.parts
        tag_dir = None
        for i, p in enumerate(parts):
            if p in ("eval", "eval_fast") and i + 1 < len(parts):
                tag_dir = parts[i + 1]
                break
        if not tag_dir:
            continue
        tag = canonical_train_tag(tag_dir) if tag_dir.startswith("H") else tag_dir
        if not tag.startswith("H"):
            tag = normalize_tag(tag_dir) or tag_dir
        out.setdefault(tag, []).append(str(path))
    return out


def table2_summary(paths: list[str]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for p in sorted(paths):
        part = Path(p).parent.name  # V2, V3, V4
        if not re.match(r"V\d+$", part):
            continue
        try:
            data = json.loads(Path(p).read_text())
            mm = data.get("table2_action_equal", {}).get("all17_mm")
            if mm is not None:
                vals[part] = float(mm)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
    return vals


def build_registry() -> dict:
    done = find_done_markers(AUDIT_ROOT)
    bests = find_model_bests(AUDIT_ROOT, PAPER_RUMPL)
    table2 = find_table2_evals(AUDIT_ROOT)

    all_tags = sorted(set(done) | set(bests) | set(table2))
    records: dict[str, TagRecord] = {}

    for tag in all_tags:
        rec = TagRecord(tag=tag)
        rec.done_files = sorted(set(done.get(tag, [])))
        rec.model_best_paths = sorted(set(bests.get(tag, [])))
        rec.table2_paths = sorted(set(table2.get(tag, [])))
        rec.table2_v234_mm = table2_summary(rec.table2_paths)

        has_v234 = all(k in rec.table2_v234_mm for k in ("V2", "V3", "V4"))
        if rec.done_files:
            rec.skip_train = True
            rec.reasons.append("completed.done marker")
        if rec.model_best_paths and has_v234:
            rec.skip_train = True
            rec.reasons.append("model_best + Table-2 V2/V3/V4 eval")
        if len(rec.model_best_paths) >= 2:
            rec.skip_train = True
            rec.reasons.append(
                f"duplicate training runs ({len(rec.model_best_paths)} ckpts) — do not retrain"
            )

        records[tag] = rec

    skip_tags = sorted(t for t, r in records.items() if r.skip_train)

    alias_skip: dict[str, str] = {}
    for variant, target in LAUNCH_ALIASES.items():
        if target in skip_tags:
            alias_skip[variant] = target

    retrain_waste = sorted(
        t
        for t, r in records.items()
        if len(r.model_best_paths) >= 2 and not r.skip_train
    )

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "audit_root": str(AUDIT_ROOT),
        "skip_train_tags": skip_tags,
        "launch_alias_skip": alias_skip,
        "retrain_duplicate_tags": retrain_waste,
        "tags": {t: asdict(r) for t, r in records.items()},
    }


def write_md(data: dict) -> None:
    lines = [
        "# Experiment skip registry (auto-generated)",
        "",
        f"Generated: `{data['generated_at']}`",
        "",
        "Re-generate:",
        "",
        "```bash",
        "python3 /home/lixiaob/cjy/OpenRUMPL_baseline_audit/scan_experiment_skip_registry_20260805.py",
        "```",
        "",
        "Use in queues:",
        "",
        "```bash",
        "source /home/lixiaob/cjy/OpenRUMPL_baseline_audit/experiment_should_skip.sh",
        "experiment_should_skip_train H127_... && echo skip",
        "experiment_should_skip_variant h138_h114_v4w && echo skip",
        "```",
        "",
        f"## Skip train ({len(data['skip_train_tags'])})",
        "",
        "| Tag | Reasons | V2/V3/V4 mm |",
        "|-----|---------|-------------|",
    ]
    for tag in data["skip_train_tags"]:
        rec = data["tags"][tag]
        reasons = "; ".join(rec["reasons"]) or "—"
        v = rec.get("table2_v234_mm") or {}
        vstr = "/".join(f"{v.get(f'V{i}', '—'):.2f}" if f"V{i}" in v else "—" for i in (2, 3, 4))
        if vstr.replace("—", "").replace("/", "") == "":
            vstr = "—"
        lines.append(f"| `{tag}` | {reasons} | {vstr} |")

    if data["launch_alias_skip"]:
        lines.extend(["", "## Launch variant aliases (skip if target done)", ""])
        for var, tgt in sorted(data["launch_alias_skip"].items()):
            lines.append(f"- `{var}` → `{tgt}`")

    if data["retrain_duplicate_tags"]:
        lines.extend(["", "## Duplicate ckpts (do not retrain same TAG)", ""])
        for tag in data["retrain_duplicate_tags"]:
            n = len(data["tags"][tag]["model_best_paths"])
            lines.append(f"- `{tag}` ({n}× model_best)")

    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = build_registry()
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(data)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"skip_train_tags: {len(data['skip_train_tags'])}")
    print(f"alias_skip: {len(data['launch_alias_skip'])}")
    print(f"retrain_duplicate_tags: {len(data['retrain_duplicate_tags'])}")


if __name__ == "__main__":
    main()
