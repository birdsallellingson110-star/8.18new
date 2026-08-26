#!/usr/bin/env python3
"""Overlay the frozen sparse benchmark images onto matching dense center keys.

Dense neighboring frames provide T=9 context.  The 2021 scored center groups
remain byte-identical to the sparse Algebraic-control-aligned benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-pkl", required=True, type=Path)
    parser.add_argument("--sparse-root", required=True, type=Path)
    parser.add_argument("--dense-root", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    sparse_images = (args.sparse_root / "images").resolve()
    dense_images = (args.dense_root / "images").resolve()
    dense_manifest_path = args.dense_root / "protocol_manifest.json"
    sparse_manifest_path = args.sparse_root / "protocol_manifest.json"
    for required in (
        args.sparse_pkl,
        sparse_images,
        dense_images,
        dense_manifest_path,
        sparse_manifest_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    with args.sparse_pkl.open("rb") as handle:
        records = pickle.load(handle)
    if len(records) != 8084:
        raise RuntimeError(f"expected 8084 sparse records, got {len(records)}")
    keys = {
        (
            int(record["subject"]),
            int(record["action"]),
            int(record["subaction"]),
            int(record["image_id"]),
        )
        for record in records
    }
    if len(keys) != 2021:
        raise RuntimeError(f"expected 2021 sparse groups, got {len(keys)}")

    copied = linked = 0
    for record in records:
        relative = Path(str(record["image"]))
        source = sparse_images / relative
        destination = dense_images / relative
        if not source.exists():
            raise FileNotFoundError(source)
        if dense_images not in destination.resolve().parents:
            # destination.resolve() follows an existing symlink; validate the
            # lexical path separately before replacing it.
            lexical = Path(os.path.abspath(destination))
            if dense_images not in lexical.parents:
                raise RuntimeError(f"destination escapes dense root: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        if source.is_symlink():
            destination.symlink_to(source.resolve())
            linked += 1
        else:
            shutil.copy2(source, destination)
            copied += 1

    dense_manifest = json.loads(dense_manifest_path.read_text())
    dense_manifest["temporal_extension_scoring"] = {
        "scored_center_groups": 2021,
        "scored_center_records": 8084,
        "center_source_root": str(args.sparse_root.resolve()),
        "center_source_manifest_sha256": sha256(sparse_manifest_path),
        "center_payload": "byte-identical copy/symlink of frozen sparse benchmark",
        "neighboring_context": "dense public-generator-derived frames",
        "copied_occluded_center_jpegs": copied,
        "linked_clean_center_images": linked,
    }
    temporary = dense_manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dense_manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, dense_manifest_path)
    (args.dense_root / "sparse_centers_OVERLAID").write_text("complete\n")
    print(json.dumps(dense_manifest["temporal_extension_scoring"], indent=2))


if __name__ == "__main__":
    main()
