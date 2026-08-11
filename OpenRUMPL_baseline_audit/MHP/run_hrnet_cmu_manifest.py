#!/usr/bin/env python3
"""Run the RUMPL HRNet model on an explicit CMU image manifest."""

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np


def atomic_json_dump(value, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    with temporary.open("w") as stream:
        json.dump(value, stream)
    os.replace(temporary, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--pose2d", default="td-hm_hrnet-w32_8xb64-210e_coco-384x288"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import mmpose
    import torch
    from mmpose.apis import MMPoseInferencer

    relative_images = [
        line.strip() for line in args.manifest.read_text().splitlines() if line.strip()
    ]
    if len(relative_images) != len(set(relative_images)):
        raise ValueError("Manifest contains duplicate image paths")

    missing_images = [path for path in relative_images if not (args.root / path).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"Manifest images are missing: count={len(missing_images)}, "
            f"first={missing_images[:5]}"
        )

    provenance = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mmpose": mmpose.__version__,
        "pose2d": args.pose2d,
        "device": args.device,
        "manifest": str(args.manifest.resolve()),
        "manifest_count": len(relative_images),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(provenance, args.output / "_provenance.json")
    print(json.dumps(provenance, indent=2), flush=True)

    inferencer = MMPoseInferencer(pose2d=args.pose2d, device=args.device)
    processed = 0
    skipped = 0
    for index, relative_image in enumerate(relative_images, start=1):
        image = args.root / relative_image
        parts = Path(relative_image).parts
        destination = (
            args.output / parts[0] / parts[-2] / Path(parts[-1]).with_suffix(".json")
        )
        if destination.is_file() and destination.stat().st_size > 2:
            skipped += 1
            continue

        result = next(inferencer(str(image), return_vis=False))
        persons = result["predictions"][0]
        output = []
        for person in persons:
            output.append(
                {
                    "keypoints": np.asarray(person["keypoints"]).tolist(),
                    "keypoint_scores": np.asarray(person["keypoint_scores"]).tolist(),
                    "bbox": np.asarray(person.get("bbox", [])).reshape(-1).tolist(),
                    "bbox_score": float(person.get("bbox_score", 1.0)),
                }
            )
        atomic_json_dump(output, destination)
        processed += 1
        if index % 100 == 0:
            print(
                f"progress={index}/{len(relative_images)} "
                f"processed={processed} skipped={skipped}",
                flush=True,
            )

    expected_outputs = {
        str(Path(path).with_suffix(".json")).replace("/hdImgs/", "/")
        for path in relative_images
    }
    present_outputs = {
        path.relative_to(args.output).as_posix()
        for path in args.output.glob("*/*/*.json")
    }
    missing_outputs = sorted(expected_outputs - present_outputs)
    extra_outputs = sorted(present_outputs - expected_outputs)
    print(
        f"expected={len(expected_outputs)} present={len(present_outputs)} "
        f"missing={len(missing_outputs)} extra={len(extra_outputs)}",
        flush=True,
    )
    if missing_outputs or extra_outputs:
        raise RuntimeError(
            f"Output validation failed; missing={missing_outputs[:5]}, "
            f"extra={extra_outputs[:5]}"
        )


if __name__ == "__main__":
    main()
