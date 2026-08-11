#!/usr/bin/env python3
"""Extract exactly the CMU frames listed by the public RUMPL repository."""

import argparse
import os
from collections import defaultdict
from pathlib import Path, PurePosixPath

import cv2


def parse_manifest(path: Path):
    required = defaultdict(set)
    manifest_paths = set()
    for raw_line in path.read_text().splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        item = PurePosixPath(raw_line)
        if len(item.parts) != 4 or item.parts[1] != "hdImgs":
            raise ValueError(f"Unexpected manifest path: {raw_line}")
        seq, camera_dir, filename = item.parts[0], item.parts[2], item.parts[3]
        frame_id = int(PurePosixPath(filename).stem.rsplit("_", 1)[1])
        required[(seq, camera_dir)].add(frame_id)
        manifest_paths.add(raw_line)
    return required, manifest_paths


def ensure_metadata_links(source_root: Path, output_root: Path, sequences):
    for seq in sequences:
        source_seq = source_root / seq
        output_seq = output_root / seq
        output_seq.mkdir(parents=True, exist_ok=True)
        for name in ("hdVideos", "hdPose3d_stage1_coco19"):
            target = source_seq / name
            link = output_seq / name
            if not link.exists():
                link.symlink_to(target)
        calibration = f"calibration_{seq}.json"
        link = output_seq / calibration
        if not link.exists():
            link.symlink_to(source_seq / calibration)


def extract_video(video: Path, output_dir: Path, camera_dir: str, frame_ids):
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = {
        frame_id
        for frame_id in frame_ids
        if not (output_dir / f"{camera_dir}_{frame_id:08d}.jpg").is_file()
    }
    if not missing:
        print(f"[{video}] already complete ({len(frame_ids)} frames)", flush=True)
        return

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    max_frame = max(missing)
    written = 0
    frame_index = 0
    while frame_index <= max_frame:
        ok = capture.grab()
        if not ok:
            break
        if frame_index in missing:
            ok, frame = capture.retrieve()
            if not ok:
                raise RuntimeError(f"Cannot decode {video} frame {frame_index}")
            if frame.shape[:2] != (1080, 1920):
                raise RuntimeError(
                    f"Unexpected frame shape {frame.shape} in {video} frame {frame_index}"
                )
            destination = output_dir / f"{camera_dir}_{frame_index:08d}.jpg"
            temporary = destination.with_suffix(".tmp.jpg")
            if not cv2.imwrite(str(temporary), frame, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                raise RuntimeError(f"Cannot write image: {temporary}")
            os.replace(temporary, destination)
            written += 1
        frame_index += 1
    capture.release()

    still_missing = [
        frame_id
        for frame_id in frame_ids
        if not (output_dir / f"{camera_dir}_{frame_id:08d}.jpg").is_file()
    ]
    if still_missing:
        raise RuntimeError(
            f"Video ended before all frames were extracted from {video}; "
            f"missing {len(still_missing)}, first={still_missing[:5]}"
        )
    print(
        f"[{video}] requested={len(frame_ids)} newly_written={written} "
        f"decoded_through={frame_index - 1}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    required, manifest_paths = parse_manifest(args.manifest)
    sequences = sorted({seq for seq, _ in required})
    ensure_metadata_links(args.source_root, args.output_root, sequences)

    for seq in sequences:
        camera_sets = {
            camera: required[(seq, camera)]
            for candidate_seq, camera in required
            if candidate_seq == seq
        }
        reference = next(iter(camera_sets.values()))
        for camera, frames in camera_sets.items():
            if frames != reference:
                raise ValueError(f"Camera frame sets differ in {seq}: {camera}")

    for (seq, camera_dir), frame_ids in sorted(required.items()):
        camera_id = camera_dir.split("_", 1)[1]
        video = args.source_root / seq / "hdVideos" / f"hd_00_{camera_id}.mp4"
        output_dir = args.output_root / seq / "hdImgs" / camera_dir
        extract_video(video, output_dir, camera_dir, sorted(frame_ids))

    present = {
        path.relative_to(args.output_root).as_posix()
        for path in args.output_root.glob("*/hdImgs/*/*.jpg")
    }
    missing = sorted(manifest_paths - present)
    extra = sorted(present - manifest_paths)
    print(
        f"manifest={len(manifest_paths)} present={len(present)} "
        f"missing={len(missing)} extra={len(extra)}",
        flush=True,
    )
    if missing or extra:
        raise RuntimeError(
            f"Output validation failed; missing={missing[:5]}, extra={extra[:5]}"
        )


if __name__ == "__main__":
    main()
