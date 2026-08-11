#!/usr/bin/env python3
"""Download the licensed Human3.6M S9/S11 files needed by RUMPL.

The script downloads only from the official Human3.6M file browser. It does
not persist the PHP session cookie. Existing partial files are resumed and
every completed archive is checked against H36M-Toolbox's published MD5.
"""

import argparse
import getpass
import hashlib
import os
from pathlib import Path
import sys
import time

import requests


BASE_URL = "http://vision.imar.ro/human3.6m/filebrowser.php"
SUBJECT_IDS = {"S9": 4, "S11": 5}
ARCHIVES = (
    ("Poses_D2_Positions_{subject}.tgz", "Poses/D2_Positions"),
    ("Poses_D3_Positions_mono_{subject}.tgz", "Poses/D3_Positions_mono"),
    (
        "Poses_D3_Positions_mono_universal_{subject}.tgz",
        "Poses/D3_Positions_mono_universal",
    ),
    ("Videos_{subject}.tgz", "Videos"),
)


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checksums(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        checksums[Path(filename).name] = digest
    return checksums


def verify_session(session: requests.Session) -> None:
    response = session.get(BASE_URL, timeout=60, allow_redirects=True)
    response.raise_for_status()
    if response.url.rstrip("/") != BASE_URL.rstrip("/"):
        raise RuntimeError(
            "Human3.6M session was rejected; log in again and refresh PHPSESSID"
        )


def download(
    session: requests.Session,
    subject: str,
    subject_id: int,
    filename: str,
    category: str,
    destination: Path,
    expected_md5: str,
) -> None:
    final_path = destination / filename
    partial_path = final_path.with_suffix(final_path.suffix + ".part")

    if final_path.exists() and md5sum(final_path) == expected_md5:
        print(f"[ok] {filename} already verified")
        return

    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    params = {
        "download": "1",
        "filepath": category,
        "filename": f"SubjectSpecific_{subject_id}.tgz",
    }
    response = session.get(
        BASE_URL,
        params=params,
        headers=headers,
        stream=True,
        timeout=(60, 300),
        allow_redirects=True,
    )
    response.raise_for_status()
    if response.url.split("?", 1)[0] != BASE_URL:
        raise RuntimeError(f"download redirected to login page for {subject}")

    resumed = offset > 0 and response.status_code == 206
    mode = "ab" if resumed else "wb"
    if offset and not resumed:
        print(f"[restart] server ignored Range for {filename}")
        offset = 0

    total_header = int(response.headers.get("Content-Length", 0))
    total = offset + total_header if total_header else 0
    downloaded = offset
    last_report = time.monotonic()
    with partial_path.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            downloaded += len(chunk)
            now = time.monotonic()
            if now - last_report >= 10:
                suffix = f"/{total / 2**30:.2f} GiB" if total else ""
                print(
                    f"[download] {filename}: {downloaded / 2**30:.2f}{suffix}",
                    flush=True,
                )
                last_report = now

    actual_md5 = md5sum(partial_path)
    if actual_md5 != expected_md5:
        raise RuntimeError(
            f"MD5 mismatch for {filename}: {actual_md5} != {expected_md5}"
        )
    partial_path.replace(final_path)
    print(f"[done] {filename}: {final_path.stat().st_size / 2**30:.2f} GiB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checksums", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", choices=SUBJECT_IDS, default=["S9", "S11"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cookie = os.environ.get("H36M_PHPSESSID") or getpass.getpass(
        "Human3.6M PHPSESSID (hidden): "
    )
    if not cookie:
        print("PHPSESSID is required", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    checksums = load_checksums(args.checksums)
    session = requests.Session()
    session.cookies.set("PHPSESSID", cookie)
    verify_session(session)

    for subject in args.subjects:
        subject_id = SUBJECT_IDS[subject]
        for filename_template, category in ARCHIVES:
            filename = filename_template.format(subject=subject)
            download(
                session,
                subject,
                subject_id,
                filename,
                category,
                args.output,
                checksums[filename],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
