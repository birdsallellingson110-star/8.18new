#!/usr/bin/env python3
import json
import os
import re
import time
from pathlib import Path

import requests


COOKIE_FILE = Path("/home/lixiaob/cjy/.secrets/baidu_h36m_cookies.json")
SHARE_URL = "https://pan.baidu.com/s/1cMUfHURFudfVziZ9hy6M3g"
SURL = "cMUfHURFudfVziZ9hy6M3g"
PASSWORD = "kcng"
TARGET = "/cjy_h36m_download_20260727"


def page_field(page: str, name: str) -> str:
    for pattern in (
        rf'"{name}"\s*:\s*"([^"]*)"',
        rf'"{name}"\s*:\s*([0-9]+)',
    ):
        match = re.search(pattern, page)
        if match:
            return match.group(1)
    return ""


with COOKIE_FILE.open() as handle:
    saved_cookies = json.load(handle)

session = requests.Session()
for item in saved_cookies:
    name = item.get("name")
    if name:
        session.cookies.set(
            name,
            item["value"],
            domain=item.get("domain") or ".baidu.com",
            path=item.get("path") or "/",
        )
session.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
)

page = session.get(SHARE_URL, timeout=30)
page.raise_for_status()
share_id = page_field(page.text, "shareid")
share_uk = page_field(page.text, "share_uk")
bdstoken = page_field(page.text, "bdstoken")
if not (share_id and share_uk and bdstoken):
    raise RuntimeError("分享页缺少 shareid/share_uk/bdstoken")

verify = session.post(
    "https://pan.baidu.com/share/verify",
    params={
        "surl": SURL,
        "t": int(time.time() * 1000),
        "channel": "chunlei",
        "web": 1,
        "app_id": 250528,
        "bdstoken": bdstoken,
        "clienttype": 0,
    },
    data={"pwd": PASSWORD, "vcode": "", "vcode_str": ""},
    headers={"Referer": SHARE_URL},
    timeout=30,
)
verify.raise_for_status()
if verify.json().get("errno") != 0:
    raise RuntimeError(f"提取码验证失败: errno={verify.json().get('errno')}")

listing = session.get(
    "https://pan.baidu.com/share/list",
    params={
        "shareid": share_id,
        "uk": share_uk,
        "shorturl": SURL,
        "root": 1,
        "web": 1,
        "app_id": 250528,
        "channel": "chunlei",
        "clienttype": 0,
        "bdstoken": bdstoken,
    },
    headers={"Referer": SHARE_URL},
    timeout=30,
)
listing.raise_for_status()
items = listing.json().get("list", [])
if not items:
    raise RuntimeError(f"分享目录为空: errno={listing.json().get('errno')}")

# Saving the single shared root directory is sometimes incorrectly rejected as
# a name conflict. Transfer its archive children directly instead.
if len(items) == 1 and items[0].get("isdir"):
    child_listing = session.get(
        "https://pan.baidu.com/share/list",
        params={
            "shareid": share_id,
            "uk": share_uk,
            "shorturl": SURL,
            "dir": items[0]["path"],
            "page": 1,
            "num": 100,
            "web": 1,
            "app_id": 250528,
            "channel": "chunlei",
            "clienttype": 0,
            "bdstoken": bdstoken,
        },
        headers={"Referer": SHARE_URL},
        timeout=30,
    )
    child_listing.raise_for_status()
    items = child_listing.json().get("list", [])
    if not items:
        raise RuntimeError(
            f"分享子目录为空: errno={child_listing.json().get('errno')}"
        )

fsid_list = [item["fs_id"] for item in items]
transfer = session.post(
    "https://pan.baidu.com/share/transfer",
    params={
        "shareid": share_id,
        "from": share_uk,
        "async": 1,
        "ondup": "newcopy",
        "bdstoken": bdstoken,
        "channel": "chunlei",
        "web": 1,
        "app_id": 250528,
        "clienttype": 0,
    },
    data={"fsidlist": json.dumps(fsid_list), "path": TARGET},
    headers={"Referer": SHARE_URL},
    timeout=60,
)
transfer.raise_for_status()
result = transfer.json()
errno = result.get("errno")
if errno not in (0, 4):
    raise RuntimeError(f"转存失败: errno={errno}, info={result.get('info')}")

print(f"TRANSFER_OK errno={errno} target={TARGET}")
