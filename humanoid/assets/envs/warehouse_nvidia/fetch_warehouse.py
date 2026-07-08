#!/usr/bin/env python3
"""fetch_warehouse.py — download the NVIDIA Isaac Sim 5.0 Simple_Warehouse scene locally.

MAY-171: the fixed ground-truth world for SLAM/nav dev. A single-file .usd grab from the
Asset Browser is an EMPTY crate (the geometry lives in a referenced payload/texture tree);
this mirrors the whole scene subtree from NVIDIA's public S3 bucket, preserving the on-server
directory layout so every relative reference — including full_warehouse's one `../../Props/
Forklift/...` escape — resolves locally with no USD editing.

    conda run -n isaac python humanoid/assets/envs/warehouse_nvidia/fetch_warehouse.py

Re-runnable: files already present with the right size are skipped. ~570 MB, ~2730 files.
"""
from __future__ import annotations

import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
# Everything below this server prefix is rooted under <this dir>/Isaac/ locally, so the
# relative offsets between Environments/Simple_Warehouse and Props/Forklift are preserved.
ROOT_PREFIX = "Assets/Isaac/5.0/Isaac/"
SUBTREES = [
    "Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/",
    "Assets/Isaac/5.0/Isaac/Props/Forklift/",
]
SKIP = ("/.thumbs/",)  # preview images — not needed to render
DEST = Path(__file__).resolve().parent / "Isaac"
WORKERS = 16


def list_keys(prefix: str) -> list[tuple[str, int]]:
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    out, token = [], None
    while True:
        url = f"{BASE}/?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(urllib.request.urlopen(url).read())
        for c in root.findall(f"{ns}Contents"):
            key = c.find(f"{ns}Key").text
            size = int(c.find(f"{ns}Size").text)
            if key.endswith("/") or any(s in key for s in SKIP):
                continue
            out.append((key, size))
        if root.find(f"{ns}IsTruncated").text != "true":
            break
        token = root.find(f"{ns}NextContinuationToken").text
    return out


def local_path(key: str) -> Path:
    return DEST / key[len(ROOT_PREFIX):]


def fetch(item: tuple[str, int]) -> tuple[str, str]:
    key, size = item
    dst = local_path(key)
    if dst.exists() and dst.stat().st_size == size:
        return ("skip", key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{urllib.parse.quote(key)}"
    for attempt in range(5):  # S3 drops the odd connection under 16-way concurrency
        try:
            urllib.request.urlretrieve(url, dst)
            if dst.stat().st_size == size:
                return ("get", key)
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))
    return ("fail", key)


def main() -> int:
    keys: list[tuple[str, int]] = []
    for st in SUBTREES:
        keys += list_keys(st)
    total_mb = sum(s for _, s in keys) / 1e6
    print(f"[fetch] {len(keys)} files, {total_mb:.1f} MB → {DEST}", flush=True)

    done = got = 0
    fails: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, k): k for k in keys}
        for fut in as_completed(futs):
            status, key = fut.result()
            done += 1
            got += status == "get"
            if status == "fail":
                fails.append(key)
            if done % 100 == 0 or done == len(keys):
                print(f"[fetch] {done}/{len(keys)}  ({got} downloaded, {len(fails)} failed)",
                      flush=True)
    print(f"[fetch] DONE — {got} downloaded, {len(fails)} failed", flush=True)
    if fails:
        print("[fetch] FAILED (re-run to retry):", flush=True)
        for k in fails[:20]:
            print("   ", k, flush=True)

    wh = DEST / "Environments/Simple_Warehouse/full_warehouse.usd"
    print(f"[fetch] scene: {wh}  ({'OK' if wh.exists() else 'MISSING!'})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
