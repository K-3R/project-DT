#!/usr/bin/env python
# [ver] inspect_usd.py 2026-08-10-r1  (ascii-only console/comments)
r"""
USD 프림 트리 덤프 -- 기존 에셋 안에 쓸 만한 소품이 들어있는지 확인한다.

Environments/Office/office.usd 같은 씬 에셋은 방 전체가 한 파일이라
모니터/키보드 같은 소품이 하위 프림으로 들어있을 수 있다. 그렇다면
새로 내려받을 필요 없이 그 프림만 참조해서 쓰면 된다.

사용
    CUDA_VISIBLE_DEVICES=3 isaaclab.sh -p inspect_usd.py --headless
    CUDA_VISIBLE_DEVICES=3 isaaclab.sh -p inspect_usd.py --headless \
        --usd "{ISAAC}/Props/KLT_Bin" --depth 4     # 디렉터리면 안의 usd 전부

출력
    hit  = 키워드에 걸린 프림 (경로 + 타입 + 월드 바운딩박스 크기)
    바운딩박스로 실제 크기를 알 수 있어 배치 스케일을 바로 정할 수 있다.
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="dump USD prim tree and find props")
parser.add_argument(
    "--usd",
    default="{ISAAC}/Environments/Office/office.usd",
    help="usd file or directory; {ISAAC} and {NVIDIA} expand",
)
parser.add_argument("--depth", type=int, default=6, help="max prim depth to print")
parser.add_argument(
    "--grep",
    default="monitor,keyboard,mouse,screen,display,laptop,desk,tv,pc,computer,"
    "phone,lamp,book,cup,mug,pen,bin,tray,box",
    help="keywords to report as hits",
)
parser.add_argument("--max-prims", type=int, default=400, help="print limit")
parser.add_argument("--bbox", type=int, default=1, help="compute bbox for hits")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.client  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR  # noqa: E402


def expand(s):
    return s.replace("{ISAAC}", ISAAC_NUCLEUS_DIR).replace(
        "{NVIDIA}", NVIDIA_NUCLEUS_DIR
    )


def usd_files(url):
    """파일이면 그대로, 디렉터리면 안의 usd 목록."""
    if url.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
        return [url]
    try:
        result, entries = omni.client.list(url)
    except Exception:  # noqa: BLE001
        return []
    if result != omni.client.Result.OK:
        return []
    out = []
    for e in entries:
        if e.relative_path.lower().endswith((".usd", ".usda", ".usdc", ".usdz")):
            out.append(f"{url}/{e.relative_path}")
    return sorted(out)


def dump(url, keys):
    print(f"\n{'=' * 70}\n{url}\n{'=' * 70}")
    stage = Usd.Stage.Open(url)
    if stage is None:
        print("  [x] cannot open")
        return
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    hits, shown = [], 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        depth = path.count("/")
        name = prim.GetName()
        tname = prim.GetTypeName()
        if depth <= args.depth and shown < args.max_prims:
            print(f"{'  ' * (depth - 1)}{name}  <{tname}>")
            shown += 1
        if any(k in name.lower() for k in keys):
            size = ""
            if args.bbox:
                try:
                    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    if not rng.IsEmpty():
                        s = rng.GetSize()
                        size = f"  size=({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})"
                except Exception:  # noqa: BLE001
                    size = "  size=?"
            hits.append(f"{path}  <{tname}>{size}")
    print(f"\n-- hits in this file ({len(hits)}) --")
    for h in hits:
        print(" ", h)
    if not hits:
        print("  (none)")


def main():
    keys = [k.strip().lower() for k in args.grep.split(",") if k.strip()]
    files = usd_files(expand(args.usd))
    if not files:
        print(f"[x] no usd found at {expand(args.usd)}")
        return
    print(f"[inspect] {len(files)} file(s), keywords={keys}")
    for f in files:
        dump(f, keys)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(0)
