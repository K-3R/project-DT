#!/usr/bin/env python
# [ver] check_assets.py 2026-08-10-r1  (ascii-only console/comments)
r"""
Nucleus 에셋 탐색기 -- 사무실 책상 씬에 쓸 USD 가 있는지 확인한다.

Isaac 에셋 서버는 버전마다 구성이 달라서, 모니터/키보드/마우스 같은
사무실 소품이 있는지는 실제로 뒤져 봐야 안다. 이 스크립트는 후보
디렉터리를 훑어 (1) 트리를 찍고 (2) 키워드에 걸리는 항목만 따로 모은다.

사용
    CUDA_VISIBLE_DEVICES=5 isaaclab.sh -p check_assets.py --headless
    CUDA_VISIBLE_DEVICES=5 isaaclab.sh -p check_assets.py --headless \
        --roots "{NVIDIA}/Assets/ArchVis" --depth 3

결과 해석
    hit 목록에 monitor/keyboard/mouse 가 나오면 그 USD 를 씬에 쓴다.
    안 나오면 프리미티브(박스/원통)로 책상 소품을 만든다.
"""

import argparse
import sys

parser = argparse.ArgumentParser(description="list Nucleus assets for the office scene")
parser.add_argument(
    "--roots",
    default="",
    help="extra dirs to scan, comma separated; {ISAAC} and {NVIDIA} expand",
)
parser.add_argument("--depth", type=int, default=2, help="recursion depth")
parser.add_argument(
    "--max-entries", type=int, default=60, help="max children printed per dir"
)
parser.add_argument(
    "--grep",
    default="monitor,keyboard,mouse,desk,office,screen,laptop,pen,marker,mug,cup,"
    "book,tray,bin,basket,box,paper,phone,lamp,chair",
    help="keywords to collect as hits",
)
parser.add_argument("--tree", type=int, default=1, help="print the full tree too")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.client  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR  # noqa: E402

DEFAULT_ROOTS = [
    "{ISAAC}/Props",
    "{ISAAC}/Environments",
    "{NVIDIA}/Assets/ArchVis",
    "{NVIDIA}/Assets/Vegetation",  # 없을 수 있음. 있으면 서버 구성을 알 수 있다
]

HITS = []


def expand(s):
    return s.replace("{ISAAC}", ISAAC_NUCLEUS_DIR).replace("{NVIDIA}", NVIDIA_NUCLEUS_DIR)


def listdir(url):
    """(name, is_dir) 목록. 접근 실패면 None."""
    try:
        result, entries = omni.client.list(url)
    except Exception as e:  # noqa: BLE001
        print(f"  [err] {url}: {e}")
        return None
    if result != omni.client.Result.OK:
        return None
    out = []
    for e in entries:
        is_dir = bool(e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)
        out.append((e.relative_path, is_dir))
    return sorted(out, key=lambda t: (not t[1], t[0].lower()))


def walk(url, depth, keys, indent=0):
    items = listdir(url)
    if items is None:
        print(f"{'  ' * indent}[x] cannot list: {url}")
        return
    shown = 0
    for name, is_dir in items:
        low = name.lower()
        child = f"{url}/{name}"
        if any(k in low for k in keys):
            HITS.append(child + ("/" if is_dir else ""))
        if args.tree and shown < args.max_entries:
            mark = "/" if is_dir else ""
            print(f"{'  ' * indent}{name}{mark}")
            shown += 1
        if is_dir and depth > 1:
            walk(child, depth - 1, keys, indent + 1)
    if args.tree and len(items) > shown:
        print(f"{'  ' * indent}... ({len(items) - shown} more)")


def main():
    keys = [k.strip().lower() for k in args.grep.split(",") if k.strip()]
    roots = DEFAULT_ROOTS + [r for r in args.roots.split(",") if r.strip()]

    print("=" * 70)
    print(f"ISAAC_NUCLEUS_DIR  = {ISAAC_NUCLEUS_DIR}")
    print(f"NVIDIA_NUCLEUS_DIR = {NVIDIA_NUCLEUS_DIR}")
    print(f"depth={args.depth}  keywords={keys}")
    print("=" * 70)

    for r in roots:
        url = expand(r)
        print(f"\n---- {url} ----")
        walk(url, args.depth, keys)

    print("\n" + "=" * 70)
    print(f"HITS ({len(HITS)}) -- office-scene candidates")
    print("=" * 70)
    for h in sorted(set(HITS)):
        print(h)
    if not HITS:
        print("(none) -> build desk props from primitives")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        import os

        os._exit(0)
