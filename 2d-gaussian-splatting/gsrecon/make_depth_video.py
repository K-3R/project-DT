#!/usr/bin/env python
# [ver] make_depth_video.py 2026-08-25-r1  (ascii-only console/comments)
"""traj 깊이 비디오 재생성 -- 재랜더 없이 vis/depth_*.tiff 를 재사용한다.

render_utils.create_videos 는 첫 프레임의 백분위로 전체 컬러맵 범위를
고정한다. unbounded 씬은 먼 배경 껍질의 깊이가 섞여 범위가 틀어지고
책상 대역이 한 색으로 뭉개진다. 여기서는

  1. 전 프레임 표본(매 8번째)의 전역 백분위(기본 3-97)로 정규화
  2. --max-depth 로 상한 절단 (책상 대역에 컬러맵을 집중)

사용 (surfel_splatting env, GPU 불필요):
    python gsrecon/make_depth_video.py --dir output/take6/traj/ours_30000
    python gsrecon/make_depth_video.py --dir output/take6/traj/ours_30000 --max-depth 25
산출: <dir>/render_traj_depth_fixed.mp4 (원본 mp4 는 건드리지 않는다)
"""

import argparse
import glob
import os
import sys

import mediapy as media
import numpy as np
from PIL import Image

try:  # matplotlib 3.9+ 는 cm.get_cmap 가 제거됨 -- 신구 겸용
    from matplotlib import colormaps as _cmaps

    def get_cmap(name):
        return _cmaps[name]

except ImportError:
    from matplotlib import cm

    def get_cmap(name):
        return cm.get_cmap(name)


def parse_args():
    p = argparse.ArgumentParser(description="rebuild the traj depth video")
    p.add_argument("--dir", required=True, help="traj/ours_XXXX directory")
    p.add_argument(
        "--out",
        default="",
        help="output mp4 (default <dir>/render_traj_depth_fixed.mp4)",
    )
    p.add_argument(
        "--max-depth",
        type=float,
        default=0.0,
        help="clip depths above this (scene units). 0 = no clip. "
        "use to focus the colormap on the desk band",
    )
    p.add_argument("--pmin", type=float, default=3.0)
    p.add_argument("--pmax", type=float, default=97.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--per-frame",
        action="store_true",
        help="normalize each frame by its own percentiles (always vivid, "
        "but colors are not comparable across frames). default = global",
    )
    p.add_argument(
        "--unit-scale",
        type=float,
        default=1.0,
        help="meters per scene unit (take6 desk calibration: 0.1209). "
        "used by --range",
    )
    p.add_argument(
        "--range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN_M", "MAX_M"),
        help="fixed normalization range in meters (via --unit-scale), "
        "overrides percentiles. emphasizes the near band, comparable "
        "across frames. beyond MAX = background color",
    )
    p.add_argument(
        "--bg",
        default="70,12,12",
        help="background color r,g,b (0-255) for invalid/out-of-range "
        "pixels (default dark red)",
    )
    return p.parse_args()


def load_depth(path):
    return np.array(Image.open(path), dtype=np.float32)


def main():
    a = parse_args()
    # 정규화 모드는 셋 중 하나다: fixed(--range) > per-frame > global.
    # fixed 는 프레임 간 색 비교가 목적이므로 per-frame 과 동시 지정은 모순
    if a.range is not None:
        if a.per_frame:
            sys.exit(
                "[depthvid] ERROR: --range and --per-frame are exclusive "
                "(a fixed range IS the cross-frame normalization)"
            )
        if a.unit_scale <= 0 or a.range[0] <= 0 or a.range[1] <= a.range[0]:
            sys.exit(
                "[depthvid] ERROR: need 0 < MIN < MAX (meters) and " "--unit-scale > 0"
            )

    files = sorted(glob.glob(os.path.join(a.dir, "vis", "depth_*.tiff")))
    if not files:
        raise SystemExit(f"[depthvid] ERROR: no vis/depth_*.tiff under {a.dir}")
    print(f"[depthvid] {len(files)} depth frames")

    if a.range is not None:
        # 미터 고정 범위 (unit-scale 로 씬 유닛 환산). max-depth 도 맞춘다
        lo = np.log(a.range[0] / a.unit_scale)
        hi = np.log(a.range[1] / a.unit_scale)
        if a.max_depth <= 0:
            a.max_depth = a.range[1] / a.unit_scale
    else:
        # 전역 범위: 매 8번째 프레임 표본의 유효 깊이 백분위 (로그 공간)
        samp = []
        for f in files[::8]:
            d = load_depth(f).flatten()
            d = d[np.isfinite(d) & (d > 0)]
            if a.max_depth > 0:
                d = d[d <= a.max_depth]
            samp.append(d)
        allv = np.concatenate(samp)
        if allv.size == 0:
            sys.exit(
                "[depthvid] ERROR: no valid depth samples " "(--max-depth too small?)"
            )
        lo, hi = np.log(np.percentile(allv, [a.pmin, a.pmax]))
    print(
        "[depthvid] range: {:.3f} .. {:.3f} units (log {:.2f}..{:.2f})".format(
            float(np.exp(lo)), float(np.exp(hi)), lo, hi
        )
    )

    cmap = get_cmap("turbo")
    bg = np.array([int(v) for v in a.bg.split(",")], dtype=np.uint8)
    frames = []
    for f in files:
        d = load_depth(f)
        invalid = ~np.isfinite(d) | (d <= 0)
        if a.max_depth > 0:
            # 상한 초과는 min 으로 누르지 않고 무효(배경 회색) 처리한다.
            # 누르면 배경 전체가 컬러맵 끝색(어두운 적색)으로 몰려
            # 프레임이 통째로 검게 보인다 (실측)
            invalid = invalid | (d >= a.max_depth)
        flo, fhi = lo, hi
        if a.per_frame:
            valid = d[~invalid]
            if a.max_depth > 0:
                valid = valid[valid <= a.max_depth]
            if valid.size > 100:
                flo, fhi = np.log(np.percentile(valid, [a.pmin, a.pmax]))
        x = (np.log(np.where(invalid, 1.0, d)) - flo) / max(fhi - flo, 1e-9)
        x = np.clip(x, 0.0, 1.0)
        rgb = (cmap(x)[..., :3] * 255).astype(np.uint8)
        # 무효(구멍/빈 하늘/범위 밖)는 배경색으로 -- 컬러맵 값과 겹치지
        # 않는 색이어야 프레임이 통째로 단색으로 보이는 사고가 없다
        rgb[invalid] = bg
        frames.append(rgb)

    out = a.out or os.path.join(a.dir, "render_traj_depth_fixed.mp4")
    media.write_video(out, frames, fps=a.fps)
    print(f"[depthvid] wrote {out} ({len(frames)} frames, fps {a.fps})")


if __name__ == "__main__":
    main()
