#!/usr/bin/env python
# [ver] check_3view_demos.py 2026-08-11-r1  (ascii-only console/comments)
r"""
재렌더 산출물(seed_3view.hdf5) 검수 도구.

구조 검사 (전 데모):
    - 원본 seed.hdf5 와 데모 수 일치
    - obs/left_wrist_view, obs/right_wrist_view 존재 + (T,H,W,3) uint8
    - T 가 actions / ego_view 와 일치
    - 원본 데이터(actions/joint_pos/ego_view) 가 바뀌지 않았는지 표본 대조
눈 검사용 샘플:
    - 데모 몇 개의 [ego | wrist_l | wrist_r] 가로 결합 PNG (처음/중간/끝)
    - --video 를 주면 파일당 첫 샘플 데모의 결합 mp4 도 저장

사용 (호스트, gr00t_sh 환경 -- NAS 직접 읽음, 컨테이너 불필요):
    python check_3view_demos.py \
        --input-root /data1/huggingface/sslunder54/datasets/franka_bimanual \
        --out ~/project/gr00t_Isaacsim/out/check_3view --video

필요 패키지: h5py numpy imageio (imageio-ffmpeg 는 --video 때만)
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np


def check_file(path, src_path, sample_demos, out_dir, make_video):
    import imageio

    name = os.path.basename(os.path.dirname(path))
    bad = []
    with h5py.File(path, "r") as h:
        grp = h["data"]
        keys = sorted(grp.keys(), key=lambda s: int(s.split("_")[-1]))

        # 원본과 데모 수 대조
        n_src = None
        if os.path.exists(src_path):
            with h5py.File(src_path, "r") as hs:
                n_src = len(hs["data"])
                # 원본 불변 표본 대조: 첫 데모의 actions / ego_view 첫 프레임
                k0 = keys[0]
                a_new = grp[k0]["actions"][...]
                a_src = hs["data"][k0]["actions"][...]
                if not np.array_equal(a_new, a_src):
                    bad.append((k0, "actions changed vs source"))
                e_new = grp[k0]["obs/ego_view"][0]
                e_src = hs["data"][k0]["obs/ego_view"][0]
                if not np.array_equal(e_new, e_src):
                    bad.append((k0, "ego_view changed vs source"))
        if n_src is not None and len(keys) != n_src:
            bad.append(("*", f"demo count {len(keys)} != source {n_src}"))

        # 전 데모 구조 검사 (메타데이터만이라 빠름)
        for k in keys:
            g = grp[k]
            try:
                T = g["actions"].shape[0]
                for w in ("left_wrist_view", "right_wrist_view"):
                    p = f"obs/{w}"
                    if p not in g:
                        bad.append((k, f"missing {p}"))
                        continue
                    d = g[p]
                    if d.dtype != np.uint8 or d.ndim != 4 or d.shape[3] != 3:
                        bad.append((k, f"{w} bad shape/dtype {d.shape} {d.dtype}"))
                    elif d.shape[0] != T:
                        bad.append((k, f"{w} T={d.shape[0]} != actions {T}"))
                if g["obs/ego_view"].shape[0] != T:
                    bad.append((k, f"ego T != actions {T}"))
            except Exception as e:  # noqa: BLE001
                bad.append((k, f"read error: {e}"))

        # 눈 검사용 샘플 저장
        os.makedirs(out_dir, exist_ok=True)
        picks = keys[:: max(len(keys) // max(sample_demos, 1), 1)][:sample_demos]
        for k in picks:
            g = grp[k]
            T = g["actions"].shape[0]
            for t in (0, T // 2, T - 1):
                row = np.concatenate(
                    [
                        g["obs/ego_view"][t],
                        g["obs/left_wrist_view"][t],
                        g["obs/right_wrist_view"][t],
                    ],
                    axis=1,
                )
                imageio.imwrite(
                    os.path.join(out_dir, f"{name}_{k}_f{t:03d}.png"), row
                )
        if make_video and picks:
            k = picks[0]
            g = grp[k]
            frames = np.concatenate(
                [
                    g["obs/ego_view"][...],
                    g["obs/left_wrist_view"][...],
                    g["obs/right_wrist_view"][...],
                ],
                axis=2,
            )
            vp = os.path.join(out_dir, f"{name}_{k}_3view.mp4")
            try:
                # format 을 명시해 ffmpeg 백엔드가 없을 때 다른 플러그인
                # (tifffile 등)으로 흘러가 죽는 것을 막는다
                imageio.mimsave(
                    vp,
                    frames,
                    fps=20,
                    codec="libx264",
                    pixelformat="yuv420p",
                    format="FFMPEG",
                )
                print(f"[chk] video -> {vp}")
            except Exception as e:  # noqa: BLE001
                print(
                    f"[chk] WARNING: video skipped ({e}) -- "
                    "pip install imageio-ffmpeg, or drop --video"
                )

    status = "OK" if not bad else f"BAD ({len(bad)} issues)"
    print(f"[chk] {name}: {len(keys)} demos {status}")
    for k, msg in bad[:10]:
        print(f"       {k}: {msg}")
    return len(bad)


def main():
    p = argparse.ArgumentParser(description="verify rerendered 3-view HDF5 files")
    p.add_argument("--input-root", required=True)
    p.add_argument("--hdf5-name", default="seed_3view.hdf5")
    p.add_argument("--src-name", default="seed.hdf5")
    p.add_argument("--out", default="./check_3view", help="dir for sample PNG/mp4")
    p.add_argument("--sample-demos", type=int, default=3, help="demos to sample per file")
    p.add_argument("--video", action="store_true", help="also write one mp4 per file")
    a = p.parse_args()

    files = sorted(
        glob.glob(os.path.join(os.path.expanduser(a.input_root), "*", a.hdf5_name))
    )
    if not files:
        print(f"[chk] ERROR: no */{a.hdf5_name} under {a.input_root}")
        sys.exit(1)

    total_bad = 0
    for f in files:
        src = os.path.join(os.path.dirname(f), a.src_name)
        total_bad += check_file(
            f, src, a.sample_demos, os.path.expanduser(a.out), a.video
        )

    if total_bad:
        print(f"[chk] FAILED: {total_bad} issues total")
        sys.exit(1)
    print(f"[chk] ALL OK ({len(files)} files). samples -> {a.out}")


if __name__ == "__main__":
    main()
