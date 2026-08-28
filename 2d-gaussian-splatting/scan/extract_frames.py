#!/usr/bin/env python
# ======================================
# File: extract_frames.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-28
# ======================================
# [ver] extract_frames.py 2026-08-20-r1  (ascii-only console/comments)
"""scan 영상(mp4) -> COLMAP 입력 이미지. 2DGS pipeline 1단계.

폴더를 주면 안에 있는 영상을 전부 처리. 영상마다 길이를 재서 목표 frame 수에
맞는 fps 를 잡으므로 1분짜리든 5분짜리든 비슷한 frame 수로 수렴함
(2DGS/3DGS 권장 100~300 frame).

mode 두 가지
    per-video (기본)  영상 하나 = scene 하나. <out>/<영상이름>/input/
    --merge <이름>    영상 여러 개를 한 scene 으로. 같은 공간을 나눠 찍은 경우.
                      파일명은 v01_0001.jpg, v02_0001.jpg ... 로 안 겹치게

사용
    python scan/extract_frames.py --src data/raw
    python scan/extract_frames.py --src <dir> --merge desk --target 300
    python scan/extract_frames.py --src <dir>/one.mp4          # 파일 하나만

산출: <out>/<scene>/input/0001.jpg ...  (2DGS convert.py 가 기대하는 형태)

다음 단계
    conda activate surfel_splatting
    cd <repo>/2d-gaussian-splatting && python convert.py -s <out>/<scene>
    (scan/run_scan.sh 는 이 script 를 1단계로 부름)
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

# 비교는 소문자로 (액션캠 파일은 .MP4/.AVI)
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")

# raw 영상 -> frame -> COLMAP -> 학습 -> mesh 를 전부 이 repo 아래에 둠.
# data/ 와 output/ 은 .gitignore 라 저장소는 안 부풀음.
#   data/raw/*.mp4  ->  data/<scene>/input/  ->  output/<scene>/...
REPO_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # scan/ 의 한 단계 위 = repo root
DEFAULT_OUT = os.path.join(REPO_DIR, "data")

# fps 상하한. 너무 성기면 COLMAP matching 이 끊기고, 너무 촘촘하면 같은 장면만 쌓임
FPS_MIN, FPS_MAX = 0.5, 10.0


# arguments (argparse)
def parse_args():
    p = argparse.ArgumentParser(
        description="extract COLMAP input frames from scan videos"
    )
    p.add_argument(
        "--src",
        required=True,
        help="directory holding the videos, or a single video file",
    )
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="scene root; each scene becomes <out>/<name>/input/",
    )
    p.add_argument(
        "--merge",
        default="",
        help="scene name to merge ALL videos into (default: one scene per video)",
    )
    p.add_argument(
        "--target",
        type=int,
        default=250,
        help="target number of frames (per scene; 100-300 recommended)",
    )
    p.add_argument(
        "--quality",
        type=int,
        default=1,
        help="jpg quality for ffmpeg -qscale:v (1 = best)",
    )
    p.add_argument(
        "--gray",
        action="store_true",
        help="desaturate frames to grayscale (keeps 3 channels; "
        "COLMAP/2DGS load them unchanged)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing non-empty input/ directory",
    )
    p.add_argument(
        "--no-check",
        action="store_true",
        help="skip the blur/sharpness check (needs opencv anyway)",
    )
    return p.parse_args()


# ---- 입력 훑기 --------------------------------------------------------------
def need(tool):
    """PATH 에 있어야 하는 외부 도구"""
    if shutil.which(tool) is None:
        sys.exit(f"[frames] ERROR: {tool} not found in PATH")


def list_videos(src):
    """src 가 파일이면 그 하나, 폴더면 안의 영상 전부"""
    if os.path.isfile(src):
        return [src]
    if not os.path.isdir(src):
        sys.exit(f"[frames] ERROR: src not found: {src}")
    vids = sorted(
        p for p in glob.glob(os.path.join(src, "*")) if p.lower().endswith(VIDEO_EXT)
    )
    if not vids:
        sys.exit(f"[frames] ERROR: no videos in {src} (looked for {VIDEO_EXT})")
    return vids


def duration_of(path):
    """ffprobe 로 영상 길이 [s]"""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        # 손상 영상 같은 진짜 원인은 ffprobe 쪽에 찍혀 있음
        tail = "\n".join(out.stderr.strip().splitlines()[-2:])
        sys.exit(f"[frames] ERROR: cannot read duration of {path}\n{tail}")


def fps_for(duration, target):
    """duration 초짜리에서 target frame 수가 나오는 fps (상하한 clamp)"""
    return min(max(target / duration, FPS_MIN), FPS_MAX)


# ---- frame 뽑기 -----------------------------------------------------------
def prepare_dir(in_dir, force):
    """빈 input/ 준비. 이미 파일이 있으면 --force 없이는 중단"""
    if os.path.isdir(in_dir) and os.listdir(in_dir):
        if not force:
            sys.exit(
                f"[frames] ERROR: {in_dir} already has files "
                "(use --force to overwrite, or pick another scene name)"
            )
        shutil.rmtree(in_dir)
    os.makedirs(in_dir, exist_ok=True)


def extract(video, in_dir, fps, quality, prefix="", gray=False):
    """ffmpeg 로 frame 추출. prefix 는 merge mode 파일명 겹침 방지용.

    gray 는 채도만 없앰 (hue=s=0). 1-channel gray jpg 로 만들면 COLMAP/2DGS
    loader 를 손봐야 해서 3-channel 유지
    """
    pattern = os.path.join(in_dir, f"{prefix}%04d.jpg")
    vf = f"fps={fps:.3f}" + (",hue=s=0" if gray else "")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        video,
        "-qscale:v",
        str(quality),
        "-qmin",
        "1",
        "-vf",
        vf,
        pattern,
    ]
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(f"[frames] ERROR: ffmpeg failed ({rc}) on {video}")


# ---- 결과 점검 --------------------------------------------------------------
def sharpness_report(in_dir):
    """Laplacian 분산으로 흐린 frame 비율 확인. opencv 없으면 건너뜀"""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("[frames] (opencv not installed -- skipping blur check)")
        return
    paths = sorted(glob.glob(os.path.join(in_dir, "*.jpg")))
    vals = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            vals.append((float(cv2.Laplacian(img, cv2.CV_64F).var()), p))
    if not vals:
        return
    med = float(np.median([v for v, _ in vals]))
    # 중앙값의 30% 미만 = 눈에 띄게 흐림, motion blur 후보
    blurry = [p for v, p in vals if v < 0.3 * med]
    print(f"[frames] sharpness median {med:.0f}, blurry {len(blurry)}/{len(vals)}")
    if len(blurry) > 0.15 * len(vals):
        print("[frames] WARNING: many blurry frames -- move slower when shooting")
        print("[frames] worst: " + ", ".join(os.path.basename(p) for p in blurry[:5]))


def finish(scene, in_dir, no_check):
    """frame 수/용량 요약 + frame 수 경고"""
    jpgs = glob.glob(os.path.join(in_dir, "*.jpg"))
    size_mb = sum(os.path.getsize(p) for p in jpgs) / 1e6
    print(f"[frames] {scene}: {len(jpgs)} images ({size_mb:.0f} MB) -> {in_dir}")
    if len(jpgs) < 80:
        print("[frames] WARNING: fewer than 80 images -- reconstruction may fail")
    elif len(jpgs) > 400:
        print("[frames] WARNING: more than 400 images -- COLMAP will be slow")
    if not no_check:
        sharpness_report(in_dir)


# ---- scene 만들기 ----------------------------------------------------------
def scene_merged(videos, a):
    """영상 여러 개 -> scene 하나. 총 길이 기준 fps 라 합쳐서 target frame 수"""
    scene_dir = os.path.join(a.out, a.merge)
    in_dir = os.path.join(scene_dir, "input")
    prepare_dir(in_dir, a.force)

    durs = [duration_of(v) for v in videos]
    total = sum(durs)
    fps = fps_for(total, a.target)
    print(f"[frames] merge '{a.merge}': total {total:.0f}s -> fps {fps:.3f}")

    for i, (v, d) in enumerate(zip(videos, durs), start=1):
        print(f"[frames]   v{i:02d} {os.path.basename(v)} ({d:.0f}s)")
        extract(v, in_dir, fps, a.quality, prefix=f"v{i:02d}_", gray=a.gray)
    finish(a.merge, in_dir, a.no_check)
    return [scene_dir]


def scenes_per_video(videos, a):
    """영상 하나 = scene 하나. 파일 이름이 곧 scene 이름"""
    scenes = []
    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        scene_dir = os.path.join(a.out, name)
        in_dir = os.path.join(scene_dir, "input")
        prepare_dir(in_dir, a.force)

        d = duration_of(v)
        fps = fps_for(d, a.target)
        print(f"[frames] {name}: {d:.0f}s -> fps {fps:.3f}")
        extract(v, in_dir, fps, a.quality, gray=a.gray)
        finish(name, in_dir, a.no_check)
        scenes.append(scene_dir)
    return scenes


def main():
    a = parse_args()
    need("ffmpeg")
    need("ffprobe")
    videos = list_videos(a.src)
    print(f"[frames] {len(videos)} video(s), target {a.target} frames/scene")

    scenes = scene_merged(videos, a) if a.merge else scenes_per_video(videos, a)

    print("[frames] next:")
    for s in scenes:
        print(f"  python convert.py -s {s}")


if __name__ == "__main__":
    main()
