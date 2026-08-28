#!/usr/bin/env python
# [ver] extract_frames.py 2026-08-20-r1  (ascii-only console/comments)
"""스캔 영상(mp4) -> COLMAP 입력 이미지 (2DGS 파이프라인 1단계).

폴더를 주면 그 안의 영상을 전부 처리한다. 영상마다 길이를 재서 목표
장수에 맞는 fps 를 자동으로 계산하므로 (2DGS/3DGS 권장 100~300 장),
1분짜리든 5분짜리든 적정 장수로 수렴한다.

두 가지 모드
    per-video (기본)  영상 하나 = 씬 하나. <out>/<영상이름>/input/
    --merge <이름>    영상 여러 개를 한 씬으로 합침 (같은 공간을 여러 번
                      나눠 찍은 경우). <out>/<이름>/input/ 에 파일명 충돌
                      없이 이어 붙인다 (v01_0001.jpg, v02_0001.jpg ...)

사용
    python gsrecon/extract_frames.py --src data/raw
    python gsrecon/extract_frames.py --src <dir> --merge desk --target 300
    python gsrecon/extract_frames.py --src <dir>/one.mp4          # 파일 하나만

산출 구조 (2DGS convert.py 가 기대하는 형태)
    <out>/<scene>/input/0001.jpg ...

다음 단계
    conda activate surfel_splatting
    cd ~/project/2d-gaussian-splatting && python convert.py -s <out>/<scene>
    (또는 gsrecon/run_recon.sh 가 이 스크립트를 1단계로 호출한다)
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

# 대소문자는 비교 시 소문자화로 처리한다 (액션캠의 .MP4/.AVI 등)
VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv")
# 전 과정(raw 영상 -> 프레임 -> COLMAP -> 학습 -> 메시)을 이 레포 아래에
# 둔다. data/ 와 output/ 은 .gitignore 되어 있어 저장소가 안 부푼다.
#   data/raw/*.mp4  ->  data/<scene>/input/  ->  output/<scene>/...
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gsrecon/ 의 한 단계 위 = 레포 루트
DEFAULT_OUT = os.path.join(REPO_DIR, "data")


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


def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f"[frames] ERROR: {tool} not found in PATH")


def list_videos(src):
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
        # ffprobe 가 말해준 실제 원인(손상 영상 등)을 같이 보여준다
        tail = "\n".join(out.stderr.strip().splitlines()[-2:])
        sys.exit(f"[frames] ERROR: cannot read duration of {path}\n{tail}")


def extract(video, in_dir, fps, quality, prefix="", gray=False):
    """ffmpeg 로 프레임을 뽑는다. prefix 는 merge 모드의 파일명 충돌 방지용.

    gray=True 는 채도만 제거한다 (hue=s=0). 1채널 gray jpg 대신 3채널을
    유지해야 COLMAP/2DGS 로더가 무수정으로 읽는다."""
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


def sharpness_report(in_dir):
    """라플라시안 분산으로 흐린 프레임 비율을 본다 (opencv 없으면 건너뜀)."""
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
    # 중앙값의 30% 미만 = 눈에 띄게 흐린 프레임 (모션 블러 후보)
    blurry = [p for v, p in vals if v < 0.3 * med]
    print(f"[frames] sharpness median {med:.0f}, blurry {len(blurry)}/{len(vals)}")
    if len(blurry) > 0.15 * len(vals):
        print("[frames] WARNING: many blurry frames -- move slower when shooting")
        print("[frames] worst: " + ", ".join(os.path.basename(p) for p in blurry[:5]))


def prepare_dir(in_dir, force):
    if os.path.isdir(in_dir) and os.listdir(in_dir):
        if not force:
            sys.exit(
                f"[frames] ERROR: {in_dir} already has files "
                "(use --force to overwrite, or pick another scene name)"
            )
        shutil.rmtree(in_dir)
    os.makedirs(in_dir, exist_ok=True)


def finish(scene, in_dir, no_check):
    n = len(glob.glob(os.path.join(in_dir, "*.jpg")))
    size_mb = (
        sum(os.path.getsize(p) for p in glob.glob(os.path.join(in_dir, "*.jpg"))) / 1e6
    )
    print(f"[frames] {scene}: {n} images ({size_mb:.0f} MB) -> {in_dir}")
    if n < 80:
        print("[frames] WARNING: fewer than 80 images -- reconstruction may fail")
    elif n > 400:
        print("[frames] WARNING: more than 400 images -- COLMAP will be slow")
    if not no_check:
        sharpness_report(in_dir)
    return n


def main():
    a = parse_args()
    need("ffmpeg")
    need("ffprobe")
    videos = list_videos(a.src)
    print(f"[frames] {len(videos)} video(s), target {a.target} frames/scene")

    if a.merge:
        # 여러 영상 -> 한 씬. 총 길이로 fps 를 잡아 전체가 target 장이 되게 한다
        scene_dir = os.path.join(a.out, a.merge)
        in_dir = os.path.join(scene_dir, "input")
        prepare_dir(in_dir, a.force)
        durs = [duration_of(v) for v in videos]
        total = sum(durs)
        fps = min(max(a.target / total, 0.5), 10.0)
        print(f"[frames] merge '{a.merge}': total {total:.0f}s -> fps {fps:.3f}")
        for i, (v, d) in enumerate(zip(videos, durs), start=1):
            print(f"[frames]   v{i:02d} {os.path.basename(v)} ({d:.0f}s)")
            extract(v, in_dir, fps, a.quality, prefix=f"v{i:02d}_", gray=a.gray)
        finish(a.merge, in_dir, a.no_check)
        scenes = [scene_dir]
    else:
        scenes = []
        for v in videos:
            name = os.path.splitext(os.path.basename(v))[0]
            scene_dir = os.path.join(a.out, name)
            in_dir = os.path.join(scene_dir, "input")
            prepare_dir(in_dir, a.force)
            d = duration_of(v)
            fps = min(max(a.target / d, 0.5), 10.0)
            print(f"[frames] {name}: {d:.0f}s -> fps {fps:.3f}")
            extract(v, in_dir, fps, a.quality, gray=a.gray)
            finish(name, in_dir, a.no_check)
            scenes.append(scene_dir)

    print("[frames] next:")
    for s in scenes:
        print(f"  python convert.py -s {s}")


if __name__ == "__main__":
    main()
