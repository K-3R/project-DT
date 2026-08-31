#!/usr/bin/env python
# ======================================
# File: preview_office.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] preview_office.py 2026-08-10-r2  (ascii-only console/comments)
r"""
사무실 책상 scene 미리보기 -- 배치를 눈으로 확인하고 조정하기 위한 도구.

물리를 잠깐 돌려 안정시킨 뒤 여러 시점에서 PNG 를 저장함. 로봇/물건
없이 배경(책상 + 모니터 2대 + 키보드/마우스)만 먼저 확인하는 용도.

사용
    CUDA_VISIBLE_DEVICES=3 isaaclab.sh -p preview_office.py --headless
    CUDA_VISIBLE_DEVICES=3 isaaclab.sh -p preview_office.py --headless \
        --robots 1 --items 4        # 나중 단계 확인

시점 (기본 3장)
    view0  사람 시점 (책상 정면, 나중에 로봇 카메라가 설 자리)
    view1  비스듬한 3인칭 (배치 확인용)
    view2  위에서 내려다보기 (평면 배치 확인용)
"""

import argparse
import os
import sys
import traceback

# 1. arguments (argparse) ------------------------------------------------
parser = argparse.ArgumentParser(
    description="render a preview of the office desk scene"
)
parser.add_argument(
    "--out",
    default="/root/project/out/office_preview",
)
parser.add_argument(
    "--settle",
    type=int,
    default=150,
    help="physics steps before shot",
)
parser.add_argument(
    "--target",
    default="",
    help="camera look-at point; empty = office_scene.EGO_TARGET",
)
parser.add_argument(
    "--views",
    default=(
        "1.20,0.00,0.90"  # 0 ego_view (학습용 시점. 선 사람이 어깨 너머로 봄)
        "|0.75,0.00,0.55"  # 1 앉은 눈높이 (비교용)
        "|1.25,1.05,0.85"  # 2 오른쪽 비스듬히
        "|1.25,-1.05,0.85"  # 3 왼쪽 비스듬히
        "|0.30,0.00,1.90"  # 4 위에서 내려다보기 (평면 배치)
        "|2.20,1.80,1.60"  # 5 멀리서 전체, 오른쪽 위 (partition 까지)
        "|2.20,-1.80,1.60"  # 6 멀리서 전체, 왼쪽 위 (5 의 반대편)
        "|0.55,0.55,0.30"  # 7 책상면 close-up (물건 확인용)
    ),
    help="camera eye positions, pipe separated",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
)
parser.add_argument(
    "--item-region",
    default="0.00,0.28,-0.35,0.60",
    help="xmin,xmax,ymin,ymax for scattered markers; keeps clear of the "
    "keyboard row and stays inside both arms' reach",
)
parser.add_argument(
    "--min-sep",
    type=float,
    default=0.10,
)

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import office_scene as osc  # noqa: E402

osc.add_scene_args(parser)
parser.set_defaults(cam_w=960, cam_h=720)  # 미리보기는 크게 봄
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def vec(s):
    return tuple(float(x) for x in s.split(","))


def place_items(scene, sim, rng, a):
    """정리 대상 물건을 구역에 겹치지 않게 뿌림 (2단계용)."""
    if a.items <= 0:
        return
    xmin, xmax, ymin, ymax = vec(a.item_region)
    pts = []
    for i in range(1, min(a.items, osc.MAX_ITEMS) + 1):
        for _ in range(200):
            p = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
            if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= a.min_sep for q in pts):
                pts.append(p)
                break
        else:
            pts.append((xmin, ymin))
        obj = scene[f"item_{i}"]
        st = obj.data.default_root_state.clone()
        st[:, 0:3] = (
            torch.tensor(
                [(pts[-1][0], pts[-1][1], osc.item_half_h(i))], device=sim.device
            )
            + scene.env_origins
        )
        # cylinder 축이 기본 z 라 y 축 90도로 눕히고, 그 위에 무작위 yaw 를 줌
        lay = osc.quat_mul(
            osc.quat_axis("z", float(rng.uniform(-180.0, 180.0))),
            osc.quat_axis("y", 90.0),
        )
        st[:, 3:7] = torch.tensor([lay], device=sim.device)
        st[:, 7:] = 0.0
        obj.write_root_state_to_sim(st)


def main():
    rng = np.random.default_rng(args.seed)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(osc.build(args, with_camera=True))
    sim.reset()
    cam = scene["scene_cam"]
    dt = sim.get_physics_dt()

    place_items(scene, sim, rng, args)
    scene.write_data_to_sim()

    for _ in range(args.settle):  # 물리 안정화 + render 누적 buffer warmup
        sim.step(render=True)
        scene.update(dt)

    import imageio

    os.makedirs(args.out, exist_ok=True)
    tgt = vec(args.target) if args.target.strip() else osc.EGO_TARGET
    target = torch.tensor([tgt], device=sim.device)
    for vi, eye in enumerate(args.views.split("|")):
        cam.set_world_poses_from_view(
            eyes=torch.tensor([vec(eye)], device=sim.device), targets=target
        )
        for _ in range(30):  # 시점 변경 후 누적 buffer 재수렴
            sim.step(render=True)
            scene.update(dt)
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        path = os.path.join(args.out, f"view{vi}.png")
        imageio.imwrite(path, rgb.astype(np.uint8))
        print(f"[preview] {path}  eye={eye}")

    if args.items > 0:
        for i in range(1, min(args.items, osc.MAX_ITEMS) + 1):
            p = scene[f"item_{i}"].data.root_pos_w[0].cpu().numpy()
            print(
                f"[item {i}] {osc.item_name(i):8s} pos={np.round(p, 3).tolist()} "
                f"grasp_w={osc.item_grasp_w(i):.3f}"
            )
    print(f"[preview] done -> {args.out}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
