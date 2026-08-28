#!/usr/bin/env python
# [ver] preview_replica.py 2026-08-11-r1  (ascii-only console/comments)
r"""Replica 방 + 로봇 씬 미리보기.

카메라 시점은 변환된 USD 에 저장된 방 bbox(extent) 에서 자동으로 잡는다
(방 크기가 씬마다 다르므로). 벽/천장이 있는 스캔이라 카메라는 반드시
방 "안"에 있어야 한다 -- 자동 시점은 전부 실내로 계산된다.

사용
    CUDA_VISIBLE_DEVICES=5 isaaclab.sh -p preview_replica.py --headless
    CUDA_VISIBLE_DEVICES=5 isaaclab.sh -p preview_replica.py --headless \
        --bg-usd /root/project/datasets/replica/office_1.usd --robots 0

시점 (기본 8장, 자동)
    view0/1  눈높이 (+x 쪽 / -x 쪽에서 방을 가로질러 본다)
    view2-5  네 구석 위에서 방 중심을 내려다본다
    view6    천장 바로 아래에서 수직 부감 (평면 배치)
    view7    로봇 클로즈업 (--robots 1 일 때 의미 있음)
--views "ex,ey,ez@tx,ty,tz|..." 로 수동 지정 가능 (@target 생략 시 방 중심).
"""

import argparse
import os
import sys
import traceback

parser = argparse.ArgumentParser(description="render previews of the replica scene")
parser.add_argument("--out", default="/root/project/out/replica_preview")
parser.add_argument("--settle", type=int, default=60, help="physics steps before shot")
parser.add_argument(
    "--views",
    default="",
    help="manual camera list 'ex,ey,ez@tx,ty,tz|...'. Empty = auto from room bbox",
)
parser.add_argument(
    "--target", default="", help="fallback look-at for manual views without @target"
)

from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replica_scene as rsc  # noqa: E402

rsc.add_scene_args(parser)
parser.set_defaults(cam_w=1280, cam_h=720)
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


def auto_views(hx, hy, hh, a):
    """방 bbox 로 실내 시점 8개를 만든다: (eye, target) 목록.

    bbox 구석은 방이 직사각형이 아니면 실내가 아닐 수 있다 (흰 화면의
    원인 = 카메라가 벽/가구 속). 그래서 시점을 안쪽으로 당겨 잡는다 --
    가까워진 만큼은 넓은 화각(--cam-fov)이 보상한다.
    """
    cz = 0.25 * hh  # 방 중심을 낮게 잡아 가구가 화면 가운데 오게
    top_z = max(1.8, hh - 0.45)
    views = [
        ((0.70 * hx, 0.0, 1.50), (-0.3 * hx, 0.0, 0.80)),
        ((-0.70 * hx, 0.0, 1.50), (0.3 * hx, 0.0, 0.80)),
        ((0.62 * hx, 0.62 * hy, 0.72 * hh), (0.0, 0.0, cz)),
        ((0.62 * hx, -0.62 * hy, 0.72 * hh), (0.0, 0.0, cz)),
        ((-0.62 * hx, 0.62 * hy, 0.72 * hh), (0.0, 0.0, cz)),
        ((-0.62 * hx, -0.62 * hy, 0.72 * hh), (0.0, 0.0, cz)),
        ((0.05, 0.0, top_z), (0.0, 0.0, 0.0)),
    ]
    # 로봇 클로즈업: 리그 앞쪽 비스듬히
    ox, oy = rsc.rot2d(1.2, 0.9, a.base_yaw)
    views.append(
        (
            (a.base_x + ox, a.base_y + oy, a.base_z + 0.55),
            (a.base_x, a.base_y, a.base_z + 0.20),
        )
    )
    return views


def parse_views(a, hx, hy, hh):
    if not a.views.strip():
        return auto_views(hx, hy, hh, a)
    fallback = rsc.vec(a.target) if a.target.strip() else (0.0, 0.0, 0.25 * hh)
    out = []
    for item in a.views.split("|"):
        if "@" in item:
            e, t = item.split("@")
            out.append((rsc.vec(e), rsc.vec(t)))
        else:
            out.append((rsc.vec(item), fallback))
    return out


def main():
    hx, hy, hh = rsc.room_bbox(args.bg_usd)
    print(f"[preview] room half-extents x={hx:.2f} y={hy:.2f} height={hh:.2f}")

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(rsc.build(args, with_camera=True))
    sim.reset()
    cam = scene["scene_cam"]
    dt = sim.get_physics_dt()

    for _ in range(args.settle):  # 물리 안정화 + 렌더 누적버퍼 워밍업
        sim.step(render=True)
        scene.update(dt)

    import imageio

    os.makedirs(args.out, exist_ok=True)
    for vi, (eye, tgt) in enumerate(parse_views(args, hx, hy, hh)):
        cam.set_world_poses_from_view(
            eyes=torch.tensor([eye], device=sim.device),
            targets=torch.tensor([tgt], device=sim.device),
        )
        for _ in range(30):  # 시점 변경 후 누적버퍼 재수렴
            sim.step(render=True)
            scene.update(dt)
        rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        path = os.path.join(args.out, f"view{vi}.png")
        imageio.imwrite(path, rgb.astype(np.uint8))
        print(
            "[preview] {}  eye=({:.2f},{:.2f},{:.2f}) target=({:.2f},{:.2f},{:.2f})".format(
                path, *eye, *tgt
            )
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
