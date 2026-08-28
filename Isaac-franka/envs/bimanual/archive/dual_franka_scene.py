#!/usr/bin/env python
# [ver] dual_franka_scene.py 2026-08-05-r1
"""
Franka 2대 양팔 환경 타당성 확인.

Isaac Lab 2.2.1 에는 양팔 로봇 에셋이 없다 (양팔은 GR1T2 휴머노이드뿐).
대신 씬에 Franka 를 두 대 놓으면 양팔 워크셀이 된다 - robosuite 의
TwoArm 환경이 Panda 2대로 하는 것과 같은 구성이다.

이 스크립트가 확인하는 것
  1. Franka 2대가 한 씬에 뜨고 물리가 안정적인가
  2. 두 팔의 작업 공간이 실제로 겹치는가 (협응 태스크가 성립하는가)
  3. 카메라에 두 팔이 함께 담기는가

사용법
  /root/project/IsaacLab/isaaclab.sh -p dual_franka_scene.py --headless --steps 300
"""

import argparse
import os
import sys
import traceback

# AppLauncher 는 다른 isaaclab 임포트보다 먼저 와야 한다.
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--layout", choices=["parallel", "opposed"], default="parallel",
                    help="parallel = 나란히 같은 방향 / opposed = 마주보게")
parser.add_argument("--base-sep", type=float, default=0.9,
                    help="두 베이스 사이 거리 [m]. Franka 리치가 약 0.855 라 "
                         "이 값이 크면 공유 작업공간이 사라진다")
parser.add_argument("--steps", type=int, default=300, help="물리 스텝 수")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--table", default="dual",
                    help="'dual'=Isaac 테이블 2개를 좌우로 붙여 각 로봇을 자기 마운트에 세움(권장) / "
                         "'single'=1개 / 'proc'=절차적 테이블(크기 자유) / 'none'=생략")
parser.add_argument("--table-dx", type=float, default=0.55,
                    help="로봇 베이스(x=0) 기준 테이블 중심의 x 오프셋. "
                         "원본 단일팔 태스크와 같은 값이어야 마운트가 로봇 밑에 온다")
parser.add_argument("--table-rot", default="0.707,0.0,0.0,0.707",
                    help="테이블 쿼터니언 w,x,y,z")
parser.add_argument("--table-size", default="1.4,1.8,0.05",
                    help="proc 모드 상판 크기 x,y,두께 [m]")
parser.add_argument("--table-height", type=float, default=1.05,
                    help="proc 모드: 지면부터 상판 윗면까지. 윗면이 z=0 이 되게 다리를 만든다")
parser.add_argument("--table-usd", default="SeattleLabTable",
                    help="dual/single 모드에서 쓸 테이블. "
                         "SeattleLabTable(마운트 판 있음) / ThorlabsTable(평판) / 절대경로")
parser.add_argument("--table-mirror", type=int, default=2,
                    help="dual 모드 반전 방식. 0=끔(다리 하나 돌출) / "
                         "1=180도 회전(x 도 뒤집혀 어긋남) / 2=거울 반사(권장, y 만 반전)")
parser.add_argument("--mirror-side", choices=["left", "right", "both"], default="left",
                    help="어느 테이블을 반전할지. left=두 판이 안쪽으로 모여 겹침(권장) / "
                         "right=바깥으로 벌어져 가운데서 맞닿음")
parser.add_argument("--cube-usd-dir", default="",
                    help="큐브 USD 디렉터리. 비면 ISAAC_NUCLEUS_DIR/Props/Blocks. "
                         "없으면 절차적 큐브로 대체한다")
parser.add_argument("--shot", default="dual_franka.png",
                    help="마지막 프레임 png 경로. 빈 문자열이면 카메라 생략")
parser.add_argument("--video", default="dual_franka.mp4",
                    help="구동 영상 mp4 경로. 빈 문자열이면 저장 안 함")
parser.add_argument("--video-every", type=int, default=4,
                    help="몇 물리 스텝마다 한 프레임 캡처할지 (120Hz / 4 = 30fps)")
parser.add_argument("--video-fps", type=int, default=30)
parser.add_argument("--cam-eye", default="1.9,0.0,1.3")
parser.add_argument("--cam-target", default="0.45,0.0,0.15")
parser.add_argument("--views", default="iso,top,front,side",
                    help="캡처할 시점들. 각각 <shot 파일명>_<view>.png 로 저장된다. "
                         "빈 문자열이면 --cam-eye/--cam-target 한 장만")
parser.add_argument("--diagram", default="",
                    help="위에서 본 배치 도해 png 경로. 리치 원, 테이블, 큐브를 그린다")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.shot:
    args.enable_cameras = True

if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ------------------------------------------------------------------ 본 임포트
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402

# IK 추종이 좋은 강성 PD 판. Franka 스택 태스크가 쓰는 것과 동일하다.
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402


def _vec(s):
    return tuple(float(x) for x in s.split(","))


CUBE_PROPS = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


def usd_ok(path):
    """로컬 마운트면 존재 여부를 확인한다. 원격(omniverse://, http)이면 통과."""
    if path.startswith(("omniverse://", "http://", "https://")):
        return True
    return os.path.exists(path)


def make_cube(prim, pos, rgb, usd_dir):
    """큐브 하나. USD 가 있으면 그것을, 없으면 절차적 박스를 쓴다."""
    if usd_dir:
        name = {"blue": "blue_block.usd", "red": "red_block.usd",
                "green": "green_block.usd"}[rgb]
        p = f"{usd_dir}/{name}"
        if usd_ok(p):
            spawn = sim_utils.UsdFileCfg(usd_path=p, scale=(1.0, 1.0, 1.0),
                                         rigid_props=CUBE_PROPS)
            return RigidObjectCfg(
                prim_path=prim,
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos,
                                                          rot=(1.0, 0.0, 0.0, 0.0)),
                spawn=spawn,
            )
    # 폴백: 4cm 정육면체를 직접 만든다 (에셋 의존 없음)
    color = {"blue": (0.1, 0.2, 0.8), "red": (0.8, 0.1, 0.1),
             "green": (0.1, 0.7, 0.2)}[rgb]
    return RigidObjectCfg(
        prim_path=prim,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=CUBE_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


def build_scene_cfg(a):
    """양팔 씬. Franka 스택 태스크의 에셋을 그대로 쓰되, 없으면 대체한다."""
    sep = a.base_sep / 2.0
    # parallel: 둘 다 +x 를 향해 y 축으로 벌린다
    # opposed : 오른쪽 팔을 180도 돌려 마주보게 한다
    rot_r = (1.0, 0.0, 0.0, 0.0) if a.layout == "parallel" else (0.0, 0.0, 0.0, 1.0)
    pos_r = (0.0, -sep, 0.0) if a.layout == "parallel" else (0.9, -sep, 0.0)

    @configclass
    class DualFrankaSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0,
                                         color=(0.75, 0.75, 0.75)),
        )
        robot_l = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotL",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, sep, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )
        robot_r = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotR",
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=pos_r, rot=rot_r
            ),
        )

    cfg = DualFrankaSceneCfg(num_envs=a.num_envs, env_spacing=3.0)

    # ---- 테이블 ----
    # 스택 태스크 규약: 작업면이 env 프레임 z=0, 지면은 -1.05 (테이블 높이)
    # 테이블이 없으면 지면 자체를 작업면으로 올려야 큐브가 안 떨어진다.
    def table_usd_path():
        u = a.table_usd
        if u.startswith("/") or u.startswith("omniverse://"):
            return u
        return f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/{u}/table_instanceable.usd"

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)

    def usd_table(name, y, mirror=0):
        """테이블 한 장.

        테이블 원점이 판 중심이 아니라 마운트 쪽에 치우쳐 있다 (실측 오프셋
        (-0.156, +0.370)). 그래서 두 장을 그냥 나란히 놓으면 판 몸통이 한쪽으로
        쏠려 먼 쪽 다리가 홀로 튀어나온다.

        mirror=1  z축 180도 회전.  x, y 오프셋이 둘 다 반전된다 -> x 가 어긋남
        mirror=2  거울 반사 (권장). y 만 반전.
                  월드 y 반사 M 에 대해  M, R = R', S  가 되도록
                  R' = Rz(-90도),  S = diag(1,-1,1)  로 분해한다.
        """
        rot = _vec(a.table_rot)
        dx = a.table_dx
        scale = (1.0, 1.0, 1.0)
        if mirror == 1:
            rot = quat_mul((0.0, 0.0, 0.0, 1.0), rot)   # 180deg about z
            dx = -a.table_dx
        elif mirror == 2:
            w, x, yq, z = rot
            rot = (w, -x, yq, -z)        # Rz(theta) -> Rz(-theta) (= M, R, M 켤레)
            scale = (1.0, -1.0, 1.0)     # local y 반사
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(dx, y, 0.0), rot=rot),
            spawn=sim_utils.UsdFileCfg(usd_path=table_usd_path(), scale=scale),
        )

    has_table = True
    if a.table == "none":
        has_table = False
        print("[asset] table 생략")

    elif a.table == "proc":
        # 상판 윗면이 z=0 이 되도록 슬랩 중심을 -t/2 에 둔다. 다리 4개로 지면까지.
        sx, sy, th = _vec(a.table_size)
        cfg.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(a.table_dx, 0.0, -th / 2)),
            spawn=sim_utils.CuboidCfg(
                size=(sx, sy, th),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.35, 0.35, 0.38), roughness=0.7),
            ),
        )
        leg = 0.06
        hz = (a.table_height - th) / 2.0
        for i, (lx, ly) in enumerate([(sx / 2 - leg, sy / 2 - leg),
                                      (sx / 2 - leg, -sy / 2 + leg),
                                      (-sx / 2 + leg, sy / 2 - leg),
                                      (-sx / 2 + leg, -sy / 2 + leg)]):
            setattr(cfg, f"table_leg_{i}", AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/TableLeg" + str(i),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(a.table_dx + lx, ly, -th - hz)),
                spawn=sim_utils.CuboidCfg(
                    size=(leg, leg, a.table_height - th),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.25, 0.25, 0.28)),
                ),
            ))
        print(f"[asset] table = 절차적 {sx}x{sy}x{th} @ x={a.table_dx}, 높이 {a.table_height}")

    else:
        tp = table_usd_path()
        if not usd_ok(tp):
            print(f"[asset] table USD 없음 -> 생략 ({tp})")
            has_table = False
        elif a.table == "dual":
            # 로봇 1대 + 테이블 1개 짝을 그대로 두 벌. 각 로봇이 자기 마운트 위에 선다.
            ml = a.table_mirror if a.mirror_side in ("left", "both") else 0
            mr = a.table_mirror if a.mirror_side in ("right", "both") else 0
            cfg.table_l = usd_table("TableL", +sep, mirror=ml)
            cfg.table_r = usd_table("TableR", -sep, mirror=mr)
            mm = {0: "off", 1: "rot180", 2: "reflect(scale y=-1)"}
            print(f"[asset] table = {tp} x2  (y=+/-{sep:.3f}, x={a.table_dx})  "
                  f"mirror: L={mm.get(ml, '?')} R={mm.get(mr, '?')} "
                  f"[side={a.mirror_side}, mode={a.table_mirror}]")
        else:  # single
            cfg.table = usd_table("Table", 0.0)
            print(f"[asset] table = {tp} x1")

    cfg.ground.init_state = AssetBaseCfg.InitialStateCfg(
        pos=(0.0, 0.0, -1.05 if has_table else 0.0)
    )
    print(f"[asset] ground z = {cfg.ground.init_state.pos[2]}")

    # ---- 두 팔이 함께 다룰 물체. 중앙선 위에 둔다 ----
    cube_dir = a.cube_usd_dir or f"{ISAAC_NUCLEUS_DIR}/Props/Blocks"
    cfg.cube_1 = make_cube("{ENV_REGEX_NS}/Cube_1", (0.45, 0.00, 0.0203), "blue", cube_dir)
    cfg.cube_2 = make_cube("{ENV_REGEX_NS}/Cube_2", (0.45, 0.18, 0.0203), "red", cube_dir)
    cfg.cube_3 = make_cube("{ENV_REGEX_NS}/Cube_3", (0.45, -0.18, 0.0203), "green", cube_dir)
    print(f"[asset] cube dir = {cube_dir} "
          f"({'USD' if usd_ok(cube_dir + '/blue_block.usd') else '절차적 큐브로 대체'})")

    if a.shot or a.video:
        cfg.scene_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene_cam",
            update_period=0.0,
            height=720, width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.05, 30.0),
            ),
        )
    return cfg


def grab_rgb(cam):
    """카메라 센서에서 uint8 RGB 한 장."""
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def save_video(frames, path, fps):
    """mp4 저장. 코덱이 환경마다 달라 폴백 사슬을 둔다."""
    if not frames:
        print(f"[vid] 프레임 0개 - 저장 생략: {path}")
        return
    try:
        import imageio
    except ImportError:
        print("[vid] imageio 없음 -> /isaac-sim/python.sh -m pip install imageio imageio-ffmpeg")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_err = None
    for codec in ("libx264", "h264", "mpeg4"):
        try:
            imageio.mimsave(path, frames, fps=fps, codec=codec)
            print(f"[vid] {path}  ({len(frames)} frames, {fps}fps, codec={codec})")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    stem = os.path.splitext(path)[0]
    os.makedirs(stem, exist_ok=True)
    for i, f in enumerate(frames):
        imageio.imwrite(os.path.join(stem, f"{i:05d}.png"), f)
    print(f"[vid] mp4 실패({last_err}) -> PNG 시퀀스: {stem}/")


def draw_diagram(path, scene, a, tables, reach=0.855):
    """위에서 본 배치 도해. (1)(2)(4) 를 한 장으로 보여준다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
    except ImportError:
        print("[diagram] matplotlib 없음 -> /isaac-sim/python.sh -m pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(9, 9))
    # 테이블 (x-y 평면 투영)
    for nm, lo, hi in tables:
        ax.add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                               fill=True, alpha=0.18, ec="k", lw=1.2, label=nm))
        ax.text((lo[0] + hi[0]) / 2, hi[1] - 0.06, nm, ha="center", fontsize=9)

    # 리치 원 + 베이스
    for nm, key, col in (("L", "robot_l", "tab:blue"), ("R", "robot_r", "tab:red")):
        b = scene[key].data.root_pos_w[0].cpu().numpy()
        ax.add_patch(Circle((b[0], b[1]), reach, fill=False, ec=col, lw=2.0, ls="--"))
        ax.plot(b[0], b[1], "o", color=col, ms=11)
        ax.text(b[0], b[1] - 0.09, f"base {nm}", color=col, ha="center", fontsize=10)

    # 큐브 + 도달 여부
    bl = scene["robot_l"].data.root_pos_w[0].cpu().numpy()
    br = scene["robot_r"].data.root_pos_w[0].cpu().numpy()
    for nm, col in (("cube_1", "tab:blue"), ("cube_2", "tab:red"), ("cube_3", "tab:green")):
        p = scene[nm].data.root_pos_w[0].cpu().numpy()
        okl = np.linalg.norm(p - bl) < reach
        okr = np.linalg.norm(p - br) < reach
        ax.plot(p[0], p[1], "s", color=col, ms=13, mec="k")
        ax.text(p[0] + 0.05, p[1], f"{nm}\nL{'O' if okl else 'X'} R{'O' if okr else 'X'}",
                fontsize=8, va="center")

    d = float(np.linalg.norm(bl[:2] - br[:2]))
    ov = 2 * reach - d
    ax.set_title(f"base_sep={d:.3f}m  reach={reach}m  공유폭={ov:.3f}m  "
                 f"({a.layout}, table={a.table})")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlim(-1.2, 2.0); ax.set_ylim(-1.6, 1.6)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[diagram] {path}")


VIEWS = {
    # 이름: (eye, target)
    "iso":   ((1.9, 1.3, 1.2), (0.45, 0.0, 0.05)),
    "top":   ((0.45, 0.0, 2.6), (0.45, 0.0, 0.0)),
    "front": ((2.2, 0.0, 0.55), (0.30, 0.0, 0.15)),
    "side":  ((0.45, 2.3, 0.9), (0.45, 0.0, 0.15)),
}


def capture_views(sim, scene, cam, names, base_path):
    """여러 시점에서 한 장씩. 카메라를 옮긴 뒤 렌더가 반영되도록 몇 스텝 돌린다."""
    import imageio
    stem, ext = os.path.splitext(base_path)
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    for nm in names:
        if nm not in VIEWS:
            print(f"[view] 알 수 없는 시점: {nm}")
            continue
        eye, tgt = VIEWS[nm]
        cam.set_world_poses_from_view(
            eyes=torch.tensor([eye], device=sim.device),
            targets=torch.tensor([tgt], device=sim.device),
        )
        for _ in range(3):          # 렌더 반영 대기
            sim.step()
            scene.update(sim.get_physics_dt())
        out = f"{stem}_{nm}{ext or '.png'}"
        imageio.imwrite(out, grab_rgb(cam))
        print(f"[view] {nm:5s} -> {out}")


def report_table_bbox():
    """테이블 프림의 월드 bbox 를 재서 배치 가능 범위를 알려준다."""
    try:
        import isaacsim.core.utils.stage as stage_utils
        from pxr import Usd, UsdGeom
    except ImportError:
        try:
            import omni.isaac.core.utils.stage as stage_utils  # 구버전
            from pxr import Usd, UsdGeom
        except ImportError:
            return None
    stage = stage_utils.get_current_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    found = []
    for nm in ("Table", "TableL", "TableR"):
        p = stage.GetPrimAtPath(f"/World/envs/env_0/{nm}")
        if p and p.IsValid():
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            found.append((nm, r.GetMin(), r.GetMax()))
    if not found:
        return None, []

    print("\n===== 테이블 =====")
    for nm, lo, hi in found:
        print(f"{nm:6s} x[{lo[0]:+.3f},{hi[0]:+.3f}] {hi[0]-lo[0]:.3f}m  "
              f"y[{lo[1]:+.3f},{hi[1]:+.3f}] {hi[1]-lo[1]:.3f}m  "
              f"상판 z={hi[2]:+.3f}")
    if len(found) == 2:
        (_, l1, h1), (_, l2, h2) = found[0], found[1]
        gap = max(l1[1], l2[1]) - min(h1[1], h2[1])
        state = "겹침" if gap < -1e-4 else ("딱 붙음" if gap < 0.01 else f"틈 {gap:.3f}m")
        print(f"두 테이블 y 방향: {state}  "
              f"(딱 붙이려면 --base-sep {(h1[1]-l1[1]):.3f})")
    # 전체를 감싸는 bbox 로 통합 반환
    lo = [min(f[1][i] for f in found) for i in range(3)]
    hi = [max(f[2][i] for f in found) for i in range(3)]
    return (lo, hi), found


def report_workspace(scene, a, tbox=None):
    """두 팔의 작업 공간이 겹치는지 수치로 확인한다."""
    FRANKA_REACH = 0.855  # panda_link0 기준 최대 도달 반경 [m]
    bl = scene["robot_l"].data.root_pos_w[0].cpu().numpy()
    br = scene["robot_r"].data.root_pos_w[0].cpu().numpy()
    d = float(np.linalg.norm(bl[:2] - br[:2]))

    print("\n===== 작업 공간 =====")
    print(f"베이스 L  {bl[:3].round(3)}")
    print(f"베이스 R  {br[:3].round(3)}")
    print(f"베이스 간 거리 {d:.3f} m  (Franka 리치 {FRANKA_REACH} m)")
    if d >= 2 * FRANKA_REACH:
        print("  X 두 리치 구가 만나지 않는다 -> 협응 태스크 불가. --base-sep 를 줄일 것")
    else:
        # 두 구의 교차 렌즈 폭 (중심선 상에서 겹치는 구간)
        overlap = 2 * FRANKA_REACH - d
        print(f"  OK 공유 작업공간 있음. 중심선 겹침 폭 ~ {overlap:.3f} m")

    for name in ("cube_1", "cube_2", "cube_3"):
        p = scene[name].data.root_pos_w[0].cpu().numpy()
        dl = float(np.linalg.norm(p[:3] - bl[:3]))
        dr = float(np.linalg.norm(p[:3] - br[:3]))
        okl = "O" if dl < FRANKA_REACH else "X"
        okr = "O" if dr < FRANKA_REACH else "X"
        print(f"  {name}: pos={p[:3].round(3)}  L={dl:.3f}({okl})  R={dr:.3f}({okr})")

    # 베이스가 테이블 위에 실제로 올라가 있는지
    if tbox is not None:
        lo, hi = tbox
        for nm, b in (("L", bl), ("R", br)):
            on = (lo[0] <= b[0] <= hi[0]) and (lo[1] <= b[1] <= hi[1])
            print(f"  베이스 {nm} 테이블 위: {'O' if on else 'X  <- 모서리에 걸침'}"
                  f"  (x={b[0]:+.3f} y={b[1]:+.3f})")


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0,
                                                    device=args.device))
    scene = InteractiveScene(build_scene_cfg(args))
    sim.reset()
    print(f"[scene] prims: {sorted(scene.keys())}")

    rl, rr = scene["robot_l"], scene["robot_r"]
    print(f"[robot] L dof={rl.num_joints} bodies={rl.num_bodies} "
          f"fixed_base={rl.is_fixed_base}")
    print(f"[robot] R dof={rr.num_joints} bodies={rr.num_bodies} "
          f"fixed_base={rr.is_fixed_base}")

    cam = None
    if args.shot or args.video:
        cam: Camera = scene["scene_cam"]
        cam.set_world_poses_from_view(
            eyes=torch.tensor([_vec(args.cam_eye)], device=sim.device),
            targets=torch.tensor([_vec(args.cam_target)], device=sim.device),
        )

    # 기본 자세를 목표로 유지시켜 물리 안정성만 본다 (정책 없음)
    for r in (rl, rr):
        q = r.data.default_joint_pos.clone()
        r.write_joint_state_to_sim(q, torch.zeros_like(q))
        r.set_joint_position_target(q)
        r.write_data_to_sim()

    q0_l = rl.data.joint_pos.clone()
    q0_r = rr.data.joint_pos.clone()

    frames = []
    for i in range(args.steps):
        rl.set_joint_position_target(rl.data.default_joint_pos)
        rr.set_joint_position_target(rr.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

        if args.video and cam is not None and (i % max(args.video_every, 1) == 0):
            frames.append(grab_rgb(cam))

        if (i + 1) % 100 == 0:
            dl = float((rl.data.joint_pos - q0_l).abs().max())
            dr = float((rr.data.joint_pos - q0_r).abs().max())
            zl = float(rl.data.root_pos_w[0, 2])
            zr = float(rr.data.root_pos_w[0, 2])
            print(f"[step {i+1:4d}] 관절 최대 편차 L={dl:.4f} R={dr:.4f} rad | "
                  f"베이스 z L={zl:.4f} R={zr:.4f}")

    tbox, tables = report_table_bbox()
    report_workspace(scene, args, tbox)

    if args.diagram:
        draw_diagram(args.diagram, scene, args, tables)

    if args.shot and cam is not None:
        try:
            import imageio  # noqa: F401
            views = [v.strip() for v in args.views.split(",") if v.strip()]
            if views:
                print()
                capture_views(sim, scene, cam, views, args.shot)
            else:
                import imageio
                os.makedirs(os.path.dirname(args.shot) or ".", exist_ok=True)
                imageio.imwrite(args.shot, grab_rgb(cam))
                print(f"\n[shot] {args.shot}")
        except ImportError:
            np.save(args.shot + ".npy", grab_rgb(cam))
            print(f"\n[shot] imageio 없음 -> {args.shot}.npy 로 저장")

    if args.video and cam is not None:
        save_video(frames, args.video, args.video_fps)

    print("\n[done] 물리 안정 + 두 팔 로드 확인 완료")


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
        # simulation_app.close() 는 헤드리스에서 반환하지 않는 경우가 있다.
        # 호출하면 그 뒤 줄에 도달하지 못해 프로세스가 걸린 채 남는다.
        # 결과는 이미 디스크에 썼으므로 호출하지 않고 바로 끊는다.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
