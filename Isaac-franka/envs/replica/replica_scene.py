#!/usr/bin/env python
# [ver] replica_scene.py 2026-08-25-r2  (ascii-only console/comments)
# r2: --floor-z (상판 z=0 자산용 바닥 높이), 조명 z 하한, room_bbox 가드
r"""Replica 스캔 방 + 양팔 Franka 만으로 이루어진 씬 (3번째 환경).

office/ 와 완전 격리다 (import 하지 않는다). 스캔 방(가구 포함)이 곧
배경이고, 우리가 얹는 것은 로봇 2대와 받침대뿐이다.

좌표계: 스캔 방이 세계의 기준이다. 변환기(replica_to_usd.py)가 바닥을
z=0, xy bbox 중심을 원점으로 재원점해 두었으므로 방은 항상 원점에 놓고,
로봇 리그를 --base-x/--base-y/--base-yaw 로 방 안의 빈 자리에 배치한다.

주의 (office 에서 확정된 함정 재적용 대상):
  - 로봇 베이스가 회전(yaw != 0 또는 기본 z180)돼 있으므로, 나중에 IK
    실행기를 이식할 때 자코비안 루트 프레임 회전을 반드시 확인할 것
    (docs/office_env_pipeline.md 시행착오 (1))
"""

import math
import os

# ---- 상수 ---------------------------------------------------------------
PEDESTAL_D = 0.36  # 받침대 깊이 (x)
PEDESTAL_MARGIN = 0.16  # 받침대가 두 베이스 바깥으로 나가는 폭
PEDESTAL_TH = 0.04  # 받침대 상판 두께
PEDESTAL_COLOR = (0.30, 0.30, 0.32)
LEG_COLOR = (0.25, 0.25, 0.26)


# ---- 헬퍼 ---------------------------------------------------------------
def vec(s):
    return tuple(float(x) for x in s.split(","))


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_axis(axis, deg):
    """x/y/z 축 회전 쿼터니언 (w,x,y,z)."""
    h = math.radians(deg) / 2.0
    c, s = math.cos(h), math.sin(h)
    return {
        "x": (c, s, 0.0, 0.0),
        "y": (c, 0.0, s, 0.0),
        "z": (c, 0.0, 0.0, s),
    }[axis]


def rot2d(x, y, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return (x * c - y * s, x * s + y * c)


def add_scene_args(p):
    g = p.add_argument_group("replica scene")
    g.add_argument(
        "--bg-usd",
        default="/root/project/datasets/replica/office_0.usd",
        help="scanned-room USD from replica_to_usd.py (floor z=0, xy centered)",
    )
    g.add_argument(
        "--ground",
        type=int,
        default=0,
        help="1: add a grid ground plane slightly below the scan floor "
        "(safety net if objects ever fall through scan holes)",
    )
    g.add_argument(
        "--light",
        type=float,
        default=10000.0,
        help="intensity of each interior ceiling light (5 disk lights under "
        "the ceiling). The closed scan mesh blocks the dome light, so the "
        "room needs its own lights",
    )
    g.add_argument(
        "--dome",
        type=float,
        default=800.0,
        help="dome (ambient) intensity. Only fills through door/window holes",
    )

    g = p.add_argument_group("robots")
    g.add_argument("--robots", type=int, default=1, help="0: scan only")
    g.add_argument("--base-x", type=float, default=0.0, help="rig center x in the room")
    g.add_argument("--base-y", type=float, default=0.0, help="rig center y in the room")
    g.add_argument(
        "--base-yaw",
        type=float,
        default=0.0,
        help="rig yaw [deg]. 0 = robots face -x (same as the office env)",
    )
    g.add_argument(
        "--base-sep", type=float, default=0.60, help="distance between bases"
    )
    g.add_argument(
        "--base-z",
        type=float,
        default=0.65,
        help="ABSOLUTE world z of the pedestal top (= robot base). "
        "note: not relative to --floor-z -- adjust both together",
    )
    g.add_argument(
        "--floor-z",
        type=float,
        default=0.0,
        help="height of the floor (= pedestal leg bottom, ground plane). "
        "use -0.70 with desk assets whose TOP is z=0 (gsrecon take6)",
    )

    g = p.add_argument_group("render")
    g.add_argument("--num-envs", type=int, default=1)
    g.add_argument("--cam-w", type=int, default=1280)
    g.add_argument("--cam-h", type=int, default=720)
    g.add_argument(
        "--cam-fov",
        type=float,
        default=65.0,
        help="scene camera horizontal FOV [deg]. Isaac default aperture is "
        "about 47 which is too tight for a whole room",
    )


def room_bbox(usd_path):
    """변환기가 authoring 한 /Bg/Geom extent 를 읽는다 (절반폭 hx,hy 와 높이).

    pxr 를 쓰므로 AppLauncher 기동 후에만 호출할 것.
    """
    from pxr import Usd

    if not os.path.exists(usd_path):
        raise SystemExit(f"[scene] ERROR: bg usd not found: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise SystemExit(f"[scene] ERROR: cannot open bg usd: {usd_path}")
    prim = stage.GetPrimAtPath("/Bg/Geom")
    ext = prim.GetAttribute("extent").Get() if prim else None
    if not ext or len(ext) != 2:
        print("[scene] WARNING: no extent on /Bg/Geom, fallback 2.2x2.5x3.0")
        return 2.2, 2.5, 3.0
    lo, hi = ext
    return float(hi[0]), float(hi[1]), float(hi[2])


# ---- 씬 조립 ------------------------------------------------------------
def build(a, with_camera=True):
    """씬 cfg 를 만든다. AppLauncher 기동 이후에만 호출할 것."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    @configclass
    class ReplicaSceneCfg(InteractiveSceneCfg):
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=a.dome, color=(0.78, 0.78, 0.80)),
        )

    cfg = ReplicaSceneCfg(num_envs=a.num_envs, env_spacing=8.0)

    # ---- 스캔 방 (가구 포함 융합 메시, 시각 전용) ------------------------
    # 변환기가 정렬을 좌표에 구웠으므로 원점에 그대로 둔다
    cfg.bg_scan = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BgScan",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(usd_path=a.bg_usd),
    )

    # ---- 실내 조명 -------------------------------------------------------
    # 닫힌 스캔 메시(벽+천장)가 돔라이트를 막아 방 안이 어둡다 (실측).
    # 천장 바로 아래에 아래를 비추는 디스크 라이트 5개를 심는다
    hx, hy, hh = room_bbox(a.bg_usd)
    # 조명 z: 방 스캔은 천장 바로 아래(hh-0.15)가 맞지만, 책상 자산처럼
    # 낮은 bbox(hh~0.4)에서는 상판 아래에 심기므로 하한을 둔다
    lz = max(hh - 0.15, 1.2)
    for i, (lx, ly) in enumerate(
        [
            (0.0, 0.0),
            (0.45 * hx, 0.45 * hy),
            (0.45 * hx, -0.45 * hy),
            (-0.45 * hx, 0.45 * hy),
            (-0.45 * hx, -0.45 * hy),
        ]
    ):
        setattr(
            cfg,
            f"room_light_{i}",
            AssetBaseCfg(
                prim_path=f"/World/room_light_{i}",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(lx, ly, lz)),
                spawn=sim_utils.DiskLightCfg(intensity=a.light, radius=0.15),
            ),
        )

    # 스캔 바닥 밑의 안전판 (기본 꺼짐 -- 격자가 방 밖에서 보이면 지저분)
    if a.ground:
        cfg.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, a.floor_z - 0.01)),
        )

    def static_box(name, pos, size, color, rough=0.6, rot=(1.0, 0.0, 0.0, 0.0)):
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, roughness=rough
                ),
            ),
        )

    # ---- 로봇 (받침대 위 양팔, office 와 동일 리그) ----------------------
    if a.robots:
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        franka_spawn = FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(
            activate_contact_sensors=True,
            articulation_props=FRANKA_PANDA_HIGH_PD_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=False
            ),
        )
        # yaw 0 = -x 를 향한다 (office 규약). 리그 전체가 base-yaw 로 돈다
        q_rig = quat_axis("z", a.base_yaw)
        face = quat_mul(q_rig, (0.0, 0.0, 0.0, 1.0))
        sep = a.base_sep / 2.0

        # 받침대: 상판 1장 + 다리 4개 (office 의 이어진 받침대와 동일 조형)
        ped_w = a.base_sep + 2 * PEDESTAL_MARGIN
        cfg.pedestal_top = static_box(
            "PedestalTop",
            (a.base_x, a.base_y, a.base_z - PEDESTAL_TH / 2),
            (PEDESTAL_D, ped_w, PEDESTAL_TH),
            PEDESTAL_COLOR,
            rough=0.55,
            rot=q_rig,
        )
        leg_h = a.base_z - PEDESTAL_TH - a.floor_z
        leg, inset = 0.06, 0.05
        for i, (dx, dy) in enumerate(
            [
                (PEDESTAL_D / 2 - inset, ped_w / 2 - inset),
                (PEDESTAL_D / 2 - inset, -ped_w / 2 + inset),
                (-PEDESTAL_D / 2 + inset, ped_w / 2 - inset),
                (-PEDESTAL_D / 2 + inset, -ped_w / 2 + inset),
            ]
        ):
            wx, wy = rot2d(dx, dy, a.base_yaw)
            setattr(
                cfg,
                f"pedestal_leg_{i}",
                static_box(
                    f"PedestalLeg{i}",
                    (a.base_x + wx, a.base_y + wy, a.floor_z + leg_h / 2),
                    (leg, leg, leg_h),
                    LEG_COLOR,
                ),
            )

        # 손가락 강성 6e3 = office 실측 절충값 (조임력 ~78N)
        grip_act = FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_hand"].replace(
            stiffness=6.0e3, damping=2.0e2
        )
        actuators = dict(FRANKA_PANDA_HIGH_PD_CFG.actuators)
        actuators["panda_hand"] = grip_act
        for tag, sgn in (("l", -1.0), ("r", 1.0)):
            ox, oy = rot2d(0.0, sgn * sep, a.base_yaw)
            setattr(
                cfg,
                f"robot_{tag}",
                FRANKA_PANDA_HIGH_PD_CFG.replace(
                    prim_path="{ENV_REGEX_NS}/Robot" + tag.upper(),
                    spawn=franka_spawn,
                    actuators=actuators,
                    init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                        pos=(a.base_x + ox, a.base_y + oy, a.base_z), rot=face
                    ),
                ),
            )

        from isaaclab.sensors import ContactSensorCfg

        cfg.contact_l = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/RobotL/panda_link[2-7]",
            update_period=0.0,
            history_length=6,
        )
        cfg.contact_r = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/RobotR/panda_link[2-7]",
            update_period=0.0,
            history_length=6,
        )

    # ---- 카메라 ---------------------------------------------------------
    if with_camera:
        cfg.scene_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene_cam",
            update_period=0.0,
            height=a.cam_h,
            width=a.cam_w,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                # aperture = 2 * focal * tan(fov/2)
                horizontal_aperture=2.0
                * 24.0
                * math.tan(math.radians(a.cam_fov) / 2.0),
                clipping_range=(0.05, 30.0),
            ),
        )

    print(
        f"[scene] bg={os.path.basename(a.bg_usd)} room=({hx:.2f},{hy:.2f},{hh:.2f}) "
        f"robots={bool(a.robots)} base=({a.base_x:.2f},{a.base_y:.2f},{a.base_z:.2f}) "
        f"yaw={a.base_yaw:.0f} fov={a.cam_fov:.0f} light={a.light:.0f}"
    )
    return cfg
