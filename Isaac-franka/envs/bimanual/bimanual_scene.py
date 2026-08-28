# [ver] bimanual_scene.py 2026-08-05-r4  (ascii-only console/comments)
"""
양팔 Franka 워크셀 -- 씬 정의.

Isaac Lab 2.2.1 에는 양팔 로봇 에셋이 없다 (양팔은 GR1T2 휴머노이드뿐).
대신 Franka 를 두 대 놓아 양팔 워크셀을 만든다.
robosuite 의 TwoArm 환경이 Panda 2대로 하는 것과 같은 구성이다.

    RobotL  (0, +base_sep/2, 0)     둘 다 +x 를 향함
    RobotR  (0, -base_sep/2, 0)
    작업면  z = 0        지면 z = -1.05 (테이블 높이)
    큐브    4cm, 중심 z = 0.0203     상판 위에 놓인 상태

주의
    이 파일의 최상단은 isaaclab 를 import 하지 않는다.
    AppLauncher 가 뜨기 전에는 omni.client 가 없어 import 자체가 실패한다.
    실행 스크립트가 인자 정의(add_scene_args)를 앱 기동 전에 써야 하므로,
    isaaclab 심볼은 전부 build() 안에서 지연 import 한다.

확정 프리셋과 실측 수치는 PRESETS.md 참조.
"""

import math
import os

# constants
CUBE_SIZE = 0.04  # 큐브 한 변 [m]
CUBE_HALF = 0.0203  # 상판 위에 놓였을 때의 중심 높이 (0.3mm 여유 포함)
MAX_CUBES = 6  # 씬에 미리 만들어 두는 개수

# cube_1 이 항상 탑의 바닥. 앞 4개는 Isaac 기본 블록 USD, 5,6번은 절차적 큐브.
CUBE_COLORS = ["blue", "red", "green", "yellow", "orange", "purple"]
_RGB = {
    "blue": (0.10, 0.20, 0.80),
    "red": (0.80, 0.10, 0.10),
    "green": (0.10, 0.70, 0.20),
    "yellow": (0.85, 0.75, 0.10),
    "orange": (0.90, 0.45, 0.05),
    "purple": (0.55, 0.15, 0.75),
}
_USD_NAME = {
    "blue": "blue_block.usd",
    "red": "red_block.usd",
    "green": "green_block.usd",
    "yellow": "yellow_block.usd",
}

# 안 쓰는 큐브를 치워두는 자리 (테이블 밖, 바닥으로 떨어져 화면에서 사라진다)
PARK = (2.5, 2.5, 0.5)

# 테이블 원점에서 상판 몸통 중심까지의 오프셋 (SeattleLabTable, 실측).
# 원점이 마운트 쪽에 치우쳐 있어 두 장을 그냥 나란히 놓으면 한쪽 다리가 튀어나온다.
# z 회전으로는 교정할 수 없고 거울 반사(scale y=-1)만 가능하다. PRESETS.md 참조.
TABLE_BODY_OFFSET = (-0.156, +0.370)


def cube_home(i):
    """i 번째(1-base) 큐브의 기본 위치. --randomize 0 일 때만 쓴다."""
    if i == 1:
        return (0.45, 0.00, CUBE_HALF)
    k = i - 2
    return (0.42 + 0.08 * (k // 2), 0.18 * (1 if k % 2 == 0 else -1), CUBE_HALF)


CUBE_HOME = {f"cube_{i}": cube_home(i) for i in range(1, MAX_CUBES + 1)}


# argparse
def add_scene_args(p):
    """씬 관련 인자. 앱 기동 전에 호출해도 안전하다 (isaaclab 불필요)."""
    g = p.add_argument_group("scene")
    g.add_argument(
        "--layout",
        choices=["parallel", "opposed"],
        default="parallel",
        help="parallel: side by side facing +x / opposed: facing each other",
    )
    g.add_argument(
        "--base-sep",
        type=float,
        default=0.9,
        help="distance between the two robot bases [m]; Franka reach is "
        "0.855 so beyond 1.71 the workspaces no longer overlap",
    )
    g.add_argument(
        "--table",
        default="dual",
        help="dual: two Isaac tables (one mount per robot) / single / "
        "proc: procedural table (free size) / none",
    )
    g.add_argument(
        "--table-usd",
        default="SeattleLabTable",
        help="SeattleLabTable (has mount plate) / ThorlabsTable (flat) / abs path",
    )
    g.add_argument(
        "--table-dx",
        type=float,
        default=0.55,
        help="table origin x offset from robot base; must match the "
        "single-arm task so the mount sits under the robot",
    )
    g.add_argument(
        "--table-rot",
        default="0.707,0.0,0.0,0.707",
        help="table quaternion w,x,y,z",
    )
    g.add_argument(
        "--table-size",
        default="1.4,1.8,0.05",
        help="proc mode: top plate x,y,thickness [m]",
    )
    g.add_argument(
        "--table-height",
        type=float,
        default=1.05,
        help="proc mode: table height [m]",
    )
    g.add_argument(
        "--table-mirror",
        type=int,
        default=2,
        help="0: off (one leg sticks out) / 1: rot180 (x flips too) / "
        "2: mirror reflection (recommended)",
    )
    g.add_argument(
        "--mirror-side",
        choices=["left", "right", "both"],
        default="right",
        help="which table to mirror (verified on render: right)",
    )
    g.add_argument("--num-envs", type=int, default=1)
    g.add_argument("--cam-w", type=int, default=1280)
    g.add_argument("--cam-h", type=int, default=720)
    g.add_argument(
        "--wrist-fov",
        type=float,
        default=75.0,
        help="wrist camera horizontal FOV [deg]. 75 matches the LIBERO / "
        "robosuite eye_in_hand look (wide, fingers at the frame edges); "
        "Isaac Lab default is about 47 which is much tighter",
    )
    g.add_argument(
        "--wrist-pos",
        default="0.13,0.0,-0.15",
        help="wrist camera offset in the panda_hand frame [m] "
        "(default = Isaac Lab official Franka stack task)",
    )
    g.add_argument(
        "--wrist-rot",
        default="-0.70614,0.03701,0.03701,-0.70614",
        help="wrist camera quaternion w,x,y,z, ros convention "
        "(default = Isaac Lab official Franka stack task)",
    )


# 헬퍼
def vec(s):
    """'a,b,c' -> (a, b, c)"""
    return tuple(float(x) for x in s.split(","))


def usd_ok(path):
    """로컬 마운트면 존재 확인, 원격 URL 이면 통과."""
    if path.startswith(("omniverse://", "http://", "https://")):
        return True
    return os.path.exists(path)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


# 씬 조립
def build(a, with_camera=True):
    """씬 cfg 를 만든다. AppLauncher 기동 이후에만 호출할 것."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg, ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

    sep = a.base_sep / 2.0

    # ---- 로봇, 지면, 조명 -------------------------------------------
    rot_r = (1.0, 0.0, 0.0, 0.0) if a.layout == "parallel" else (0.0, 0.0, 0.0, 1.0)
    pos_r = (0.0, -sep, 0.0) if a.layout == "parallel" else (0.9, -sep, 0.0)

    # 접촉 센서를 쓰려면 spawn 에서 contact reporter API 를 켜야 한다
    # (Franka 기본값 False). self-collision 은 net force 판정의 오탐이 되므로
    # 끈다 (기본값 True). replace 로 사본을 만들어 원본 cfg 를 오염시키지 않는다.
    franka_spawn = FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(
        activate_contact_sensors=True,
        articulation_props=FRANKA_PANDA_HIGH_PD_CFG.spawn.articulation_props.replace(
            enabled_self_collisions=False
        ),
    )

    @configclass
    class DualFrankaSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg()
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        # IK 추종이 좋은 강성 PD 판. Franka 스택 태스크가 쓰는 것과 동일.
        robot_l = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotL",
            spawn=franka_spawn,
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(0.0, sep, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
            ),
        )
        robot_r = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotR",
            spawn=franka_spawn,
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=pos_r, rot=rot_r
            ),
        )

    cfg = DualFrankaSceneCfg(num_envs=a.num_envs, env_spacing=3.0)

    # ---- 접촉 센서 (로봇-로봇/테이블 충돌 감지 -> 에피소드 실패 처리) ----
    # multi-body prim + filter 조합은 API 미지원이라 필터 없이 net force 를 쓴다.
    # hand/finger 는 물체를 정상적으로 만지므로 제외하고 link2~7 만 본다.
    # history_length>0 이면 scene.update 마다 버퍼가 강제 갱신되어
    # 렌더 데시메이션 중에도 순간 접촉을 놓치지 않는다.
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

    # ---- 카메라 ------------------------------------------------------
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
                horizontal_aperture=20.955,
                clipping_range=(0.05, 30.0),
            ),
        )
        # 손목 카메라 2대 (2026-08-11, 3뷰 전환)
        #   장착 위치/자세: Isaac Lab 공식 Franka 스택 태스크의 wrist_cam
        #     (stack_ik_rel_visuomotor_env_cfg.py, panda_hand 부착, ros 규약)
        #   화각: LIBERO / robosuite eye_in_hand 를 따라 광각 (기본 75도).
        #     Isaac Lab 기본 aperture 20.955 는 약 47도라 훨씬 좁다.
        #     focal 24 기준 aperture = 2 * f * tan(fov/2)
        wrist_ap = 2.0 * 24.0 * math.tan(math.radians(a.wrist_fov) / 2.0)
        wrist_pos = vec(a.wrist_pos)
        wrist_rot = vec(a.wrist_rot)

        def wrist_cam_cfg(robot):
            return CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{robot}/panda_hand/wrist_cam",
                update_period=0.0,
                height=a.cam_h,
                width=a.cam_w,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=wrist_ap,
                    clipping_range=(0.02, 2.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=wrist_pos, rot=wrist_rot, convention="ros"
                ),
            )

        cfg.wrist_cam_l = wrist_cam_cfg("RobotL")
        cfg.wrist_cam_r = wrist_cam_cfg("RobotR")

    # ---- 테이블 ------------------------------------------------------
    def table_path():
        u = a.table_usd
        if u.startswith(("/", "omniverse://")):
            return u
        return f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/{u}/table_instanceable.usd"

    def usd_table(name, y, mirror):
        """테이블 한 장. mirror=2 는 거울 반사 (TABLE_BODY_OFFSET 주석 참조)."""
        rot, dx, scale = vec(a.table_rot), a.table_dx, (1.0, 1.0, 1.0)
        if mirror == 1:  # z축 180도. x 오프셋도 뒤집혀 어긋난다
            rot = _quat_mul((0.0, 0.0, 0.0, 1.0), rot)
            dx = -a.table_dx
        elif mirror == 2:  # 거울 반사: y 만 반전. Rz(t) -> Rz(-t) + scale y=-1
            w, x, yq, z = rot
            rot = (w, -x, yq, -z)
            scale = (1.0, -1.0, 1.0)
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(dx, y, 0.0), rot=rot),
            spawn=sim_utils.UsdFileCfg(usd_path=table_path(), scale=scale),
        )

    def proc_table():
        """절차적 테이블. 원점이 곧 상판 중심이라 반전, 마운트판 문제가 없다."""
        sx, sy, th = vec(a.table_size)
        cfg.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(a.table_dx, 0.0, -th / 2)),
            spawn=sim_utils.CuboidCfg(
                size=(sx, sy, th),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.35, 0.35, 0.38), roughness=0.7
                ),
            ),
        )
        leg, hz = 0.06, (a.table_height - th) / 2.0
        corners = [
            (sx / 2 - leg, sy / 2 - leg),
            (sx / 2 - leg, -sy / 2 + leg),
            (-sx / 2 + leg, sy / 2 - leg),
            (-sx / 2 + leg, -sy / 2 + leg),
        ]
        for i, (lx, ly) in enumerate(corners):
            setattr(
                cfg,
                f"table_leg_{i}",
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/TableLeg" + str(i),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(a.table_dx + lx, ly, -th - hz)
                    ),
                    spawn=sim_utils.CuboidCfg(
                        size=(leg, leg, a.table_height - th),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.25, 0.25, 0.28)
                        ),
                    ),
                ),
            )

    has_table = True
    if a.table == "none":
        has_table = False
    elif a.table == "proc":
        proc_table()
    elif not usd_ok(table_path()):
        has_table = False
        print(f"[scene] table USD not found -> skipped ({table_path()})")
    elif a.table == "dual":
        ml = a.table_mirror if a.mirror_side in ("left", "both") else 0
        mr = a.table_mirror if a.mirror_side in ("right", "both") else 0
        cfg.table_l = usd_table("TableL", +sep, ml)
        cfg.table_r = usd_table("TableR", -sep, mr)
    else:
        cfg.table = usd_table("Table", 0.0, 0)

    # 테이블이 없으면 지면 자체가 작업면이 되어야 큐브가 안 떨어진다.
    cfg.ground.init_state = AssetBaseCfg.InitialStateCfg(
        pos=(0.0, 0.0, -1.05 if has_table else 0.0)
    )

    # ---- 큐브 --------------------------------------------------------
    # 항상 MAX_CUBES 개를 만들어 두고, 에피소드마다 필요한 개수만 배치한다.
    # 안 쓰는 것은 PARK 로 치운다 (씬을 다시 만들지 않고 개수를 바꿀 수 있다).
    cube_dir = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks"
    cube_props = sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )

    def make_cube(i):
        color = CUBE_COLORS[i - 1]
        prim = "{ENV_REGEX_NS}/Cube_" + str(i)
        init = RigidObjectCfg.InitialStateCfg(
            pos=CUBE_HOME[f"cube_{i}"], rot=(1.0, 0.0, 0.0, 0.0)
        )
        usd = f"{cube_dir}/{_USD_NAME[color]}" if color in _USD_NAME else None
        if usd and usd_ok(usd):
            spawn = sim_utils.UsdFileCfg(
                usd_path=usd, scale=(1.0, 1.0, 1.0), rigid_props=cube_props
            )
        else:  # USD 가 없으면 같은 색 절차적 큐브
            spawn = sim_utils.CuboidCfg(
                size=(CUBE_SIZE,) * 3,
                rigid_props=cube_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_RGB[color]),
            )
        return RigidObjectCfg(prim_path=prim, init_state=init, spawn=spawn)

    for i in range(1, MAX_CUBES + 1):
        setattr(cfg, f"cube_{i}", make_cube(i))

    print(
        f"[scene] layout={a.layout} base_sep={a.base_sep} table={a.table} "
        f"mirror={a.table_mirror}/{a.mirror_side} cubes={MAX_CUBES}"
    )
    return cfg
