#!/usr/bin/env python
# ======================================
# File: office_scan_scene.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] office_scan_scene.py 2026-08-31-r5  (ascii-only console/comments)
# r5: env 격리 -- office_scene import/sys.modules 위임을 물리 사본으로 대체
#     ([A] 엔진 사본 + [B] scan 오버라이드 합성. effective 값/PROTOCOL 불변
#      = office-scan-v1 유지, 벤치마크 비교성 보존)
# r4: 자산 위치 datasets/ -> envs/office_scan/assets/ (env 자산 동거)
# r3: effective JSON 에 portrait_side / desk_color_raw 기록 (값 불변)
# r2: protocol 상수 명문화, 구운 가구 정책 2원화(기본값+delattr), 색 변환
#     helper 통일, scan-pos 자동배치 yaw 부호 반영, effective JSON 로그
r"""scan 책상 배경의 office marker task scene (4번째 환경 = 실사 배경).

env 격리 원칙의 자기완결 모듈: [A] = office_scene 엔진 본문 사본 (무수정),
[B] = scan 오버라이드. 하는 일:

  1. scan mesh 에 이미 구워진 가구를 procedural 에서 뺌
       기본값 off (CLI 로 복원 가능): partition / keyboard / mouse(pad)
       / pc / 세로 모니터 / holder (holder 는 task 고정물이라 eval 이
       자기 기본값으로 재점등)
       토글이 없어 cfg 에서 직접 떼는 것: _BAKED (메인 모니터 3피스
       + 단상 3피스)
       남기는 것: 책상 상판/다리 -- 벤치 전체의 "충돌" 담당
  2. scan 책상 USD 를 얹음 (시각 전용, 충돌 없음). scan z0 = office z0
     = 상판이라 두 면이 자동 일치, z-fighting 은 --scan-lift (기본 2mm)
     로 회피. 배치는 칸막이 기준 자동 정렬
  3. 축 변환: scan frame 은 칸막이가 +y, office 는 뒤쪽이 -x
     -> z 축 +90도 회전 (자동 배치는 yaw 부호를 따라감)

task(마커/연필꽂이)/로봇 rig/카메라/성공 판정 = office 와 동일.
확정 상수([B] 의 PROTOCOL 블록) = benchmark 정의의 정본 -- 값을 바꾸면
PROTOCOL 문자열을 올림 (runner 는 이 값들을 전달하지 않음).
[A] 원본 = envs/office/office_scene.py (r3). 원본 갱신 시 [A] 재복사 필수.
"""

# =========================================================================
# [A] office_scene r3 엔진 본문 사본 (무수정 -- 원본과 diff 0 이 계약.
#     엔진 설명은 원본 헤더 참조)
# =========================================================================

import math
import os

PROPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "props")

# ---- 책상 ---------------------------------------------------------------
# 가운데 책상 + 양옆 보조 책상이 눕힌 디귿(디귿을 눕힌 U)을 이룸.
# 세 책상의 뒤쪽(-x) 가장자리를 맞추고, 보조 책상이 사람 쪽(+x)으로 뻗음.
DESK_W = 1.4  # 가운데 책상 y 방향 폭
DESK_D = 0.7  # 가운데 책상 x 방향 깊이
DESK_H = 0.70  # 바닥에서 상판까지 (사용자 실측)
DESK_TH = 0.04  # 상판 두께
DESK_COLOR = (0.93, 0.93, 0.91)  # 흰색
LEG_COLOR = (0.80, 0.80, 0.80)
WING_LEFT = (0.6, 1.2)  # 왼쪽(-y) 보조: y 폭, x 깊이
WING_RIGHT = (0.4, 1.2)  # 오른쪽(+y) 보조: y 폭, x 깊이
# 파티션(칸막이): 뒤 + 좌우를 둘러쌈. 높이는 바닥 기준.
PARTITION_H = 1.2
PARTITION_TH = 0.02
PARTITION_COLOR = (0.28, 0.28, 0.30)  # 어두운 회색

# ---- 모니터 -------------------------------------------------------------
# 실측 (extract_props.py, office.usd 의 SM_Monitor2): 두께 x, 가로 y, 세로 z.
# 대각선이 32.8인치라, 원하는 인치를 주면 그 비율로 축척함 (--panel-inch).
PANEL_SIZE = (0.078, 0.726, 0.408)
PANEL_DIAG_IN = 32.79  # sqrt(0.726^2 + 0.408^2) = 0.833m
# 스탠드는 "긴 기둥 + 그 앞면에 붙는 패널" 구조 (실제 모니터와 같은 형태).
# 기둥 위에 패널을 얹지 않음.
STAND_COL = (0.05, 0.06)  # 기둥 단면 x, y (높이는 패널에서 계산)
STAND_BASE = (0.20, 0.24, 0.015)  # 스탠드 받침
# 기둥은 패널 "중앙"에 붙음 (VESA mount 와 같은 위치).
# 하단 가장자리에 얹히면 모니터가 아니라 받침대처럼 보임.
COL_TOP_RATIO = 0.5
# 단상은 눕힌 디귿(아치) 모양. 상판 + 양옆 다리, 아래는 비어 있음.
# 너비는 모니터 가로폭과 같게 맞춤 (--panel-inch 를 따라감).
RISER_D = 0.26  # x 깊이
RISER_H = 0.10  # 높이 (사용자 실측: 10cm 단상)
RISER_TH = 0.018  # 상판/다리 두께
MONITOR_X = -0.25  # 먼 가장자리
PANEL_COLOR = (0.04, 0.04, 0.05)
STAND_COLOR = (0.12, 0.12, 0.14)
RISER_COLOR = (0.74, 0.60, 0.42)  # 밝은 나무색

# ---- 입력장치 -----------------------------------------------------------
# 치수는 전부 extract_props.py 실측값 (primitive 대체용으로도 같은 값 사용)
KEYBOARD_X = 0.10
KEYBOARD_SIZE = (0.156, 0.355, 0.016)  # x=깊이, y=가로, z=높이
KEYBOARD_COLOR = (0.14, 0.14, 0.16)
MOUSEPAD_XY = (0.08, 0.32)  # 사람 기준 오른쪽
MOUSEPAD_SIZE = (0.200, 0.245, 0.002)
MOUSEPAD_COLOR = (0.10, 0.10, 0.12)
MOUSE_SIZE = (0.105, 0.053, 0.030)  # 파지 폭 5.3cm (gripper 8cm 안)
MOUSE_COLOR = (0.16, 0.16, 0.18)

# ---- 본체 (PC 타워) -----------------------------------------------------
# office.usd 의 SM_PC* 는 전부 화면 일체형(iMac 형)이라 타워로 못 씀.
# 일반적인 미들타워 치수로 직접 제작. 긴 면(x)이 벽과 나란히 섬.
PC_SIZE = (0.45, 0.20, 0.45)  # x 깊이(벽 방향), y 폭, z 높이
PC_GAP_FRONT = 0.15  # 앞쪽(사람이 마주보는) 파티션에서 띄우는 거리
PC_COLOR = (0.03, 0.03, 0.035)  # 검정
PC_FRONT_COLOR = (0.07, 0.07, 0.08)  # 전면 패널 (방 쪽을 향하는 면). 살짝만 밝게

# ---- 로봇 (사람 자리. 받침대 위에 세움) ---------------------------------
ROBOT_BASE_X = 0.62
ROBOT_BASE_SEP = 0.60  # 두 base 사이 거리
ROBOT_BASE_Z = -0.05  # 상판(0) 보다 살짝 아래
# 받침대는 두 팔을 잇는 한 장의 낮은 책상. 따로 세우면 어색함
PEDESTAL_D = 0.36  # x 깊이
PEDESTAL_MARGIN = 0.16  # 베이스 바깥으로 남기는 여유 (양쪽)
# 색은 선형값이라 화면에서는 감마 보정으로 훨씬 밝아짐.
# linear 0.30 은 sRGB 로 약 58% (중간 회색보다 밝음). 어두운 회색은 0.07 근처.
PEDESTAL_COLOR = (0.075, 0.075, 0.085)  # 어두운 회색 (sRGB 약 30%)

# ---- 학습용 카메라 (ego_view) -------------------------------------------
# 서 있는 사람이 로봇 뒤에서 어깨 너머로 책상을 내려다보는 시점.
# 바닥에서 1.60m (바닥이 z=-0.70), 로봇 base(x=0.62)보다 0.58m 뒤.
# 생성기와 평가 harness 가 같은 값을 써야 학습/평가 분포가 일치함.
EGO_EYE = (1.20, 0.00, 0.90)
EGO_TARGET = (0.05, 0.00, 0.08)

# ---- 연필꽂이 (정리 목표 용기) ------------------------------------------
# 왼쪽(세로) 모니터 바로 앞. 왼팔 0.57m 로 여유롭게 닿음 (오른팔은 0.95m 라
# 불가 -> 오른쪽 물건은 공중 handover 로 왼팔에 넘겨야 함).
# USD 컵(SM_PencilCup 등)은 충돌이 볼록껍질이라 속이 막혀서 못 씀.
# 바닥판 + 벽 4장으로 직접 제작.
# (08-10 사용자) ego_view 에 다 보여야 해서 오른쪽으로 이동 (-0.48 -> -0.36).
# 왼팔 0.59 로 여유, 오른팔 0.88 로 여전히 불가 = handover 구조 유지.
HOLDER_XY = (0.10, -0.36)
HOLDER_INNER = (0.17, 0.17, 0.100)  # 안쪽 x, y, 벽 높이 (08-10 사용자: 2배 확대)
HOLDER_WALL = 0.008
HOLDER_COLOR = (0.20, 0.26, 0.34)

# ---- 정리 대상: 마커 ----------------------------------------------------
# 실물 펜 asset(SM_Pen)은 높이 1.2cm 라 눕힌 것을 집으면 손끝이 상판에 닿음
# (1번 환경에서 검증된 안전 파지 높이는 2.5cm). 지름 2.6cm 마커로 제작.
# 길이 14cm 라 양 끝을 각각 잡을 수 있어 공중 handover 에도 맞음.
PARK = (2.5, 2.5, 0.5)
MARKER_R = 0.013
MARKER_SQ = 0.024  # 각단면(box) mode 의 한 변. 면 파지라 원기둥보다 훨씬 안정
MARKER_L = 0.140
# build() 에서 --marker-shape 값으로 설정됨 (item_half_h 등이 참조)
MARKER_SHAPE = "cyl"
MARKER_COLORS = [
    (0.80, 0.15, 0.15),
    (0.15, 0.35, 0.80),
    (0.15, 0.60, 0.25),
    (0.85, 0.65, 0.10),
]
MAX_ITEMS = len(MARKER_COLORS)


def item_name(i):
    return f"marker{i}"


def item_half_h(i):
    """눕혀 놓았을 때의 중심 높이 [m]. usd mode 도 충돌체는 box 라 동일."""
    h = MARKER_R if MARKER_SHAPE == "cyl" else MARKER_SQ / 2.0
    return h + 0.0003


def item_grasp_w(i):
    """gripper 가 무는 폭 [m]. 8cm 초과 금지. usd = box 충돌체."""
    return 2.0 * MARKER_R if MARKER_SHAPE == "cyl" else MARKER_SQ


# ---- 헬퍼 ---------------------------------------------------------------
def vec(s):
    return tuple(float(x) for x in s.split(","))


def prop_usd(name):
    """props/ 에 추출해 둔 USD 경로. 없으면 None (primitive 로 대체)."""
    p = os.path.join(PROPS_DIR, f"{name}.usd")
    return p if os.path.exists(p) else None


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
    """x/y/z 축 회전 quaternion (w,x,y,z)."""
    import math

    h = math.radians(deg) / 2.0
    c, s = math.cos(h), math.sin(h)
    return {
        "x": (c, s, 0.0, 0.0),
        "y": (c, 0.0, s, 0.0),
        "z": (c, 0.0, 0.0, s),
    }[axis]


# ---- argparse -----------------------------------------------------------
def add_scene_args(p):
    """scene 인자. 앱 기동 전에 호출해도 안전함 (isaaclab 불필요)."""
    g = p.add_argument_group("desk")
    g.add_argument("--desk-w", type=float, default=DESK_W, help="desk width along y")
    g.add_argument("--desk-d", type=float, default=DESK_D, help="desk depth along x")
    g.add_argument("--desk-h", type=float, default=DESK_H, help="floor to top [m]")
    g.add_argument(
        "--desk-usd", default="", help="use a desk USD instead of the procedural one"
    )
    g.add_argument(
        "--wing-left",
        default=f"{WING_LEFT[0]},{WING_LEFT[1]}",
        help="left side desk 'y_width,x_depth' [m]; empty = none. "
        "Its back edge lines up with the center desk and it runs toward +x",
    )
    g.add_argument(
        "--wing-right",
        default=f"{WING_RIGHT[0]},{WING_RIGHT[1]}",
        help="right side desk 'y_width,x_depth' [m]; empty = none",
    )
    g.add_argument(
        "--partition",
        type=int,
        default=1,
        help="1: gray cubicle partitions around the back and both sides",
    )
    g.add_argument(
        "--partition-h",
        type=float,
        default=PARTITION_H,
        help="partition height measured from the floor [m]",
    )

    g = p.add_argument_group("monitors")
    g.add_argument(
        "--portrait-side",
        choices=["left", "right", "none"],
        default="left",
        help="which side the pivoted (portrait) monitor sits on, "
        "seen by the person at +x (left = -y)",
    )
    g.add_argument(
        "--monitor-x",
        type=float,
        default=0.0,
        help="monitor row x; 0 = auto (riser tucked against the back edge)",
    )
    g.add_argument(
        "--riser-h",
        type=float,
        default=RISER_H,
        help="height of the platform under the main monitor [m]",
    )
    g.add_argument(
        "--panel-yaw",
        type=float,
        default=0.0,
        help="extra yaw for the panel USD if its native facing is off [deg]",
    )
    g.add_argument(
        "--panel-inch",
        type=float,
        default=26.0,
        help="monitor diagonal in inches; the source panel is 32.8 inch "
        "and gets scaled by inch/32.8",
    )
    g.add_argument(
        "--panel-origin",
        choices=["center", "bottom"],
        default="center",
        help="origin convention of props/monitor_panel.usd; check the "
        "'z range' column printed by extract_props.py "
        "(0..h = bottom, -h/2..h/2 = center)",
    )
    g.add_argument(
        "--main-lift",
        type=float,
        default=STAND_BASE[2] + 0.075,
        help="main panel bottom above the riser top [m]",
    )
    g.add_argument(
        "--sub-yaw",
        type=float,
        default=25.0,
        help="toe-in angle of the portrait monitor toward the person [deg]; "
        "the stand turns with it",
    )
    g.add_argument(
        "--monitor-gap",
        type=float,
        default=0.020,
        help="gap between the two screens [m]; 5mm made the bezels touch, "
        "so 2cm keeps them continuous but clear of each other",
    )
    g.add_argument(
        "--sub-lift",
        type=float,
        default=STAND_BASE[2] + 0.01,
        help="portrait panel bottom above the desk; must clear the stand "
        "base plate (0.015) or the panel sinks into it, so base + 1cm",
    )
    g.add_argument(
        "--use-usd",
        type=int,
        default=1,
        help="1: use props/*.usd when present / 0: primitives only",
    )

    g = p.add_argument_group("pc tower")
    g.add_argument(
        "--pc",
        type=int,
        default=1,
        help="1: PC tower against the right partition, long side to the wall",
    )
    g.add_argument(
        "--pc-place",
        choices=["desk", "floor"],
        default="desk",
        help="stand the tower on the side desk or on the floor",
    )
    g.add_argument(
        "--pc-gap",
        type=float,
        default=PC_GAP_FRONT,
        help="distance from the front partition (the one the person faces) [m]",
    )

    g = p.add_argument_group("holder")
    g.add_argument(
        "--holder",
        type=int,
        default=1,
        help="1: pencil holder in front of the portrait monitor (left arm only)",
    )
    g.add_argument(
        "--holder-xy",
        default=f"{HOLDER_XY[0]},{HOLDER_XY[1]}",
        help="pencil holder center x,y [m]",
    )

    g = p.add_argument_group("desk items")
    g.add_argument("--keyboard", type=int, default=1)
    g.add_argument("--mousepad", type=int, default=1)
    g.add_argument("--mouse", type=int, default=1)
    g.add_argument(
        "--mouse-rot",
        default="",
        help="extra rotation for the mouse USD, e.g. x180 / y90 / z-90. "
        "The extracted asset may be flipped or on its side; try values here "
        "instead of re-extracting. Empty = as extracted",
    )
    g.add_argument(
        "--items",
        type=int,
        default=0,
        help="number of tidy-up items to spawn (stage 2; 0 = background only)",
    )
    g.add_argument(
        "--marker-shape",
        choices=["cyl", "box", "usd"],
        default="box",
        help="marker cross-section. box(default) = square pen: flat-face "
        "pinch, no rolling, no in-grip rotation. cyl was abandoned -- "
        "round pinch kept slipping (08-10). usd = real pen visual from "
        "props/ over the SAME invisible box collider (physics unchanged)",
    )
    g.add_argument(
        "--marker-usd",
        default="marker_blue",
        help="props/<name>.usd for --marker-shape usd (made by "
        "extract_props.py --marker-src; falls back to box if missing)",
    )

    g = p.add_argument_group("robots (stage 3)")
    g.add_argument(
        "--robots",
        type=int,
        default=0,
        help="0: desk only / 1: two Frankas facing the desk from the person side",
    )
    g.add_argument("--base-sep", type=float, default=ROBOT_BASE_SEP)
    g.add_argument("--base-x", type=float, default=ROBOT_BASE_X)
    g.add_argument(
        "--base-z",
        type=float,
        default=ROBOT_BASE_Z,
        help="robot base height relative to the desk top (0); negative = "
        "below the top. A pedestal is built from the floor up to it",
    )

    g = p.add_argument_group("render")
    g.add_argument("--num-envs", type=int, default=1)
    g.add_argument("--cam-w", type=int, default=1280)
    g.add_argument("--cam-h", type=int, default=720)
    # 손목 카메라 2대 (3-view 전환, docs/threeview_camera_upgrade.md).
    # 장착 위치/자세는 Isaac Lab 공식 Franka stack task 의 wrist_cam 값,
    # 화각은 LIBERO/robosuite eye_in_hand 스타일 광각 75도.
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
    # 쇼케이스용 고화질 다각도 카메라 (학습 데이터와 무관, 기본 꺼짐).
    # pipe 로 나눈 시점 목록을 주면 그 수만큼 HD 카메라가 추가됨.
    g.add_argument(
        "--hd-views",
        default="",
        help="extra high-res cameras 'x,y,z|x,y,z|...' (eyes; all look at "
        "EGO_TARGET). Empty = off. Rendering cost grows per camera",
    )
    g.add_argument("--hd-w", type=int, default=1280)
    g.add_argument("--hd-h", type=int, default=720)


# ---- scene 조립 ---------------------------------------------------------
def build(a, with_camera=True):
    """scene cfg 생성. AppLauncher 기동 이후에만 호출할 것."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    @configclass
    class OfficeDeskSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -a.desk_h)),
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.80)),
        )

    cfg = OfficeDeskSceneCfg(num_envs=a.num_envs, env_spacing=4.0)

    # 형광등 느낌의 보조 조명 (책상 위가 너무 평평하지 않게)
    cfg.key_light = AssetBaseCfg(
        prim_path="/World/key_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, 1.6)),
        spawn=sim_utils.DiskLightCfg(intensity=6000.0, radius=0.35),
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

    def static_usd(name, usd, pos, rot=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        # 정적 소품은 시각 맥락이 목적이라 충돌을 붙이지 않음. 붙이면 mesh
        # 볼록껍질이 생겨 나중에 팔이 근처를 지날 때 불필요하게 걸림.
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=sim_utils.UsdFileCfg(usd_path=usd, scale=scale),
        )

    def pick_usd(name):
        return prop_usd(name) if a.use_usd else None

    # ---- 책상 (가운데 + 좌우 보조 = 눕힌 디귿) ---------------------------
    def desk_slab(tag, cx, cy, sx, sy, top_z=0.0, color=DESK_COLOR):
        """상판 + 다리 4개. (cx, cy) 는 상판 중심, (sx, sy) 는 x/y 크기.

        top_z 는 상판 윗면 높이 (기본 0 = 책상 상판). 로봇 받침대처럼
        더 낮은 판을 만들 때 사용. 다리는 항상 바닥까지 내림.
        상판에는 마찰 재질 부여 (마커가 손끝에 밀려 굴러가는 것 억제).
        """
        setattr(
            cfg,
            f"desk_top_{tag}",
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + f"DeskTop{tag}",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(cx, cy, top_z - DESK_TH / 2)
                ),
                spawn=sim_utils.CuboidCfg(
                    size=(sx, sy, DESK_TH),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.1, dynamic_friction=1.0
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=color, roughness=0.55
                    ),
                ),
            ),
        )
        leg_h = max(top_z - DESK_TH + a.desk_h, 0.02)
        hz = leg_h / 2.0
        leg, inset = 0.06, 0.05
        for i, (dx, dy) in enumerate(
            [
                (sx / 2 - inset, sy / 2 - inset),
                (sx / 2 - inset, -sy / 2 + inset),
                (-sx / 2 + inset, sy / 2 - inset),
                (-sx / 2 + inset, -sy / 2 + inset),
            ]
        ):
            setattr(
                cfg,
                f"desk_leg_{tag}_{i}",
                static_box(
                    f"DeskLeg{tag}{i}",
                    (cx + dx, cy + dy, top_z - DESK_TH - hz),
                    (leg, leg, leg_h),
                    LEG_COLOR,
                ),
            )

    back_x = -a.desk_d / 2  # 세 책상이 공유하는 뒤쪽 가장자리
    if a.desk_usd:
        cfg.desk = static_usd("Desk", a.desk_usd, (0.0, 0.0, -a.desk_h))
    else:
        desk_slab("C", 0.0, 0.0, a.desk_d, a.desk_w)
        # 보조 책상: 뒤쪽 가장자리를 맞추고 사람 쪽(+x)으로 뻗음.
        # 가운데 책상 "바깥"에 붙음 -> 전체 폭 = 왼쪽 + 가운데 + 오른쪽,
        # 사람 자리(포켓)는 가운데 책상 폭 그대로 남음.
        for tag, spec, sgn in (("L", a.wing_left, -1.0), ("R", a.wing_right, 1.0)):
            if not spec.strip():
                continue
            wy, wx = vec(spec)
            desk_slab(
                tag,
                back_x + wx / 2,
                sgn * (a.desk_w / 2 + wy / 2),
                wx,
                wy,
            )

    # ---- 파티션 (회색 칸막이. 뒤 + 좌우를 둘러쌈) ------------------------
    if a.partition:
        th = PARTITION_TH
        pz = -a.desk_h + a.partition_h / 2  # 바닥에서 올라옴
        # 보조 책상이 좌우로 얼마나 나가 있는지에 맞춰 파티션을 두름
        wl = vec(a.wing_left) if a.wing_left.strip() else None
        wr = vec(a.wing_right) if a.wing_right.strip() else None
        y_left = -(a.desk_w / 2 + (wl[0] if wl else 0.0))
        y_right = a.desk_w / 2 + (wr[0] if wr else 0.0)
        depth_l = wl[1] if wl else a.desk_d
        depth_r = wr[1] if wr else a.desk_d
        # 뒤쪽: 좌우 끝까지 한 장으로 덮음 (좌우 폭이 달라 중심이 0 이 아님)
        cfg.partition_back = static_box(
            "PartitionBack",
            (back_x - th / 2, (y_left + y_right) / 2, pz),
            (th, (y_right - y_left) + 2 * th, a.partition_h),
            PARTITION_COLOR,
            rough=0.85,
        )
        # 좌우: 보조 책상 바깥면을 따라 사람 쪽으로 뻗음
        for tag, ye, dep, sgn in (
            ("L", y_left, depth_l, -1.0),
            ("R", y_right, depth_r, 1.0),
        ):
            setattr(
                cfg,
                f"partition_{tag.lower()}",
                static_box(
                    f"Partition{tag}",
                    (back_x + dep / 2, ye + sgn * th / 2, pz),
                    (dep, th, a.partition_h),
                    PARTITION_COLOR,
                    rough=0.85,
                ),
            )

    # ---- 모니터 (긴 기둥 + 그 앞면에 붙는 패널) -------------------------
    # 실제 모니터처럼 기둥 위에 얹는 것이 아니라 기둥 앞면(+x, 사람 쪽)에
    # 패널을 붙임. 세로 모니터는 패널만 화면 법선(x축) 기준 90도 돌림.
    panel_usd = pick_usd("monitor_panel")
    s = a.panel_inch / PANEL_DIAG_IN
    pt, pw, ph = (v * s for v in PANEL_SIZE)  # 두께, 가로, 세로 (축척 후)
    yaw_fix = quat_axis("z", a.panel_yaw)
    # 모니터 열 위치. 0 이면 단상이 책상 뒤 가장자리에 딱 붙도록 자동 계산함
    # (책상 깊이를 줄이면 모니터가 책상 밖으로 나가는 것을 막음)
    mon_x = a.monitor_x if a.monitor_x != 0.0 else back_x + RISER_D / 2 + 0.01

    def monitor(tag, y, portrait, base_z, lift, yaw=0.0, mx=None):
        """받침 + 긴 기둥 + 앞면에 붙는 패널.

        base_z  스탠드가 서는 높이 (단상 위면 단상 높이)
        lift    패널 아래끝이 base_z 위로 뜨는 양
                (세로 모니터는 화면이 상판에 거의 닿아야 해서 아주 작음)
        yaw     사람 쪽으로 트는 각도 [deg]. 스탠드도 같이 돌고, 패널이
                기둥 앞에 붙는 offset 도 함께 회전함
        """
        import math

        if mx is None:
            mx = mon_x
        bx, by, bz = STAND_BASE
        kx, ky = STAND_COL
        panel_h = pw if portrait else ph  # 화면 높이
        panel_w = ph if portrait else pw  # 화면 가로
        panel_bottom = base_z + lift
        # 기둥 꼭대기가 패널 중앙에 오도록 길이를 잡음 (받침 두께는 뺌)
        col_h = max(panel_bottom + panel_h * COL_TOP_RATIO - (base_z + bz), 0.02)
        q_yaw = quat_axis("z", yaw)
        c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))

        setattr(
            cfg,
            f"stand_base_{tag}",
            static_box(
                f"StandBase{tag}",
                (mx, y, base_z + bz / 2),
                (bx, by, bz),
                STAND_COLOR,
                rot=q_yaw,
            ),
        )
        setattr(
            cfg,
            f"stand_col_{tag}",
            static_box(
                f"StandCol{tag}",
                (mx, y, base_z + bz + col_h / 2),
                (kx, ky, col_h),
                STAND_COLOR,
                rot=q_yaw,
            ),
        )

        # 기둥 중심 기준 offset 을 먼저 정하고, 그 벡터를 yaw 만큼 돌림.
        # (돌리지 않으면 패널이 기둥에서 떨어져 보임)
        ox = kx / 2 + pt / 2  # 기둥 앞면 + 패널 두께 절반
        oy, pz = 0.0, panel_bottom + panel_h / 2
        rot = quat_mul(quat_axis("x", 90.0), yaw_fix) if portrait else yaw_fix
        if panel_usd and a.panel_origin != "center":
            # bottom 규약: 세로로 돌리면 패널이 -y 로 눕는 만큼 보정 필요
            if portrait:
                oy = panel_w / 2
            else:
                pz = panel_bottom
        pos = (
            mx + ox * c - oy * sn,
            y + ox * sn + oy * c,
            pz,
        )
        rot = quat_mul(q_yaw, rot)

        if panel_usd:
            setattr(
                cfg,
                f"panel_{tag}",
                static_usd(f"Panel{tag}", panel_usd, pos, rot=rot, scale=(s, s, s)),
            )
        else:
            # primitive 는 항상 중심 원점이라 크기만 바꿔 놓으면 됨
            size = (pt, panel_w, panel_h)
            setattr(
                cfg,
                f"panel_{tag}",
                static_box(f"Panel{tag}", pos, size, PANEL_COLOR, rough=0.25, rot=rot),
            )
        return panel_h

    # 메인 단상: 눕힌 디귿(아치). 상판 + 양옆 다리, 너비는 모니터 가로폭과 같음
    cfg.riser_top = static_box(
        "RiserTop",
        (mon_x, 0.0, a.riser_h - RISER_TH / 2),
        (RISER_D, pw, RISER_TH),
        RISER_COLOR,
        rough=0.5,
    )
    leg_h = a.riser_h - RISER_TH
    for i, sgn in enumerate((-1.0, 1.0)):
        setattr(
            cfg,
            f"riser_leg_{i}",
            static_box(
                f"RiserLeg{i}",
                (mon_x, sgn * (pw / 2 - RISER_TH / 2), leg_h / 2),
                (RISER_D, RISER_TH, leg_h),
                RISER_COLOR,
                rough=0.5,
            ),
        )
    monitor("main", 0.0, portrait=False, base_z=a.riser_h, lift=a.main_lift)

    # 세로: 사람 기준 왼쪽 = -y (오른쪽이면 +y). 화면이 상판에 거의 닿고,
    # 사람 쪽으로 --sub-yaw 만큼 틀어 화면이 정면으로 보이게 함.
    if a.portrait_side != "none":
        # 주의: 여기서 지역 import math 를 하면 build() 전체에서 math 가
        # 지역 이름이 되어, 이 분기를 안 타는 --portrait-side none 실행이
        # 아래 손목 카메라 aperture 계산에서 UnboundLocalError 로 죽음.
        # 모듈 상단의 import math 사용 (전수 감사에서 확정된 결함).
        sign = -1.0 if a.portrait_side == "left" else 1.0
        yaw = -sign * a.sub_yaw  # 왼쪽 모니터는 +z 로 돌아야 사람을 향함
        t = math.radians(a.sub_yaw)
        cy, sy = math.cos(t), math.sin(t)
        ox = STAND_COL[0] / 2 + pt / 2  # 기둥 중심에서 패널 중심까지

        # 두 화면의 이음매가 실제로 맞닿도록 역으로 풂.
        # 세로 패널의 안쪽 모서리 = 패널중심 + Rz(yaw)*(0, +/-ph/2) 이고,
        # 패널중심은 기둥에서 ox 만큼 앞으로 나가 있음 (그것도 yaw 로 돎).
        #   안쪽 모서리 y = 메인 패널 가장자리 -/+ gap
        #   안쪽 모서리 x = 메인 패널 면
        y_sub = sign * (pw / 2 + a.monitor_gap + ox * sy + (ph / 2) * cy)
        mx_sub = mon_x + ox * (1.0 - cy) + (ph / 2) * sy
        monitor(
            "sub",
            y_sub,
            portrait=True,
            base_z=0.0,
            lift=a.sub_lift,
            yaw=yaw,
            mx=mx_sub,
        )

    # ---- 연필꽂이 (왼쪽 모니터 앞. 바닥판 + 벽 4장) ----------------------
    if a.holder:
        hx, hy = vec(a.holder_xy)
        ix, iy, wh = HOLDER_INNER
        w = HOLDER_WALL
        cfg.holder_floor = static_box(
            "HolderFloor",
            (hx, hy, w / 2),
            (ix + 2 * w, iy + 2 * w, w),
            HOLDER_COLOR,
            rough=0.6,
        )
        for nm, dx, dy, sxx, syy in (
            ("Xp", ix / 2 + w / 2, 0.0, w, iy + 2 * w),
            ("Xn", -ix / 2 - w / 2, 0.0, w, iy + 2 * w),
            ("Yp", 0.0, iy / 2 + w / 2, ix, w),
            ("Yn", 0.0, -iy / 2 - w / 2, ix, w),
        ):
            setattr(
                cfg,
                f"holder_wall_{nm.lower()}",
                static_box(
                    f"HolderWall{nm}",
                    (hx + dx, hy + dy, w + wh / 2),
                    (sxx, syy, wh),
                    HOLDER_COLOR,
                    rough=0.6,
                ),
            )

    # ---- 본체 (오른쪽 파티션에 붙이고, 긴 면을 벽에 나란히) --------------
    if a.pc:
        wr_spec = vec(a.wing_right) if a.wing_right.strip() else None
        y_wall = a.desk_w / 2 + (wr_spec[0] if wr_spec else 0.0)  # 파티션 안쪽면
        foot_x, foot_y, pc_h = PC_SIZE
        pc_x = back_x + a.pc_gap + foot_x / 2  # 앞쪽 파티션에서 pc_gap 만큼
        pc_y = y_wall - foot_y / 2  # 벽에 딱 붙임
        pc_z = 0.0 if a.pc_place == "desk" else -a.desk_h
        cfg.pc = static_box(
            "PC",
            (pc_x, pc_y, pc_z + pc_h / 2),
            (foot_x, foot_y, pc_h),
            PC_COLOR,
            rough=0.35,
        )
        # 전면 패널: 방 쪽(-y)을 향하는 면에 얇게 덧대 타워처럼 보이게 함
        cfg.pc_front = static_box(
            "PCFront",
            (pc_x, pc_y - foot_y / 2 - 0.004, pc_z + pc_h / 2),
            (foot_x * 0.92, 0.008, pc_h * 0.92),
            PC_FRONT_COLOR,
            rough=0.3,
        )

    # ---- 키보드 / 마우스패드 / 마우스 -----------------------------------
    if a.keyboard:
        kb = pick_usd("keyboard")
        if kb:
            cfg.keyboard = static_usd("Keyboard", kb, (KEYBOARD_X, 0.0, 0.0))
        else:
            kx, ky, kz = KEYBOARD_SIZE
            cfg.keyboard = static_box(
                "Keyboard", (KEYBOARD_X, 0.0, kz / 2), (kx, ky, kz), KEYBOARD_COLOR
            )

    if a.mousepad:
        mp = pick_usd("mousepad")
        if mp:
            cfg.mousepad = static_usd(
                "MousePad", mp, (MOUSEPAD_XY[0], MOUSEPAD_XY[1], 0.0)
            )
        else:
            px, py, pz2 = MOUSEPAD_SIZE
            cfg.mousepad = static_box(
                "MousePad",
                (MOUSEPAD_XY[0], MOUSEPAD_XY[1], pz2 / 2),
                (px, py, pz2),
                MOUSEPAD_COLOR,
                rough=0.9,
            )

    if a.mouse:
        # 마우스는 나중에 정리 대상으로도 쓸 수 있게 rigid body 로 둠
        rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        mu = pick_usd("mouse")
        mz = MOUSE_SIZE[2]
        # 추출 USD 는 원점이 바닥, primitive 는 중심이라 놓는 높이가 다름.
        # --mouse-rot 을 주면 회전 때문에 바닥이 어긋날 수 있으므로 살짝 띄워
        # 떨어뜨림 (rigid body 라 물리가 알아서 안착시킴).
        rot = (1.0, 0.0, 0.0, 0.0)
        lift = 0.005
        if a.mouse_rot.strip():
            axis, deg = a.mouse_rot.strip()[0].lower(), float(a.mouse_rot.strip()[1:])
            rot = quat_axis(axis, deg)
            lift = 0.06  # 회전 후 어느 면이 아래로 갈지 모르니 넉넉히 띄움
        z0 = lift if mu else mz / 2 + lift
        init = RigidObjectCfg.InitialStateCfg(
            pos=(MOUSEPAD_XY[0], MOUSEPAD_XY[1], z0), rot=rot
        )
        if mu:
            spawn = sim_utils.UsdFileCfg(
                usd_path=mu,
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.09),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
        else:
            spawn = sim_utils.CuboidCfg(
                size=MOUSE_SIZE,
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.09),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=MOUSE_COLOR, roughness=0.4
                ),
            )
        cfg.mouse = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Mouse", init_state=init, spawn=spawn
        )

    # ---- 정리 대상 물건 (2단계) ----------------------------------------
    if a.items > 0:
        rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        # item_half_h 등 모듈 함수가 참조할 수 있게 전역에 반영
        global MARKER_SHAPE
        MARKER_SHAPE = a.marker_shape

        def make_item(i):
            """마커 하나. 긴 축은 z 라, 눕히는 회전은 생성기가 reset 때 줌."""
            shape = a.marker_shape
            if shape == "usd":
                mu = prop_usd(a.marker_usd)
                if mu:
                    # visual = 실물 펜 asset, 충돌/질량/마찰 = box 마커와 동일.
                    # box 충돌체(guide, 비가시)와 마찰 재질(2.0/1.8 combine=
                    # max)은 wrapper USD 에 authoring 됨 (extract_props
                    # --marker-src). UsdFileCfg 는 physics_material 을 못
                    # 받으므로 여기서 다시 주지 않음.
                    return RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}/Item_" + str(i),
                        init_state=RigidObjectCfg.InitialStateCfg(pos=PARK),
                        spawn=sim_utils.UsdFileCfg(
                            usd_path=mu,
                            rigid_props=rigid,
                            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                        ),
                    )
                print(f"[scene] props/{a.marker_usd}.usd missing -> box markers")
                shape = "box"
            color = MARKER_COLORS[(i - 1) % len(MARKER_COLORS)]
            common = dict(
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # 질량은 실물답게 20g 유지 (무거우면 grip 에서 기우는 토크가
                # 커짐). 미끄러짐/구름은 마찰로만 잡음.
                # combine=max: PhysX 기본은 두 면의 "평균"이라 손가락 패드
                # (~0.7)와 합치면 절반으로 깎임 -> max 로 2.0 을 강제
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=2.0,
                    dynamic_friction=1.8,
                    friction_combine_mode="max",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, roughness=0.45
                ),
            )
            if shape == "box":
                spawn = sim_utils.CuboidCfg(
                    size=(MARKER_SQ, MARKER_SQ, MARKER_L), **common
                )
            else:
                spawn = sim_utils.CylinderCfg(
                    radius=MARKER_R, height=MARKER_L, **common
                )
            return RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Item_" + str(i),
                init_state=RigidObjectCfg.InitialStateCfg(pos=PARK),
                spawn=spawn,
            )

        for i in range(1, min(a.items, MAX_ITEMS) + 1):
            setattr(cfg, f"item_{i}", make_item(i))

    # ---- 로봇 (사람 자리에서 책상을 마주봄) -----------------------------
    # 받침대 위에 올려 base 를 상판보다 살짝 아래(--base-z)에 둠.
    # 바닥에 그대로 세우면 상판 높이에서의 유효 반경이 크게 줄어듦.
    if a.robots:
        from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

        franka_spawn = FRANKA_PANDA_HIGH_PD_CFG.spawn.replace(
            activate_contact_sensors=True,
            articulation_props=FRANKA_PANDA_HIGH_PD_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=False
            ),
        )
        face_desk = (0.0, 0.0, 0.0, 1.0)  # z 축 180도. -x 를 향함
        sep = a.base_sep / 2.0
        # 받침대는 두 팔을 잇는 한 장의 낮은 책상 (따로 세우면 어색함).
        # 상판 윗면이 곧 로봇 base 높이가 됨.
        desk_slab(
            "ROB",
            a.base_x,
            0.0,
            PEDESTAL_D,
            a.base_sep + 2 * PEDESTAL_MARGIN,
            top_z=a.base_z,
            color=PEDESTAL_COLOR,
        )
        # 손가락 강성 강화: 기본 stiffness 2e3 이면 마커(반지름 1.3cm) 기준
        # 조임력 ~26N 이라 둥근 물체가 grip 안에서 기울고 빠짐 (실측).
        # 단, 너무 세면(1e4 = ~130N) 둥근 단면이 닫히는 손가락에서 튕겨
        # 나감 (squirt). 6e3 = ~78N 이 절충값.
        grip_act = FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_hand"].replace(
            stiffness=6.0e3, damping=2.0e2
        )
        actuators = dict(FRANKA_PANDA_HIGH_PD_CFG.actuators)
        actuators["panda_hand"] = grip_act
        cfg.robot_l = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotL",
            spawn=franka_spawn,
            actuators=actuators,
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(a.base_x, -sep, a.base_z), rot=face_desk
            ),
        )
        cfg.robot_r = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotR",
            spawn=franka_spawn,
            actuators=actuators,
            init_state=FRANKA_PANDA_HIGH_PD_CFG.init_state.replace(
                pos=(a.base_x, +sep, a.base_z), rot=face_desk
            ),
        )
        # 접촉 센서 (팔이 책상/파티션/서로에 부딪히면 episode 실패 처리)
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
                horizontal_aperture=20.955,
                clipping_range=(0.05, 30.0),
            ),
        )
        # 손목 카메라 2대 (3-view 전환, docs/threeview_camera_upgrade.md 3절).
        #   panda_hand 부착이라 로봇이 있어야 prim 경로가 성립함.
        #   near 0.02 = 손가락이 가까움 / aperture = 2 * f * tan(fov/2)
        if a.robots:
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

        # 쇼케이스 HD 카메라들 (시점 지정은 실행 script 가 runtime 에 함)
        n_hd = len([v for v in a.hd_views.split("|") if v.strip()])
        for i in range(n_hd):
            setattr(
                cfg,
                f"hd_cam_{i}",
                CameraCfg(
                    prim_path="{ENV_REGEX_NS}/hd_cam_" + str(i),
                    update_period=0.0,
                    height=a.hd_h,
                    width=a.hd_w,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=20.955,
                        clipping_range=(0.05, 30.0),
                    ),
                ),
            )

    src = "usd" if (a.use_usd and panel_usd) else "primitive"
    print(
        f"[scene] desk {a.desk_w:.2f}x{a.desk_d:.2f}x{a.desk_h:.2f} "
        f"portrait={a.portrait_side} riser={a.riser_h:.2f} "
        f"items={a.items} robots={bool(a.robots)}"
    )
    print(
        f"[scene] panels={src} {a.panel_inch:.0f}in -> {pw:.3f}x{ph:.3f}m "
        f"(scale {s:.3f}, origin={a.panel_origin}) "
        f"lift main={a.main_lift:.3f} sub={a.sub_lift:.3f}"
    )
    return cfg


# =========================================================================
# [B] scan 오버라이드 (구 office_scan_scene r4 의 델타 -- 합성 구조 보존)
# =========================================================================
import json  # noqa: E402
import sys  # noqa: E402

# [A] 의 원본 함수를 잡아 두고 아래에서 같은 이름으로 오버라이드함
_office_add_scene_args = add_scene_args
_office_build = build

# ---- 확정값 (08-25 실측/튜닝 종결) ---------------------------------------
# 이 블록이 벤치마크 프로토콜의 정본임. 값 변경 = PROTOCOL 버전업.
PROTOCOL = "office-scan-v1"
# 자산은 env 폴더에 동거함 (env4 = 코드+프로토콜+자산 자기완결)
SCAN_USD_DEFAULT = "/root/project/Isaac-franka/envs/office_scan/assets/take6_desk_hq.usd"
SCAN_SCALE_DEFAULT = 1.73  # 자산(칸막이 장축=1.4m 가정) -> 실물(벤치 2.4m)
DESK_COLOR_DEFAULT = "0.878,0.878,0.871"  # 스캔 상판 클린패치 중앙값 (sRGB)
PEDESTAL_COLOR_DEFAULT = "0.28,0.28,0.30"  # 로봇 받침대 (어두운 회색)
HOLDER_XY_DEFAULT = "0.16,-0.36"  # 사람 쪽 +6cm (구운 소품 간섭 회피)
REGION_DEFAULT = "0.12,0.26,-0.12,0.38"  # 키보드 회피 + 홀더 금지박스 정합
# 스캔 자산 프레임에서 칸막이 밑선의 y (상판 중앙 원점 기준, 자산 미터)
SCAN_PARTITION_Y = 0.35
# 스캔에 구워져 있는데 토글이 없어 cfg 에서 직접 떼는 부품 (office 의
# cfg 속성명 규약에 의존함 -- office 가 이름을 바꾸면 build 가 경고함)
_BAKED = (
    "stand_base_main",
    "stand_col_main",
    "panel_main",
    "riser_top",
    "riser_leg_0",
    "riser_leg_1",
)


def srgb_to_linear(rgb):
    """sRGB(0-1) -> linear. replica_to_usd 의 감마 규약과 동일한 EOTF 역변환.

    스캔 버텍스컬러는 이 변환을 거쳐 USD 에 들어가므로, 프로시저럴 면을
    같은 sRGB 값으로 맞추려면 같은 변환을 태워야 렌더에서 만남."""
    return tuple(
        (v / 12.92) if v <= 0.04045 else (((v + 0.055) / 1.055) ** 2.4) for v in rgb
    )


def add_scene_args(p):
    _office_add_scene_args(p)
    # 구운 가구 정책: "사용자 입력은 존중, 우리는 기본값만 변경".
    # 스캔에 구워진 가구를 기본 꺼짐으로 깔되, CLI 로 되살릴 수 있음
    # (스캔과 겹쳐 보이는 것은 사용자 선택). 연필꽂이(holder)는 태스크
    # 고정물이라 eval script 가 자기 set_defaults 로 다시 켬.
    # region 은 eval, item_region 은 preview 의 인자라 둘 다 걸어둠
    # (없는 쪽 키는 무해). 이 기본값 생존은 "eval 의 후속 set_defaults 가
    # 이 키들을 안 건드림" 에 의존함 -- build 가 effective 값을 로그로
    # 찍으므로 계약이 깨지면 즉시 드러남.
    p.set_defaults(
        partition=0,
        pc=0,
        keyboard=0,
        mousepad=0,
        mouse=0,
        portrait_side="none",
        holder=0,
        holder_xy=HOLDER_XY_DEFAULT,
        region=REGION_DEFAULT,
        item_region=REGION_DEFAULT,
    )
    g = p.add_argument_group("office-scan desk")
    g.add_argument("--scan-usd", default=SCAN_USD_DEFAULT)
    g.add_argument(
        "--scan-yaw",
        type=float,
        default=90.0,
        help="z rotation of the scan asset [deg]. 90 = scan partition (+y) "
        "faces the office back (-x). auto placement follows the sign; "
        "values other than +-90 need --scan-pos",
    )
    g.add_argument(
        "--scan-lift",
        type=float,
        default=0.002,
        help="scan mesh z offset above the procedural desk top (anti z-fight)",
    )
    g.add_argument(
        "--scan-scale",
        type=float,
        default=SCAN_SCALE_DEFAULT,
        help="uniform scale of the scan asset (= real_width / scanned_width). "
        "origin is the desk-top center, so the top stays at z=0",
    )
    g.add_argument(
        "--scan-pos",
        default="",
        help="scan asset x,y in the office frame. empty = auto: partition "
        "line to the desks' back edge, long axis centered on the full bench",
    )
    g.add_argument(
        "--light-dome",
        type=float,
        default=800.0,
        help="dome light intensity (office default 2500 is too hot for the "
        "scan -- its vertex colors already carry the capture lighting)",
    )
    g.add_argument(
        "--light-key",
        type=float,
        default=2000.0,
        help="key disk light intensity (office default 6000)",
    )
    g.add_argument(
        "--desk-color",
        default=DESK_COLOR_DEFAULT,
        help="sRGB 'r,g,b' (0-1) for the procedural desk slabs/legs peeking "
        "through scan gaps (default = sampled from the scan top). "
        "empty string = office white",
    )
    g.add_argument(
        "--pedestal-color",
        default=PEDESTAL_COLOR_DEFAULT,
        help="sRGB color for the robot pedestal (kept separate from "
        "--desk-color so the rig stays visually dark). empty = office color",
    )
    g.add_argument(
        "--desk-color-raw",
        action="store_true",
        help="pass --desk-color/--pedestal-color through without the "
        "srgb-to-linear conversion (office's own constants are authored "
        "raw; use if the converted tone renders too dark)",
    )


def _tone(color_str, raw):
    rgb = vec(color_str)
    return tuple(rgb) if raw else srgb_to_linear(rgb)


def _recolor(cfg, match_fn, lin, what):
    """조건에 맞는 책상 부품들의 diffuse 를 갈아끼우고 매칭 수를 검증함."""
    hit = 0
    for name in list(vars(cfg)):
        if match_fn(name):
            getattr(cfg, name).spawn.visual_material.diffuse_color = lin
            hit += 1
    if hit == 0:
        print(
            f"[office-scan] WARNING: no cfg parts matched for {what} "
            "(office naming may have changed)"
        )
    return hit


def build(a, with_camera=True):
    cfg = _office_build(a, with_camera)

    # 토글이 없는 구운 부품 제거. office 가 부품 이름을 바꾸면 조용히
    # no-op 이 되므로 (스캔과 프로시저럴 모니터가 겹쳐 보임) 개수 검증
    removed = [n for n in _BAKED if hasattr(cfg, n)]
    for n in removed:
        delattr(cfg, n)
    if len(removed) != len(_BAKED):
        missing = sorted(set(_BAKED) - set(removed))
        print(
            f"[office-scan] WARNING: baked parts not found in cfg: "
            f"{missing} (office naming changed? scan may double-render)"
        )

    # 조명: 스캔 버텍스컬러에는 촬영 조명이 구워져 있어 office 기본값이 과함
    cfg.light.spawn.intensity = a.light_dome
    cfg.key_light.spawn.intensity = a.light_key

    # 프로시저럴 책상 면 색 = 스캔 톤 (틈으로 비치는 부분의 위장색).
    # 로봇 받침대(태그 ROB)는 별도 어두운 톤 유지
    def is_pedestal(n):
        return n == "desk_top_ROB" or n.startswith("desk_leg_ROB_")

    def is_desk(n):
        return (
            n.startswith("desk_top_") or n.startswith("desk_leg_")
        ) and not is_pedestal(n)

    if a.desk_color.strip():
        lin = _tone(a.desk_color, a.desk_color_raw)
        _recolor(cfg, is_desk, lin, "--desk-color")
    if a.pedestal_color.strip() and a.robots:
        ped = _tone(a.pedestal_color, a.desk_color_raw)
        _recolor(cfg, is_pedestal, ped, "--pedestal-color")

    # 배치: 파티션 기준 자동 정렬. 스캔 칸막이 밑선(자산 y=+SCAN_PARTITION_Y,
    # yaw 회전 후 office -x 쪽)을 프로시저럴 책상들의 뒤 가장자리(-desk_d/2)에
    # 붙이고, 장축 중심은 wings 비대칭(왼 0.6/오른 0.4)의 벤치 중심에 둠.
    # 수식은 yaw 부호를 따라감 (+90/-90 외의 각도는 --scan-pos 수동)
    if a.scan_pos.strip():
        px, py = vec(a.scan_pos)
    else:
        s_yaw = math.sin(math.radians(a.scan_yaw))
        if abs(abs(s_yaw) - 1.0) > 0.01:
            print(
                "[office-scan] WARNING: auto --scan-pos assumes "
                "yaw=+-90; give --scan-pos manually for other angles"
            )
        wl = vec(a.wing_left)[0] if a.wing_left.strip() else 0.0
        wr = vec(a.wing_right)[0] if a.wing_right.strip() else 0.0
        px = -a.desk_d / 2 + SCAN_PARTITION_Y * a.scan_scale * s_yaw
        py = (wr - wl) / 2.0

    # 스캔 책상 (시각 전용 -- 충돌은 프로시저럴 상판이 담당).
    # 경로 오타는 Isaac 기동 수십 초 뒤 깊은 곳에서 죽으므로 조기 검사
    if not os.path.exists(a.scan_usd):
        sys.exit(f"[office-scan] ERROR: scan usd not found: {a.scan_usd}")
    sidecar = a.scan_usd + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as f:
                sc = json.load(f)
            print(
                f"[office-scan] scan sidecar: ver={sc.get('ver')} "
                f"date={sc.get('date')} scale={sc.get('scale')}"
            )
        except (OSError, ValueError):
            pass

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    cfg.scan_desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ScanDesk",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(px, py, a.scan_lift),
            rot=quat_axis("z", a.scan_yaw),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=a.scan_usd,
            scale=(a.scan_scale, a.scan_scale, a.scan_scale),
        ),
    )

    # 벤치마크 정체성 로그: 이 한 줄이 산출물 로그에 남아 "어느 프로토콜/
    # 자산/배치로 돌았나" 를 사후 추적 가능하게 함 (JSON 파싱 가능)
    eff = {
        "protocol": PROTOCOL,
        "scan_usd": a.scan_usd,
        "scan_scale": a.scan_scale,
        "scan_yaw": a.scan_yaw,
        "scan_pos": [round(px, 4), round(py, 4)],
        "scan_lift": a.scan_lift,
        "desk_color": a.desk_color,
        "desk_color_raw": int(bool(getattr(a, "desk_color_raw", False))),
        "pedestal_color": a.pedestal_color,
        "light_dome": a.light_dome,
        "light_key": a.light_key,
        "holder": int(getattr(a, "holder", 0)),
        "holder_xy": getattr(a, "holder_xy", "?"),
        # region 은 eval, item_region 은 preview 의 실제 인자 -- set_defaults
        # 가 양쪽 키를 다 심어 어느 쪽이 진짜인지 여기선 모르므로 둘 다 기록
        "region": getattr(a, "region", "?"),
        "item_region": getattr(a, "item_region", "?"),
        "furniture": {
            **{
                k: int(getattr(a, k, 0))
                for k in ("partition", "pc", "keyboard", "mousepad", "mouse")
            },
            # portrait_side 는 int 토글이 아닌 none|left|right 문자열
            "portrait_side": str(getattr(a, "portrait_side", "none")),
        },
    }
    print(f"[office-scan] effective: {json.dumps(eff)}")
    return cfg
