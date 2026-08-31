# ======================================
# File: office_scene.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] office_scene.py 2026-08-10-r3  (ascii-only console/comments)
r"""
사무실 책상 scene -- 연구실 책상 재현 (두 번째 benchmark 환경).

1단계는 책상만 구성 (로봇 없음). 로봇은 나중에 --robots 1 로 부착.

좌표계 (책상 중심 기준. 로봇 workcell 은 나중에 사람 자리에 얹음)
    z = 0        책상 상판. 바닥은 z = -desk_h
    원점         책상 중심
    +x           사람이 앉는 쪽 (카메라가 여기서 -x 를 봄)
    사람 기준    왼쪽 = -y, 오른쪽 = +y

    배치
        x = -0.25   먼 가장자리: 세로 모니터(-y) + 메인 모니터(y=0, 10cm 단상)
        x = +0.10   키보드(y=0), 마우스패드+마우스(+y)
        x = +0.62   나중에 로봇 base 가 올 자리 (모니터까지 0.87m
                    = Franka reach 0.855 밖이라 팔이 모니터를 못 건드림)

모니터
    pivot(세로) 모니터는 스탠드까지 돌리면 안 되므로 "패널 + 스탠드" 로
    분리 구성. 세로는 패널만 화면 법선(x축) 기준 90도 회전.
    메인은 그 아래에 10cm 단상 배치.

asset
    props/ 에 extract_props.py 로 뽑아 둔 USD 가 있으면 그것을 쓰고,
    없으면 같은 치수의 primitive 로 대체 (둘 다 동작함).
    원본은 Isaac asset 의 Environments/Office/office.usd.

주의
    최상단에서 isaaclab import 금지. AppLauncher 기동 전에는
    omni.client 가 없어 import 자체가 실패함. isaaclab 심볼은 전부
    build() 안에서 지연 import.
"""

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
# 안 쓰는 마커를 치워두는 자리 (리치/화각 밖. 바닥으로 떨어져 화면에서 사라짐)
PARK = (2.5, 2.5, 0.5)
MARKER_R = 0.013
MARKER_SQ = 0.024  # 각단면(box) mode 의 한 변. 면 파지라 원기둥보다 훨씬 안정
MARKER_L = 0.140
# build() 에서 --marker-shape 값으로 설정됨 (item_half_h 등이 참조)
MARKER_SHAPE = "box"
MARKER_COLORS = [
    (0.80, 0.15, 0.15),
    (0.15, 0.35, 0.80),
    (0.15, 0.60, 0.25),
    (0.85, 0.65, 0.10),
]
MAX_ITEMS = len(MARKER_COLORS)


# 0. Marker 배치 helper -------------------------------------------------
def item_name(i):
    """marker1, marker2, ... marker primitive 이름 반환"""
    # item_name(1) -> marker1
    return f"marker{i}"


def item_half_h(i):
    """눕혀 놓았을 때의 중심 높이 [m]. usd mode 도 충돌체는 box 라 동일."""
    # 단면 기준으로 생각했을때의 마커의 중심 높이
    # 마커 두께 MARKER_SQ / 2 -> 중심 높이

    if MARKER_SHAPE == "cyl":
        h = MARKER_R
    else:
        h = MARKER_SQ / 2.0

    # 마커를 배치할때 상판과 겹친 채 시작하는 것을 막기 위해 0.3 [mm] 올려서 배치
    return h + 0.0003


def item_grasp_w(i):
    """gripper 가 무는 폭 [m]. 8cm 초과 금지. usd = box 충돌체."""
    # 마커 두께 MARKER_SQ

    if MARKER_SHAPE == "cyl":
        return 2.0 * MARKER_R
    else:
        return MARKER_SQ


# 1. helper -------------------------------------------------
def vec(s):
    """cli로 받는 문자열 설정을 튜플로 변환"""
    # vec("0.16,-0.36") -> (0.16, -0.36)
    return tuple(float(x) for x in s.split(","))


def prop_usd(name):
    """props/ 에 추출해 둔 USD 경로. 없으면 None (primitive 로 대체)."""
    # props/ 에 extract_props.py 로 뽑아둔 USD 가 있으면 경로를, 없으면 None
    p = os.path.join(PROPS_DIR, f"{name}.usd")
    return p if os.path.exists(p) else None


def quat_mul(q1, q2):
    """두 quaternion 곱 (w,x,y,z) 반환"""
    # q1*q2 = q2 를 먼저 적용하고 q1 을 나중에 적용 (world 기준. 왼쪽이 나중)
    # e.g. quat_mul(quat_axis("z", yaw), quat_axis("y", 90)) = 눕힌 뒤 yaw 회전
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
    # e.g. yaw(z)으로 deg 만큼의 회전 -> quaternion으로 반환함

    import math

    h = math.radians(deg) / 2.0
    c, s = math.cos(h), math.sin(h)
    return {
        "x": (c, s, 0.0, 0.0),
        "y": (c, 0.0, s, 0.0),
        "z": (c, 0.0, 0.0, s),
    }[axis]


# 2. arguments (argparse) ------------------------------------------------
def add_scene_args(p):
    """scene 인자. 앱 기동 전에 호출해도 안전함 (isaaclab 불필요)."""
    g = p.add_argument_group("desk")
    g.add_argument(
        "--desk-w",
        type=float,
        default=DESK_W,
        help="desk width along y",
    )
    g.add_argument(
        "--desk-d",
        type=float,
        default=DESK_D,
        help="desk depth along x",
    )
    g.add_argument(
        "--desk-h",
        type=float,
        default=DESK_H,
        help="floor to top [m]",
    )
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
    g.add_argument(
        "--keyboard",
        type=int,
        default=1,
    )
    g.add_argument(
        "--mousepad",
        type=int,
        default=1,
    )
    g.add_argument(
        "--mouse",
        type=int,
        default=1,
    )
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
    g.add_argument(
        "--base-sep",
        type=float,
        default=ROBOT_BASE_SEP,
    )
    g.add_argument(
        "--base-x",
        type=float,
        default=ROBOT_BASE_X,
    )
    g.add_argument(
        "--base-z",
        type=float,
        default=ROBOT_BASE_Z,
        help="robot base height relative to the desk top (0); negative = "
        "below the top. A pedestal is built from the floor up to it",
    )

    g = p.add_argument_group("render")
    g.add_argument(
        "--num-envs",
        type=int,
        default=1,
    )
    g.add_argument(
        "--cam-w",
        type=int,
        default=1280,
    )
    g.add_argument(
        "--cam-h",
        type=int,
        default=720,
    )
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
    g.add_argument(
        "--hd-w",
        type=int,
        default=1280,
    )
    g.add_argument(
        "--hd-h",
        type=int,
        default=720,
    )


# 3. scene 조립 ---------------------------------------------------------
def build(a, with_camera=True):
    """scene cfg 생성. AppLauncher 기동 이후에만 호출할 것."""
    # a: argparse로 받은 인자
    # a.xxx 로 접근하여 설정을 주입함
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass

    # ./project-DT/IsaacLab/source/isaaclab/isaaclab/scene/ 아래에 있는
    # interactive_scene_cfg.py -> InteractiveSceneCfg (설정 뼈대)
    # interactive_scene.py     -> InteractiveScene (cfg 를 실체화)

    @configclass
    class OfficeDeskSceneCfg(InteractiveSceneCfg):
        # InteractiveSceneCfg를 상속해 office scene 부품 필드를 추가
        # 월드 전체의 바닥과 조명을 정의
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

    # 형광등 느낌의 보조 조명 (전체 조명만 있으면 너무 단조로움)
    cfg.key_light = AssetBaseCfg(
        prim_path="/World/key_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.0, 1.6)),
        spawn=sim_utils.DiskLightCfg(intensity=6000.0, radius=0.35),
    )

    # helper functions -------------------------------------------------
    # 특정 크기의 box를 생성하는 helper (책상 다리, 파티션, 연필꽂이 벽 등)
    # 정적 오브젝트: 옮기거나 잡을 수 없음. 충돌은 있어서 물체가 얹히거나
    # 부딪힐 수는 있음 (움직이는 물체는 RigidObjectCfg 로 따로 만듦)
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

    # 배경을 위한 정적 오브젝트용 helper
    def static_usd(name, usd, pos, rot=(1.0, 0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        # 정적 소품은 시각 맥락이 목적이라 충돌을 붙이지 않음. 붙이면 mesh
        # 볼록껍질이 생겨 나중에 팔이 근처를 지날 때 불필요하게 걸림.
        return AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + name,
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
            spawn=sim_utils.UsdFileCfg(usd_path=usd, scale=scale),
        )

    # usd 파일이 있다면 그 파일을 사용
    # 없다면 직접 만든 primitive box를 사용
    def pick_usd(name):
        return prop_usd(name) if a.use_usd else None

    # 책상 (가운데 + 좌우 보조 = 눕힌 디귿)
    # 상판(마찰 재질이 필요해 직접 조립) + 다리 4개(static_box)로 책상 하나
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

    # scene 조립 구간
    # 3-1. 책상 배치
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

    # 3-2. 파티션 배치 (회색 칸막이. 뒤 + 좌우를 둘러쌈)
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

    # 3-3. 모니터 정의 (받침대 + 기둥 + 그 앞면에 붙는 패널)
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

    # 3-4. 메인 단상
    # 눕힌 디귿(아치). 상판 + 양옆 다리, 너비는 모니터 가로폭과 같음
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
    # 3-5. 모니터 배치 (정면 모니터. 단상 위에 섬)
    monitor("main", 0.0, portrait=False, base_z=a.riser_h, lift=a.main_lift)

    # 3-6. 모니터 배치 (세로 모니터)
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

    # 3-7. 연필꽂이 (왼쪽 모니터 앞. 바닥판 + 벽 4장)
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

    # 3-8. 본체 (오른쪽 파티션에 붙이고, 긴 면을 벽에 나란히)
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

    # 3-9. 키보드
    if a.keyboard:
        kb = pick_usd("keyboard")
        if kb:
            cfg.keyboard = static_usd("Keyboard", kb, (KEYBOARD_X, 0.0, 0.0))
        else:
            kx, ky, kz = KEYBOARD_SIZE
            cfg.keyboard = static_box(
                "Keyboard", (KEYBOARD_X, 0.0, kz / 2), (kx, ky, kz), KEYBOARD_COLOR
            )

    # 3-10. 마우스패드
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

    # 3-11. 마우스
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

    # 3-12. 마커 (정리 대상)
    # 주어진 갯수 만큼 PARK 위치에 생성만 함 -- 책상 위로 옮기는 것은
    # episode 마다 생성기/평가기의 reset 담당 (scene 재생성 없이 재배치)
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

        # marker 하나를 만드는 helper
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

    # 3-13. 로봇 (사람 자리에서 책상을 마주봄)
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

    # 3-14. 카메라 (ego 1 + wrist 2 = 3-view)
    # wrist-view 는 panda_hand 에 부착하는 형태라 로봇이 먼저 있어야 함
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

    # 3-15. scene 요약 로그
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
