#!/usr/bin/env python
# ======================================
# File: office_scan_scene.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] office_scan_scene.py 2026-08-28-r4  (ascii-only console/comments)
# r4: 스캔 자산 위치 datasets/ -> envs/office_scan/assets/ (env 자산 동거;
#     파일 내용 동일이라 프로토콜 office-scan-v1 유지)
# r3: effective JSON 에 portrait_side / desk_color_raw 기록 (로깅 보강만 --
#     값 불변이라 프로토콜은 office-scan-v1 유지)
# r2: 프로토콜 상수 명문화, 구운 가구 정책 2원화(기본값+delattr), 색 변환
#     헬퍼 통일, scan-pos 자동배치의 yaw 부호 반영, 자산/제거 검증 가드,
#     effective 설정 JSON 로그 (벤치마크 산출물 정체성 기록)
"""office 씬의 책상만 스캔 USD 로 갈아끼운 변형 (4번째 환경 = 실사 배경).

office_scene.py 는 무수정 엔진으로 사용함 (사용자 지시 -- 새 파일로만).
이 모듈의 역할:

  1. 스캔 메시에 이미 구워져 있는 가구를 프로시저럴에서 뺌
       기본값으로 꺼지는 것 (CLI 로 되살릴 수 있음):
           파티션 / 키보드 / 마우스(패드) / PC / 세로 모니터 / 연필꽂이
           (연필꽂이는 태스크 고정물이라 eval 이 자기 기본값으로 되켬)
       토글이 없어 cfg 에서 떼는 것 (_BAKED):
           메인 모니터 3피스 + 단상 3피스
       남기는 것: 가운데+보조 책상 상판/다리 -- 벤치 전체의 "충돌" 담당
  2. 스캔 책상 USD 를 얹음 (시각 전용, 충돌 없음).
     스캔 z0 = office z0 = 상판이라 두 면이 자동 일치, z-fighting 은
     --scan-lift (기본 2mm) 로 회피. 배치는 칸막이 기준 자동 정렬.
  3. 축 변환: 스캔 프레임은 칸막이가 +y, office 는 뒤쪽이 -x
     -> z 축 +90도 회전 (자동 배치는 yaw 부호를 따라감)

태스크(마커/연필꽂이)와 로봇 리그, 카메라, 성공 판정은 office 그대로임.

확정 상수(아래 PROTOCOL 블록)가 벤치마크 정의의 정본임 -- 값을 바꾸면
PROTOCOL 문자열을 올림 (러너는 이 값들을 전달하지 않음).
정본 문서: gr00t_isaacsim/docs/gsrecon_pipeline.md
"""

import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "office"))
import office_scene as osc  # noqa: E402

# 자주 쓰는 심볼의 grep 가시성용 재노출 -- 없어도 아래 __getattr__ 가
# 위임하므로 동작에는 필요 없음 (문서화 목적)
EGO_EYE = osc.EGO_EYE
EGO_TARGET = osc.EGO_TARGET


def __getattr__(name):
    """여기 없는 심볼은 전부 office_scene 으로 위임함 (PEP 562).

    eval/preview 가 osc.MAX_ITEMS, osc.quat_axis 등 다수를 참조하므로
    이 모듈이 office_scene 의 완전한 대체물이 됨. 주의: 없는 이름의
    AttributeError 는 office_scene 명의로 뜸 (디버깅 시 참고)."""
    return getattr(osc, name)


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
    osc.add_scene_args(p)
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
    rgb = osc.vec(color_str)
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
    cfg = osc.build(a, with_camera)

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
        px, py = osc.vec(a.scan_pos)
    else:
        s_yaw = math.sin(math.radians(a.scan_yaw))
        if abs(abs(s_yaw) - 1.0) > 0.01:
            print(
                "[office-scan] WARNING: auto --scan-pos assumes "
                "yaw=+-90; give --scan-pos manually for other angles"
            )
        wl = osc.vec(a.wing_left)[0] if a.wing_left.strip() else 0.0
        wr = osc.vec(a.wing_right)[0] if a.wing_right.strip() else 0.0
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
            rot=osc.quat_axis("z", a.scan_yaw),
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
