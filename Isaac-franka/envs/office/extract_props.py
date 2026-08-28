#!/usr/bin/env python
# [ver] extract_props.py 2026-08-10-r1  (ascii-only console/comments)
r"""
office.usd 안의 소품을 독립 USD 로 뽑아낸다 (에셋 다운로드 불필요).

Isaac 에셋의 Environments/Office/office.usd 에는 모니터/키보드/마우스/
서류트레이/펜꽂이 같은 사무실 소품이 전부 들어 있다. 다만 방 안에 놓인
상태라 (1) 방 좌표의 위치/회전이 박혀 있고 (2) 크기를 알 수 없다.

이 스크립트는 소품마다 작은 래퍼 USD 를 만들어
    - 원본 프림을 reference 로 물고 (재질/텍스처가 그대로 따라온다)
    - 방 좌표의 변환을 지우고 (identity)
    - xy 중심 + z 바닥이 원점에 오도록 재정렬한다
그 결과 우리 씬에서 일반 에셋처럼 UsdFileCfg 로 쓸 수 있다.

사용
    CUDA_VISIBLE_DEVICES=3 isaaclab.sh -p extract_props.py --headless
    CUDA_VISIBLE_DEVICES=3 isaaclab.sh -p extract_props.py --headless \
        --picks "monitor=/Root/SM_MonitorPC2,mouse=/Root/SM_MousePC10"

출력
    --out 아래에 <이름>.usd. 마지막에 실제 치수(회전 제거 후)를 표로 찍는다.
    그 수치를 보고 office_scene.py 의 배치를 확정한다.
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="extract office props into standalone USDs")
parser.add_argument(
    "--src",
    default="{ISAAC}/Environments/Office/office.usd",
    help="source scene usd; {ISAAC} expands",
)
parser.add_argument(
    "--out",
    default="/root/project/isaac_franka/envs/office/props",
    help="output dir for the extracted usd files",
)
parser.add_argument(
    "--picks",
    default="",
    help="name=/Root/PrimPath pairs, comma separated; empty = DEFAULT_PICKS",
)
parser.add_argument(
    "--keep-rot",
    type=int,
    default=0,
    help="1: keep the room rotation (debug) / 0: reset to identity",
)
# ---- 마커 래퍼 모드 (--marker-src 를 주면 이것만 실행) --------------------
parser.add_argument(
    "--marker-src",
    default="",
    help="prim path of a pen-like asset to wrap as the task marker: "
    "real-pen VISUAL scaled onto the validated invisible box collider. "
    "e.g. /Root/BP_PencilBox_filled_933/SM_MarkerBlue",
)
parser.add_argument(
    "--marker-name",
    default="marker_blue",
    help="output name under --out (props/<name>.usd)",
)
parser.add_argument(
    "--marker-fit",
    default="0.024,0.024,0.140",
    help="target visual bbox x,y,z [m]; must equal the scene box collider "
    "(office_scene MARKER_SQ, MARKER_SQ, MARKER_L)",
)
parser.add_argument(
    "--marker-prerot",
    default="",
    help="pre-rotation so the pen long axis becomes z (e.g. x90 / y90); "
    "run once without it and follow the printed hint",
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR  # noqa: E402

# 기본 추출 목록. (이름, office.usd 안의 프림, 원점 모드, 물리)
# 회전이 덜 섞인 인스턴스를 골랐다 (inspect_usd.py 의 bbox 로 판단).
#
# 원점 모드
#   bottom = xy 중심 + z 바닥 (책상 위에 z=0 으로 그냥 놓는 물건)
#   center = 기하 중심 (회전시켜 쓰는 물건. 모니터 패널을 세로로 피벗할 때
#            원점이 바닥이면 90도 돌렸을 때 옆으로 드러눕는다)
#
# 물리
#   none  = 시각 전용 (정적 배경 소품)
#   rigid = RigidBodyAPI + MassAPI + 메시 CollisionAPI 를 붙인다.
#           Isaac Lab 의 UsdFileCfg 는 rigid_props 를 "수정"만 하므로
#           USD 에 스키마가 없으면 "Failed to find a rigid body" 로 죽는다.
DEFAULT_PICKS = [
    # 정적 맥락 (팔이 닿지 않는 자리)
    # monitor_pc = 스탠드 포함, monitor_panel = 패널만(피벗 세로 배치에 쓴다)
    ("monitor_pc", "/Root/SM_MonitorPC2", "bottom", "none"),
    ("monitor_panel", "/Root/SM_Monitor2", "center", "none"),
    ("keyboard", "/Root/SM_KeyboardPC3", "bottom", "none"),
    ("mousepad", "/Root/SM_MousePad10", "bottom", "none"),
    ("lamp", "/Root/SM_Lamp3_647", "bottom", "none"),
    # 본체(타워)는 office.usd 에 없다. SM_PC* 는 전부 화면 일체형(아이맥형)이라
    # 책상 옆 타워로 쓸 수 없어서, 씬에서 프리미티브로 만든다.
    ("papertray", "/Root/SM_PaperTray10", "bottom", "none"),
    # 파지 후보 (그리퍼 개구 8cm 안에 드는지 치수로 판단할 것)
    # 마우스는 원본이 뒤집혀 있어 x 로 180도 돌려 세운다 (렌더에서 확인)
    ("mouse", "/Root/SM_MousePC10", "bottom", "rigid", "x180"),
    ("pencilcase", "/Root/SM_PencilCase3_439", "bottom", "rigid"),
    ("pencilcup", "/Root/SM_PencilCup3_258", "bottom", "rigid"),
    ("donutbox", "/Root/SM_DonutBox3", "bottom", "rigid"),
    ("dispenser", "/Root/SM_Dispenser3", "bottom", "rigid"),
    ("book", "/Root/SM_Book_10", "bottom", "rigid"),
    ("pen", "/Root/SM_Pen3_104", "bottom", "rigid"),
]


def expand(s):
    return s.replace("{ISAAC}", ISAAC_NUCLEUS_DIR).replace(
        "{NVIDIA}", NVIDIA_NUCLEUS_DIR
    )


def bbox_of(stage, prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return None
    return rng.GetMin(), rng.GetMax()


def deinstance(stage):
    """instanceable 프림을 전부 푼다.

    office.usd 의 소품은 instanceable 로 authoring 돼 있는데, 그대로 두면
    Isaac Lab 이 RigidBody/Collision API 를 못 붙인다 ("Could not perform
    modify_rigid_body_properties ... instanced prim paths" 경고 뒤
    "Failed to find a rigid body" 로 죽는다). 로컬 레이어에서 꺼 준다.
    """
    for _ in range(6):  # 상위를 풀면 하위가 새로 드러나므로 반복한다
        changed = False
        for p in stage.TraverseAll():
            if p.IsInstanceable():
                p.SetInstanceable(False)
                changed = True
        if not changed:
            break


def add_physics(stage, root_prim):
    """리지드 바디 스키마를 붙인다 (파지 대상 물건용).

    Isaac Lab 의 UsdFileCfg 는 rigid_props/mass_props 를 "수정"만 하므로,
    시각 메시만 있는 USD 를 그대로 쓰면 초기화에서
    "Failed to find a rigid body" 로 죽는다. 여기서 미리 authoring 한다.
    충돌은 볼록껍질 근사 (작은 문구류라 이 정도면 충분하다).
    """
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)
    n_mesh = 0
    for p in stage.TraverseAll():
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
            api = UsdPhysics.MeshCollisionAPI.Apply(p)
            api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            n_mesh += 1
    return n_mesh


# 씬의 box 마커와 동일해야 한다 (office_scene.py make_item 의 physics_material)
MARKER_FRICTION = (2.0, 1.8)  # static, dynamic (combine=max)


def extract_marker(src, meta):
    """펜류 에셋을 '실물 비주얼 + 보이지 않는 box 충돌체' 래퍼로 뽑는다.

    실물 펜/마커는 물리 파지가 안 된다 (단면이 얇아 손끝이 상판에 닿고,
    원형 단면은 26mm 원기둥도 미끄러졌다 -- 둘 다 실험으로 확정). 그래서
    충돌/질량은 검증된 box 마커 치수(--marker-fit) 그대로 두고, 비주얼만
    실물 에셋을 그 크기로 비균등 스케일해 입힌다.
    물리 스키마와 마찰 재질(2.0/1.8 combine=max)은 USD 에 직접 authoring
    한다 -- Isaac Lab 의 UsdFileCfg 는 physics_material 을 못 받는다.
    원점 = center (CuboidCfg 와 동일 규약), 긴 축 = z (--marker-prerot).
    충돌체는 purpose=guide 라 렌더에 안 보이지만 PhysX 는 충돌로 파싱한다.
    """
    from pxr import PhysxSchema, UsdShade

    fit = tuple(float(v) for v in args.marker_fit.split(","))
    out_path = os.path.join(args.out, f"{args.marker_name}.usd")
    if os.path.exists(out_path):
        os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, meta["up"])
    UsdGeom.SetStageMetersPerUnit(stage, meta["mpu"])

    # 2단 구조 (extract() 와 같은 이유: /Prop 의 변환은 Isaac Lab 이 덮어씀)
    root = UsdGeom.Xform.Define(stage, "/Prop")
    prim = root.GetPrim()
    stage.SetDefaultPrim(prim)
    geom = UsdGeom.Xform.Define(stage, "/Prop/Geom")
    geom.GetPrim().GetReferences().AddReference(Sdf.Reference(src, args.marker_src))
    deinstance(stage)

    # 1) 변환 op 는 한 번만 만들고 값만 갱신한다 (같은 op 를 다시 Add
    #    하면 USD 가 중복으로 거부한다). 나열 [T, S, R] = R -> S -> T 적용.
    #    초기값 항등이므로 첫 실측은 프리롯만 반영된 원본 치수다.
    geom.SetXformOpOrder([])
    t_op, t_vec = add_vec_op(geom, "translate")
    s_op, s_vec = add_vec_op(geom, "scale")
    apply_prerot(geom, args.marker_prerot)
    t_op.Set(t_vec(0.0, 0.0, 0.0))
    s_op.Set(s_vec(1.0, 1.0, 1.0))
    box = bbox_of(stage, prim)
    if box is None:
        print(f"  [x] marker: empty bbox ({args.marker_src})")
        print("      try another prim (e.g. a top-level SM_Felt__Pen_* path)")
        return
    lo, hi = box
    size0 = tuple(hi[i] - lo[i] for i in range(3))
    print(
        f"  [marker] native size (after prerot "
        f"'{args.marker_prerot or 'none'}') = "
        f"({size0[0]:.4f}, {size0[1]:.4f}, {size0[2]:.4f})"
    )
    long_axis = max(range(3), key=lambda i: size0[i])
    if long_axis != 2:
        hint = "y90" if long_axis == 0 else "x90"
        print(
            f"  [!] long axis is '{'xyz'[long_axis]}', not z -> rerun with "
            f"--marker-prerot {hint} (do not use this output)"
        )

    # 2) 비주얼을 fit 치수로 비균등 스케일 + center 원점 정렬
    #    (스케일은 prerot 이후 프레임에서 쟀으므로 R 다음에 S 순서가 맞다)
    s_op.Set(s_vec(*(fit[i] / max(size0[i], 1e-6) for i in range(3))))
    box = bbox_of(stage, prim)
    lo, hi = box
    t_op.Set(t_vec(*(-(lo[i] + hi[i]) / 2.0 for i in range(3))))
    box = bbox_of(stage, prim)
    lo, hi = box
    size = tuple(hi[i] - lo[i] for i in range(3))

    # 3) 물리: 강체 + 보이지 않는 box 충돌체. 비주얼 메시에는 충돌을 안
    #    붙인다 (볼록껍질이면 둥근 단면이 되살아나 미끄러짐이 재발한다)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    cube = UsdGeom.Cube.Define(stage, "/Prop/Collider")
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*fit))
    cube.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # 4) 마찰 재질을 충돌체에 physics 바인딩 (씬 box 마커와 동일 값)
    mat = UsdShade.Material.Define(stage, "/Prop/PhysMat")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(MARKER_FRICTION[0])
    pm.CreateDynamicFrictionAttr().Set(MARKER_FRICTION[1])
    pm.CreateRestitutionAttr().Set(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set("max")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    stage.GetRootLayer().Save()
    print(f"  [ok] marker wrapper -> {out_path}")
    print(
        f"       visual bbox ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}) "
        f"z[{lo[2]:+.4f},{hi[2]:+.4f}] (want: fit size, center origin)"
    )
    print(
        f"       collider box {fit} + friction {MARKER_FRICTION} combine=max"
    )


def add_vec_op(geom, kind):
    """translate/scale op 를 기존 attr 의 정밀도에 맞춰 추가한다.

    reference 로 끌려온 원본 프림에 xformOp:scale 등이 double3 로 이미
    authoring 돼 있으면, 기본 정밀도(float)로 Add 할 때 Tf 예외가 난다
    (SM_MarkerBlue 실측). (op, 벡터 생성자) 를 돌려준다."""
    attr = geom.GetPrim().GetAttribute("xformOp:" + kind)
    if attr:
        dbl = attr.GetTypeName() == Sdf.ValueTypeNames.Double3
    else:
        dbl = kind == "translate"  # USD 기본: translate=double, scale=float
    prec = UsdGeom.XformOp.PrecisionDouble if dbl else UsdGeom.XformOp.PrecisionFloat
    fn = geom.AddTranslateOp if kind == "translate" else geom.AddScaleOp
    return fn(prec), (Gf.Vec3d if dbl else Gf.Vec3f)


def apply_prerot(geom, spec):
    """소품별 사전 회전 ('x180', 'z90' 등). 원본이 뒤집혀 authoring 된 경우
    방 좌표 변환을 지우면 그대로 드러나므로 여기서 바로잡는다.
    회전을 먼저 넣고 그 뒤에 크기/원점을 재야 정렬이 어긋나지 않는다.
    정밀도는 기존 attr 에 맞춘다 (add_vec_op 와 같은 이유)."""
    if not spec:
        return
    axis, deg = spec[0].lower(), float(spec[1:])
    attr = geom.GetPrim().GetAttribute("xformOp:rotate" + axis.upper())
    dbl = bool(attr) and attr.GetTypeName() == Sdf.ValueTypeNames.Double
    prec = UsdGeom.XformOp.PrecisionDouble if dbl else UsdGeom.XformOp.PrecisionFloat
    op = {"x": geom.AddRotateXOp, "y": geom.AddRotateYOp, "z": geom.AddRotateZOp}[axis]
    op(prec).Set(deg)


def extract(src, name, prim_path, out_dir, meta, mode, physics, prerot=""):
    """소품 하나를 래퍼 USD 로 저장하고 (치수, 경로) 를 돌려준다."""
    out_path = os.path.join(out_dir, f"{name}.usd")
    if os.path.exists(out_path):
        os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)
    # 원본과 단위/업축을 맞춘다 (안 맞으면 100배 크기로 들어온다)
    UsdGeom.SetStageUpAxis(stage, meta["up"])
    UsdGeom.SetStageMetersPerUnit(stage, meta["mpu"])

    # 구조를 2단으로 나눈다.
    #   /Prop       변환 없이 비워 둔다. Isaac Lab 이 스폰할 때 자기 translate/
    #               orient/scale 을 여기에 authoring 하기 때문이다.
    #   /Prop/Geom  참조 + 우리 정렬 변환. 한 단계 아래라 덮어써지지 않는다.
    # 정렬을 /Prop 에 두면 Isaac Lab 이 그것을 지우고, 패널이 원본 로컬 원점
    # (화면 구석 등) 기준으로 놓여 회전 시 엉뚱한 곳에 붙는다 (실측 확인).
    root = UsdGeom.Xform.Define(stage, "/Prop")
    prim = root.GetPrim()
    stage.SetDefaultPrim(prim)
    geom = UsdGeom.Xform.Define(stage, "/Prop/Geom")
    geom.GetPrim().GetReferences().AddReference(Sdf.Reference(src, prim_path))
    deinstance(stage)

    # 1) 방 좌표의 변환 제거 + 사전 회전.
    #    로컬 레이어의 xformOpOrder 가 reference 를 이긴다
    if not args.keep_rot:
        geom.SetXformOpOrder([])
        apply_prerot(geom, prerot)

    box = bbox_of(stage, prim)
    if box is None:
        print(f"  [x] {name}: empty bbox ({prim_path})")
        return None
    lo, hi = box

    # 2) 원점 정렬. bottom = 책상 위에 z=0 으로 놓는 물건,
    #    center = 회전시켜 쓰는 물건 (패널을 세로로 피벗해도 안 드러눕는다)
    #    USD 는 나열 순서의 뒤쪽이 먼저 적용되므로 [translate, rotate] 로 둔다
    zshift = -(lo[2] + hi[2]) / 2.0 if mode == "center" else -lo[2]
    shift = Gf.Vec3d(-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, zshift)
    geom.SetXformOpOrder([])
    geom.AddTranslateOp().Set(shift)
    apply_prerot(geom, prerot)

    box = bbox_of(stage, prim)
    lo, hi = box
    size = (hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])

    # 3) 물리 스키마 (파지 대상만). 크기를 다 잰 뒤에 붙인다
    if physics == "rigid":
        n_mesh = add_physics(stage, prim)
        if n_mesh == 0:
            print(f"  [!] {name}: no mesh found -> collision may be missing")

    stage.GetRootLayer().Save()
    # zmin/zmax 를 같이 돌려준다. 원점 규약(bottom 이면 zmin=0,
    # center 면 zmin=-h/2)을 눈으로 확인할 수 있어야 배치 버그를 막는다
    return size, out_path, float(lo[2]), float(hi[2])


def main():
    src = expand(args.src)
    picks = DEFAULT_PICKS
    if args.picks.strip():
        picks = []
        for tok in args.picks.split(","):
            if "=" in tok:
                n, p = tok.split("=", 1)
                # name=/Root/Prim[:origin[:physics]]
                mode, phys = "bottom", "none"
                parts = p.split(":")
                p = parts[0]
                if len(parts) > 1:
                    mode = parts[1]
                if len(parts) > 2:
                    phys = parts[2]
                prerot = parts[3] if len(parts) > 3 else ""
                picks.append(
                    (n.strip(), p.strip(), mode.strip(), phys.strip(), prerot.strip())
                )

    src_stage = Usd.Stage.Open(src)
    if src_stage is None:
        print(f"[x] cannot open source: {src}")
        return
    meta = {
        "up": UsdGeom.GetStageUpAxis(src_stage),
        "mpu": UsdGeom.GetStageMetersPerUnit(src_stage),
    }
    print(f"[src] {src}")
    print(f"[src] upAxis={meta['up']} metersPerUnit={meta['mpu']}")

    os.makedirs(args.out, exist_ok=True)

    # 마커 래퍼 모드: 이것만 하고 끝낸다 (기존 추출물은 건드리지 않는다)
    if args.marker_src.strip():
        p = src_stage.GetPrimAtPath(args.marker_src.strip())
        if p is None or not p.IsValid():
            print(f"  [x] marker prim not found: {args.marker_src}")
            print("      check the path with inspect_usd.py")
            return
        extract_marker(src, meta)
        return
    rows = []
    # (이름, 프림, 원점, 물리[, 사전회전]) -- 사전회전은 생략 가능
    picks = [tuple(t) + ("",) * (5 - len(t)) for t in picks]
    for name, prim_path, mode, phys, prerot in picks:
        if src_stage.GetPrimAtPath(prim_path) is None or not src_stage.GetPrimAtPath(
            prim_path
        ).IsValid():
            print(f"  [x] {name}: prim not found {prim_path}")
            continue
        got = extract(src, name, prim_path, args.out, meta, mode, phys, prerot)
        if got:
            size, path, zlo, zhi = got
            rows.append((name, size, path, mode, phys, zlo, zhi))
            print(
                f"  [ok] {name:12s} {os.path.basename(path):20s} "
                f"origin={mode} physics={phys}"
            )

    print("\n" + "=" * 96)
    print(
        f"{'name':12s} {'size x,y,z [m]':26s} {'z range':18s} "
        f"{'origin':7s} {'phys':6s} {'grasp?':6s} note"
    )
    print("=" * 96)
    for name, s, _, mode, phys, zlo, zhi in rows:
        w = min(s[0], s[1])  # 짧은 변으로 물면 된다
        graspable = "yes" if w < 0.078 else "NO"
        note = "" if w < 0.078 else f"min width {w:.3f} > gripper 0.078"
        # z range 로 원점 규약을 눈으로 확인한다 (bottom -> 0..h, center -> -h/2..h/2)
        print(
            f"{name:12s} ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}){'':6s} "
            f"[{zlo:+.3f},{zhi:+.3f}]{'':3s} {mode:7s} {phys:6s} "
            f"{graspable:6s} {note}"
        )
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    # 주의: finally 의 os._exit 는 예외 트레이스백 출력을 삼킨다 --
    # 여기서 직접 찍고 rc 를 넘겨야 실패가 조용히 사라지지 않는다
    rc = 0
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
