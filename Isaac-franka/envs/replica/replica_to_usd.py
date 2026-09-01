#!/usr/bin/env python3
# ======================================
# File: replica_to_usd.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] replica_to_usd.py 2026-08-26-r3  (ascii-only console/comments)
# r3: simulation_app.close() 제거 (헤드리스에서 반환 안 하는 경우가 있어
#     os._exit(0) 도달 전에 좀비화 -- 형제 스크립트들과 동일 정책으로 통일)
# r2: --color-gamma 추가, 기본 srgb (PLY 8bit 색 = sRGB 인데 USD 는 linear
#     해석 -> 무변환이면 물빠진 색). 08-25 이전에 변환된 .usd 는 linear
#     passthrough 판이므로 색을 맞추려면 재변환하고, 비트 동일 재현이
#     필요하면 --color-gamma linear 를 쓸 것.
"""Replica 스캔 mesh.ply 를 Isaac 용 정적 배경 USD 로 변환한다.

Replica (facebookresearch/Replica-Dataset) 의 scene mesh 는 방 전체가 한
덩어리로 융합된 스캔 mesh 임. 이 script 는 그 ply 를 읽어 다음을 수행함:

  1. binary little endian ply parsing (외부 의존성 없이 numpy 만 사용)
  2. 업축 정렬 (y-up 스캔이면 z-up 으로 회전) - 좌표에 직접 구움
  3. 바닥 재원점 (바닥 높이 백분위 추정 -> z=0), xy 중심 재원점
  4. vertex color -> displayColor primvar + UsdPreviewSurface material
     (RTX 가 material 없는 displayColor 를 무시해도 보이게 안전판)
  5. 2단 구조 /Bg(빈 Xform, 기본 prim) + /Bg/Geom(mesh) - Isaac Lab 이
     spawn 때 최상위 변환을 덮어쓰는 함정 대응 (extract_props 와 동일)
  6. 선택: 정적 삼각 mesh 충돌 (--physics static). 기본은 시각 전용
     (책상 rig 에서 벽까지 멀어 충돌이 불필요하고 cooking 비용만 듦)

공용 계약: office_scan 의 take 자산 변환도 이 변환기를 씀
(scan/postprocess_mesh.py 다음 단계; --up z --floor-pct -1 --no-recenter).
옵션 기본값/색 규약(sRGB->linear)을 바꾸면 replica 와 office_scan 양쪽
자산 계보가 함께 영향받음 -- ASSETS.md 대조 필수.

실행 (container 안. pxr 는 kit app 이 떠야 잡히므로 --headless 필수,
extract_props.py 와 동일 pattern):
  docker exec -u 0 gr00t_dt bash -lc "umask 000 && \
    CUDA_VISIBLE_DEVICES=5 /root/project/IsaacLab/isaaclab.sh -p \
    /root/project/Isaac-franka/envs/replica/replica_to_usd.py --headless \
    --ply /root/project/datasets/replica/office_0/mesh.ply \
    --out /root/project/datasets/replica/office_0.usd"

주의: container 는 NAS(/data1)를 못 보므로 ply 는 홈(~/project) 경유로 복사.
"""

import argparse
import os
import struct
import sys

import numpy as np

# ply type 이름 -> numpy dtype (binary little endian)
PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def parse_header(f):
    """ply header 를 parse 해 (형식, 원소 목록) 을 돌려줌.

    원소 = (이름, 개수, [scalar 속성 (type, 이름)] 또는 list 속성 1개).
    """
    magic = f.readline().strip()
    if magic != b"ply":
        raise SystemExit("ERROR: not a ply file")
    fmt = None
    elements = (
        []
    )  # (name, count, props) props: [("scalar", np_t, name)] or [("list", cnt_t, idx_t, name)]
    while True:
        line = f.readline()
        if not line:
            raise SystemExit("ERROR: unexpected EOF in ply header")
        tok = line.decode("ascii", "replace").strip().split()
        if not tok:
            continue
        if tok[0] == "comment":
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            elements.append([tok[1], int(tok[2]), []])
        elif tok[0] == "property":
            if tok[1] == "list":
                elements[-1][2].append(
                    ("list", PLY_TYPES[tok[2]], PLY_TYPES[tok[3]], tok[4])
                )
            else:
                elements[-1][2].append(("scalar", PLY_TYPES[tok[1]], tok[2]))
        elif tok[0] == "end_header":
            break
    if fmt != "binary_little_endian":
        raise SystemExit(
            f"ERROR: unsupported ply format '{fmt}' (need binary_little_endian)"
        )
    return elements


def read_scalar_element(f, count, props):
    """scalar 속성만 있는 원소 block 을 구조체 배열로 읽음."""
    if any(p[0] == "list" for p in props):
        raise SystemExit(
            "ERROR: list property inside a scalar element is not supported"
        )
    dt = np.dtype([(name, "<" + t) for kind, t, name in props])
    arr = np.fromfile(f, dtype=dt, count=count)
    if arr.shape[0] != count:
        raise SystemExit("ERROR: ply vertex block truncated")
    return arr


def read_face_element(f, count, props):
    """list 속성(면 index) 원소를 읽음. 균일 꼭짓점 수는 vectorized fast path.

    반환: (face_counts int32[N], face_indices int32[sum]).
    """
    lists = [p for p in props if p[0] == "list"]
    if len(props) != 1 or len(lists) != 1:
        raise SystemExit("ERROR: face element with extra properties is not supported")
    _, cnt_t, idx_t, _ = lists[0]
    cnt_sz = np.dtype(cnt_t).itemsize
    idx_sz = np.dtype(idx_t).itemsize

    pos = f.tell()
    head = np.fromfile(f, dtype="<" + cnt_t, count=1)
    if head.shape[0] != 1:
        raise SystemExit("ERROR: ply face block empty")
    k = int(head[0])
    f.seek(pos)

    # fast path: 모든 면이 같은 꼭짓점 수라고 가정하고 통째로 읽음
    rec = np.dtype([("n", "<" + cnt_t), ("idx", "<" + idx_t, (k,))])
    arr = np.fromfile(f, dtype=rec, count=count)
    if arr.shape[0] == count and bool((arr["n"] == k).all()):
        counts = np.full(count, k, dtype=np.int32)
        indices = arr["idx"].astype(np.int32).reshape(-1)
        return counts, indices

    # fallback: 면마다 꼭짓점 수가 다른 드문 경우 (느리지만 안전)
    f.seek(pos)
    counts = np.empty(count, dtype=np.int32)
    chunks = []
    for i in range(count):
        c = struct.unpack("<" + np.dtype(cnt_t).char, f.read(cnt_sz))[0]
        counts[i] = c
        chunks.append(
            np.frombuffer(f.read(idx_sz * c), dtype="<" + idx_t).astype(np.int32)
        )
    return counts, np.concatenate(chunks)


def read_ply(path):
    """ply 를 읽어 (verts float64[N,3], colors uint8[N,3]|None, counts, indices)."""
    with open(path, "rb") as f:
        elements = parse_header(f)
        verts = colors = counts = indices = None
        for name, count, props in elements:
            if name == "vertex":
                arr = read_scalar_element(f, count, props)
                verts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(
                    np.float64
                )
                names = arr.dtype.names
                if "red" in names and "green" in names and "blue" in names:
                    colors = np.stack(
                        [arr["red"], arr["green"], arr["blue"]], axis=1
                    ).astype(np.uint8)
            elif name == "face":
                counts, indices = read_face_element(f, count, props)
            else:
                # 관심 없는 원소 (edge 등): scalar 면 건너뛰고, list 면 포기
                if any(p[0] == "list" for p in props):
                    raise SystemExit(f"ERROR: cannot skip list element '{name}'")
                sz = sum(np.dtype(t).itemsize for kind, t, n in props)
                f.seek(sz * count, 1)
    if verts is None or counts is None:
        raise SystemExit("ERROR: ply missing vertex or face element")
    return verts, colors, counts, indices


def pick_up_axis(verts, mode):
    """업축 결정. auto 는 bbox 가 가장 짧은 축(실내는 높이가 최소)을 고름."""
    if mode in ("y", "z"):
        return mode
    ext = verts.max(axis=0) - verts.min(axis=0)
    up = "y" if ext[1] <= ext[2] else "z"
    print(
        "[replica2usd] up-axis auto: extents x={:.2f} y={:.2f} z={:.2f} -> {}-up".format(
            ext[0], ext[1], ext[2], up
        )
    )
    return up


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ply",
        required=True,
        help="input mesh.ply (binary little endian)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="output .usd path",
    )
    # Replica 는 z-up 확정 (repo issue #26, 저자: gravity = -z). ScanNet 등
    # 다른 source 를 쓸 때만 auto/y 를 고려할 것 (auto 는 bbox 최단축 heuristic).
    p.add_argument(
        "--up",
        choices=["auto", "y", "z"],
        default="z",
        help="up axis of the INPUT mesh. y means rotate to z-up (bake into points). "
        "Replica is z-up (default)",
    )
    p.add_argument(
        "--floor-pct",
        type=float,
        default=0.5,
        help="percentile of up-coords taken as the floor height (shifted to z=0). "
        "negative value disables the shift",
    )
    p.add_argument(
        "--no-recenter",
        action="store_true",
        help="keep original xy origin (default recenters bbox center to 0,0)",
    )
    p.add_argument(
        "--physics",
        choices=["none", "static"],
        default="none",
        help="static: author triangle-mesh CollisionAPI (cooking cost at load). "
        "default none = visual backdrop only",
    )
    p.add_argument(
        "--no-material",
        action="store_true",
        help="skip the UsdPreviewSurface vertex-color material (displayColor only)",
    )
    p.add_argument(
        "--color-gamma",
        choices=["srgb", "linear"],
        default="srgb",
        help="srgb (default): convert 8-bit PLY colors from sRGB to linear "
        "for USD (fixes washed-out rendering). linear = legacy passthrough",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()

    # 공용 서버 규약: GPU 명시 필수 (미지정 시 kit 이 GPU0 을 무는 사고 방지)
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        sys.exit("[replica2usd] ERROR: set CUDA_VISIBLE_DEVICES explicitly (shared server)")

    # 1. binary little endian ply parsing (외부 의존성 없이 numpy 만 사용)
    print(f"[replica2usd] read {a.ply}")
    verts, colors, counts, indices = read_ply(a.ply)
    n_v, n_f = verts.shape[0], counts.shape[0]
    print(
        "[replica2usd] verts={} faces={} (counts: {}) colors={}".format(
            n_v,
            n_f,
            sorted(set(np.unique(counts).tolist())),
            "yes" if colors is not None else "NO",
        )
    )
    if indices.max() >= n_v or indices.min() < 0:
        raise SystemExit("ERROR: face index out of range")

    # 2. 업축 정렬: y-up 이면 +90 deg about X = (x, y, z) -> (x, -z, y)
    up = pick_up_axis(verts, a.up)
    if up == "y":
        verts = np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)
        print("[replica2usd] rotated y-up -> z-up")

    # 3. 재원점: xy 는 bbox 중심, z 는 바닥 백분위
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    shift = np.zeros(3)
    if not a.no_recenter:
        shift[0] = -(lo[0] + hi[0]) / 2.0
        shift[1] = -(lo[1] + hi[1]) / 2.0
    if a.floor_pct >= 0.0:
        shift[2] = -np.percentile(verts[:, 2], a.floor_pct)
    verts = verts + shift
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    print(
        "[replica2usd] shift=({:.3f},{:.3f},{:.3f}) bbox min=({:.2f},{:.2f},{:.2f}) "
        "max=({:.2f},{:.2f},{:.2f})".format(*shift, *lo, *hi)
    )

    # pxr 는 kit app 이 떠야 import 됨 (extract_props 와 동일 pattern).
    # ply 읽기/정렬을 app 기동보다 앞에 둬서 입력 오류는 빨리 실패시킴.
    app_launcher = AppLauncher(a)
    simulation_app = app_launcher.app
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    if os.path.exists(a.out):
        os.remove(a.out)
    stage = Usd.Stage.CreateNew(a.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # 5. 2단 구조: 최상위 /Bg 는 비워 두고 (Isaac Lab 이 spawn 변환으로 덮어씀)
    # geometry 는 /Bg/Geom 아래에 둠. 정렬은 좌표에 이미 구움.
    top = UsdGeom.Xform.Define(stage, "/Bg")
    stage.SetDefaultPrim(top.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Bg/Geom")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    # 스캔 mesh 는 구멍/뒤집힌 면이 흔해 양면 render 가 안전함
    mesh.CreateDoubleSidedAttr(True)
    # Gf.Vec3f 는 numpy scalar 를 안 받음 (Boost.Python) - 순수 float 로
    mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(float(lo[0]), float(lo[1]), float(lo[2])),
                Gf.Vec3f(float(hi[0]), float(hi[1]), float(hi[2])),
            ]
        )
    )

    # 4. vertex color -> displayColor primvar + UsdPreviewSurface material
    if colors is not None:
        pv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        c = colors.astype(np.float32) / 255.0
        if a.color_gamma == "srgb":
            # PLY 의 8bit 색은 sRGB 인데 USD displayColor/diffuseColor 는
            # linear 로 해석됨. 그대로 넣으면 밝고 채도 빠진(물빠진) 색이
            # 됨 -> sRGB EOTF 역변환을 구움 (08-25 실측)
            c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        pv.Set(Vt.Vec3fArray.FromNumpy(c.astype(np.float32)))
        if not a.no_material:
            # RTX 가 material 없는 mesh 의 displayColor 를 안 그릴 수 있어
            # primvar reader 로 diffuse 에 연결한 material 을 같이 심음
            mat = UsdShade.Material.Define(stage, "/Bg/VtxColorMat")
            surf = UsdShade.Shader.Define(stage, "/Bg/VtxColorMat/Surface")
            surf.CreateIdAttr("UsdPreviewSurface")
            reader = UsdShade.Shader.Define(stage, "/Bg/VtxColorMat/Reader")
            reader.CreateIdAttr("UsdPrimvarReader_float3")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            surf.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(out)
            surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            mat.CreateSurfaceOutput().ConnectToSource(
                surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            )
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    # 6. 선택: 정적 삼각 mesh 충돌 (--physics static)
    if a.physics == "static":
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mapi = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mapi.CreateApproximationAttr(UsdPhysics.Tokens.none)
        print("[replica2usd] static triangle-mesh collision authored")

    stage.GetRootLayer().Save()
    sz = os.path.getsize(a.out) / 1e6
    print(f"[replica2usd] wrote {a.out} ({sz:.1f} MB)")
    # simulation_app.close() 는 헤드리스에서 반환하지 않는 경우가 있어 생략
    # (형제 스크립트들과 동일 정책) -- 종료는 아래 os._exit(0) 가 담당


if __name__ == "__main__":
    # isaaclab.sh -p 가 프로세스를 붙잡는 경우 대비 강제 종료. 예외/sys.exit
    # 도 여기서 rc 보존 -- kit 기동 후 일반 종료는 좀비화 가능 (r3 참조)
    rc = 0
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, int):
            rc = e.code
        elif e.code is not None:
            print(e.code, file=sys.stderr)
            rc = 1
    except BaseException:
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
