#!/usr/bin/env python
# [ver] postprocess_mesh.py 2026-08-26-r3  (ascii-only console/comments)
# r3: 픽 1-3 퇴화(일직선/중복) 가드, cut-above 는 pick 모드에서 픽 1-3 만
#     피팅 (픽 4 = 상판 기준점이라 섞으면 절단면이 기움) + 비공면 경고,
#     저장 전 빈 메시 가드
# r2: 모드 해소(ransac|pick|raw 상호배타) + 스테이지별 좌표 재독 불변식
#     + 크롭 박스 목록 단일화 + 픽 파싱 통일 + 사이드카 json 기록
"""2DGS TSDF 메시 -> Isaac 배경용 정렬/스케일/정리 (파이프라인 후처리).

COLMAP 좌표계는 방향/크기/원점이 전부 임의라서, 메시를 그대로 USD 로
바꾸면 Isaac 에서 삐딱하고 크기도 엉터리다. 이 스크립트가 표준 자세
(상판 중심 원점, 상판 = z0, 미터 단위)로 만든다.

정렬 모드 (상호배타, 하나만):
    (기본) RANSAC 자동   밀집 코어에서 평면 후보를 찾아 상판으로 가정.
                         --plane-idx / --flip / --yaw-deg 로 교정.
    --pick-plane         meshlab 픽 4점으로 직접 정렬 (RANSAC 불신 시 정본).
                         픽 1-3 = 칸막이 상단 (비일직선 3점 = 수평 기준면),
                         픽 4 = 상판 위 한 점 (z=0), 픽 1->3 = x축.
                         --pick 4점 + --scale 필수.
    --raw-frame          정렬 생략 (이미 이 스크립트로 정렬된 파일의 재가공).
                         --scale 은 단위 보정용으로 계속 적용된다.

레시피 1 -- RANSAC 자동 (첫 시도):
    python gsrecon/postprocess_mesh.py --ply <fuse.ply> --out <out.ply>
    리포트의 평면 목록(extent 로 상판/껍질 판별)과 프리뷰를 보고
    --plane-idx / --flip 을 교정해 재실행한다 (수 분짜리라 싸다).

레시피 2 -- 픽 정본 (meshlab 4픽 -> 2단 실행):
    1) meshlab 에서 raw ply 에 픽 4점 (칸막이 상단 3 + 상판 1)
    2) 좌표 확인:  --pick "x,y,z;..." --crop-xy 0  로 실행하면 픽들의
       정렬 좌표와 추천 크롭 박스가 출력된다 (검증: 같은 높이 픽들의
       z 가 같고, 상판 픽 z = 0 이면 정렬 정답)
    3) 최종:  --pick-plane --scale <실측/픽거리> --pick "..." --pick-box 1.2
       (또는 --crop-box 로 정밀 박스)

파이프라인 순서 (코드 절 번호와 동일):
    [1] 프레임 계산 -> [2] 적용 -> [3] 픽 출력 -> [4] 재수평(--relevel)
    -> [5] 천장 절단(--cut-above) -> [6] 크롭 -> [7] 데시메이트
    -> [8] 성분 필터 -> [9] 평행이동 -> [10] 저장 (+사이드카 json)
불변식: 메시를 바꾼 스테이지 뒤에서는 반드시 좌표를 다시 읽는다.
픽 좌표(picks_m)는 재수평까지 따라 변환되므로 이후 스테이지에서 유효하다.

다음 단계 (변환기에는 정렬을 끄고 넘긴다 -- 여기서 이미 다 했으므로):
    replica_to_usd.py --up z --floor-pct -1 --no-recenter --ply <out> ...
주의: z=0 은 바닥이 아니라 "책상 상판"이다 (로봇 리그 배치 기준).
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import open3d as o3d

# 재수평(--relevel)이 상판 슬랩을 고르는 창 (정렬 프레임 미터 기준).
# 씬이 바뀌어 슬랩 위치가 다르면 여기를 조정한다 (기본 = office 책상 실측역).
RELEVEL_Z_ABS = 0.08  # 상판 근방 |z| 허용
RELEVEL_X_ABS = 0.6  # 책상 폭 안쪽
RELEVEL_Y_RANGE = (0.05, 0.65)  # 책상 깊이 안쪽
RELEVEL_TH = 0.008  # 슬랩 RANSAC 거리 임계 [m]


def parse_args():
    p = argparse.ArgumentParser(description="align/scale/crop a 2DGS TSDF mesh")
    p.add_argument("--ply", required=True, help="input fuse_*_post.ply from 2DGS")
    p.add_argument("--out", required=True, help="output .ply (binary, vertex colors)")

    g = p.add_argument_group("alignment (pick exactly one mode)")
    g.add_argument(
        "--pick-plane",
        action="store_true",
        help="align from --pick points instead of RANSAC (see recipe 2 in "
        "the module docstring). requires 4 picks and --scale",
    )
    g.add_argument(
        "--raw-frame",
        action="store_true",
        help="skip alignment (input already aligned by this script); "
        "--scale still applies as a unit fix, other alignment knobs are inert",
    )
    g.add_argument(
        "--flip",
        action="store_true",
        help="flip the up direction (use when the result is upside down)",
    )
    g.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="extra yaw in degrees after the automatic alignment "
        "(front/back is ambiguous -- use 180 if reversed)",
    )

    g = p.add_argument_group("scale")
    g.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="direct units-to-meters factor; overrides --desk-width when > 0. "
        "best source: tape measure / meshlab pick distance",
    )
    g.add_argument(
        "--desk-width",
        type=float,
        default=1.4,
        help="[ransac mode] scale so the chosen plane's major extent equals "
        "this many meters. 0 = keep original units",
    )

    g = p.add_argument_group("ransac mode knobs")
    g.add_argument(
        "--plane-idx",
        type=int,
        default=0,
        help="which RANSAC plane is the desk top (0 = most inliers)",
    )
    g.add_argument("--n-planes", type=int, default=3, help="candidates to report")
    g.add_argument(
        "--ransac-th",
        type=float,
        default=0.0,
        help="RANSAC distance threshold in INPUT units. 0 = auto "
        "(1 percent of the dense-core radius)",
    )
    g.add_argument(
        "--core-pct",
        type=float,
        default=60.0,
        help="restrict the plane search to the densest core (distance "
        "percentile from the robust center; 0 = full scene). guards against "
        "big flat shell patches beating the desk top",
    )

    g = p.add_argument_group("picks")
    g.add_argument(
        "--pick",
        default="",
        help="meshlab picks in the INPUT ply's frame, 'x,y,z;x,y,z;...'. "
        "echoed back in the aligned metric frame with a suggested crop box. "
        "note: values starting with '-' need the --pick=\"...\" form",
    )
    g.add_argument(
        "--pick-box",
        type=float,
        default=0.0,
        help="crop to the picks' bounding box grown by this factor about its "
        "center (e.g. 1.2). requires --pick; overrides --crop-box/--auto-box",
    )
    g.add_argument(
        "--cut-above",
        action="store_true",
        help="remove everything above the plane through the picks "
        "(pick mode: picks 1-3, the reference plane; other modes: all "
        "picks). requires --pick",
    )

    g = p.add_argument_group(
        "crop (priority: pick-box > crop-box > auto-box "
        "> crop-xy; crop-box2 unions with the winner)"
    )
    g.add_argument(
        "--crop-box",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="explicit crop box in aligned meters",
    )
    g.add_argument(
        "--crop-box2",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="extra crop box; kept region = union with the primary box "
        "(staircase ceilings). also works alone",
    )
    g.add_argument(
        "--auto-box",
        action="store_true",
        help="[ransac mode] derive xy from the desk-top inlier footprint "
        "(1-99 percentile + 5cm margin); z from --crop-z",
    )
    g.add_argument(
        "--crop-xy",
        type=float,
        default=3.0,
        help="fallback symmetric crop |x|,|y| <= this (meters). 0 = off",
    )
    g.add_argument(
        "--crop-z",
        type=float,
        nargs=2,
        default=(-1.5, 2.5),
        metavar=("ZMIN", "ZMAX"),
        help="z window used by --crop-xy and --auto-box",
    )

    g = p.add_argument_group("post steps")
    g.add_argument(
        "--relevel",
        action="store_true",
        help="re-level using the desk top itself: robust plane fit on the "
        "near-z0 desk slab (window constants at the top of this file), "
        "rotate flat and re-zero z. fixes tilted pick references. "
        "pick coordinates are transformed along, so later steps stay valid",
    )
    g.add_argument(
        "--shift",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="translate the mesh after cropping (e.g. origin to desk center)",
    )
    g.add_argument(
        "--target-tris",
        type=int,
        default=1_000_000,
        help="decimation target triangle count. 0 = no decimation",
    )
    g.add_argument(
        "--keep-comps",
        type=int,
        default=0,
        help="keep only the N largest connected components (1 = desk blob). "
        "overrides --min-comp-tris. desk items that are separate islands "
        "get dropped too -- raise N if something vanishes",
    )
    g.add_argument(
        "--min-comp-tris",
        type=int,
        default=5000,
        help="drop connected components smaller than this many triangles "
        "(floating 'cloud' debris). 0 = keep everything",
    )
    return p.parse_args()


# ---- 공용 헬퍼 -----------------------------------------------------------
def parse_picks(s):
    """--pick 문자열 -> (N,3) float 배열. 형식 오류는 즉시 종료."""
    try:
        pk = np.array(
            [[float(v) for v in t.split(",")] for t in s.split(";") if t.strip()]
        )
    except ValueError:
        sys.exit("[postmesh] ERROR: bad --pick format (want 'x,y,z;x,y,z;...')")
    if pk.ndim != 2 or pk.shape[1] != 3:
        sys.exit("[postmesh] ERROR: bad --pick shape (want triples)")
    return pk


def ransac_plane(pts, th, iters, rng):
    """시드 고정 RANSAC 평면. open3d segment_plane 은 시드를 못 줘서
    실행마다 다른 평면이 잡히는 함정이 있다 (08-25 실측) -> 자체 구현."""
    n = pts.shape[0]
    best_cnt, best = 0, None
    for _ in range(iters):
        i = rng.choice(n, 3, replace=False)
        nrm = np.cross(pts[i[1]] - pts[i[0]], pts[i[2]] - pts[i[0]])
        length = np.linalg.norm(nrm)
        if length < 1e-12:
            continue
        nrm = nrm / length
        cnt = int((np.abs((pts - pts[i[0]]) @ nrm) < th).sum())
        if cnt > best_cnt:
            best_cnt, best = cnt, (nrm, pts[i[0]])
    if best is None:
        sys.exit("[postmesh] ERROR: ransac found no plane")
    nrm, p0 = best
    inl = np.abs((pts - p0) @ nrm) < th
    # SVD 로 평면 정련 후 인라이어 재선정
    c = pts[inl].mean(axis=0)
    _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    nrm = vt[2]
    inl = np.abs((pts - c) @ nrm) < th
    return nrm, inl


def plane_extent(pts, nrm):
    """평면 인라이어의 장축 강건 폭 (2.5-97.5 백분위). 상판이면 책상폭
    근처, 껍질/바닥 조각이면 훨씬 큰 값이 나와 판별 지표가 된다."""
    c = pts.mean(axis=0)
    q = pts - c - np.outer((pts - c) @ nrm, nrm)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    t = q @ vt[0]
    return float(np.percentile(t, 97.5) - np.percentile(t, 2.5))


def rot_a_to_b(a, b):
    """단위벡터 a 를 b 로 보내는 회전행렬 (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))


def rot_z(yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_mask(verts, box):
    x0, x1, y0, y1, z0, z1 = box
    return (
        (verts[:, 0] >= x0)
        & (verts[:, 0] <= x1)
        & (verts[:, 1] >= y0)
        & (verts[:, 1] <= y1)
        & (verts[:, 2] >= z0)
        & (verts[:, 2] <= z1)
    )


def clean_after_removal(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()


# ---- 모드 해소 -----------------------------------------------------------
def resolve_mode(a):
    """정렬 모드 하나를 확정하고, 모드와 모순인 플래그 조합을 조기 거부.

    조용한 무시(플래그를 줬는데 아무 일도 안 일어남)를 없애는 것이 목적."""
    if a.pick_plane and a.raw_frame:
        sys.exit("[postmesh] ERROR: --pick-plane and --raw-frame are exclusive")
    mode = "pick" if a.pick_plane else ("raw" if a.raw_frame else "ransac")
    if mode == "pick":
        if not a.pick:
            sys.exit("[postmesh] ERROR: --pick-plane needs --pick (4 points)")
        if a.scale <= 0:
            sys.exit(
                "[postmesh] ERROR: --pick-plane needs --scale "
                "(measured_m / raw_units)"
            )
    if a.pick_box > 0 and not a.pick:
        sys.exit("[postmesh] ERROR: --pick-box needs --pick")
    if a.cut_above and not a.pick:
        sys.exit("[postmesh] ERROR: --cut-above needs --pick")
    # 픽 개수 가드 (평면 SVD 는 3점, bbox 는 2점이 최소)
    if a.pick:
        n_picks = parse_picks(a.pick).shape[0]
        if a.cut_above and n_picks < 3:
            sys.exit("[postmesh] ERROR: --cut-above needs >= 3 picks")
        if a.pick_box > 0 and n_picks < 2:
            sys.exit("[postmesh] ERROR: --pick-box needs >= 2 picks")
    if a.auto_box and mode != "ransac":
        sys.exit(
            "[postmesh] ERROR: --auto-box needs the RANSAC mode "
            "(it uses the desk-top inliers)"
        )
    if a.pick_box > 0 and a.crop_box is not None:
        print("[postmesh] WARNING: --pick-box overrides --crop-box")
    print(f"[postmesh] alignment mode: {mode}")
    return mode


# ---- [1] 프레임 계산 (M 회전, origin, scale) -----------------------------
def frame_ransac(verts, a, rng):
    """RANSAC 자동 정렬. 반환: M, origin, scale, 상판 인라이어(M-프레임)."""
    # 함정 (08-25 실측): 씬 전체에서 '최대 평면'을 찾으면 unbounded 배경
    # 껍질의 큰 평면 조각이 책상 상판을 이긴다 -> 그 폭으로 스케일이 잡혀
    # 씬 전체가 수십 배 축소된다. 카메라 궤도의 중심 = 책상 = 버텍스 밀집
    # 코어이므로, 평면 탐색을 밀집 코어(중앙 거리 백분위)로 제한한다.
    n_v = verts.shape[0]
    n_s = min(n_v, 300_000)
    sample = verts[rng.choice(n_v, n_s, replace=False)]
    center = np.median(sample, axis=0)
    dist = np.linalg.norm(sample - center, axis=1)
    core = sample[dist <= np.percentile(dist, a.core_pct)] if a.core_pct > 0 else sample
    core_r = float(np.percentile(np.linalg.norm(core - center, axis=1), 95))
    th = a.ransac_th if a.ransac_th > 0 else 0.01 * core_r
    print(
        "[postmesh] core {}/{} pts (pct {:.0f}), radius {:.3f}, ransac th {:.4f}".format(
            core.shape[0], n_s, a.core_pct, core_r, th
        )
    )

    planes = []  # (normal, inlier_pts)
    remain = core
    for i in range(a.n_planes):
        if remain.shape[0] < 2000:
            break
        nrm, inl_mask = ransac_plane(remain, th, 500, rng)
        pts = remain[inl_mask]
        planes.append((nrm, pts))
        print(
            "[postmesh] plane {}: inliers {} ({:.0f}%) normal ({:.2f},{:.2f},{:.2f}) "
            "extent {:.2f} units".format(
                i,
                pts.shape[0],
                100.0 * pts.shape[0] / core.shape[0],
                *nrm,
                plane_extent(pts, nrm),
            )
        )
        remain = remain[~inl_mask]
    if a.plane_idx >= len(planes):
        sys.exit(f"[postmesh] ERROR: plane-idx {a.plane_idx} not found")
    nrm, inl = planes[a.plane_idx]

    # 위쪽 결정: 상판 아래는 가려져 버텍스가 적다 -> 많은 쪽을 위로 가정.
    # 틀리면 --flip (프리뷰 한 번 보고 뒤집는 게 어떤 휴리스틱보다 확실하다)
    signed = (sample - inl.mean(axis=0)) @ nrm
    n_above = int((signed > 3 * th).sum())
    n_below = int((signed < -3 * th).sum())
    up = nrm if n_above >= n_below else -nrm
    if a.flip:
        up = -up
    sign = "+" if bool((up == nrm).all()) else "-"
    print(
        f"[postmesh] verts above/below plane: {n_above}/{n_below} "
        f"(up={sign}normal, flip={a.flip})"
    )
    R = rot_a_to_b(up, np.array([0.0, 0.0, 1.0]))

    # yaw: 상판 인라이어 PCA 장축 -> x
    inl_r = inl @ R.T
    xy = inl_r[:, :2] - inl_r[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(xy.T))
    major = evecs[:, int(np.argmax(evals))]
    yaw = -np.arctan2(major[1], major[0]) + np.radians(a.yaw_deg)
    M = rot_z(yaw) @ R

    # 원점 = 상판 인라이어 중심/높이, 스케일 = 장축 폭 -> desk-width
    inl_m = inl @ M.T
    origin = np.array(
        [inl_m[:, 0].mean(), inl_m[:, 1].mean(), float(np.median(inl_m[:, 2]))]
    )
    width = float(np.percentile(inl_m[:, 0], 97.5) - np.percentile(inl_m[:, 0], 2.5))
    if a.scale > 0:
        scale = a.scale
    else:
        scale = (a.desk_width / width) if a.desk_width > 0 else 1.0
    print(
        "[postmesh] plane width {:.3f} units -> scale {:.4f} "
        "(desk width {:.2f} m)".format(width, scale, a.desk_width)
    )
    return M, origin, scale, inl_m


def frame_pick(a):
    """픽 4점 정렬. 픽 1-3 = 기준 평면, 픽 4 = z0, 픽 1->3 = x축."""
    pk = parse_picks(a.pick)
    if pk.shape[0] < 4:
        sys.exit(
            "[postmesh] ERROR: --pick-plane needs 4 picks "
            "(3 reference-plane + 1 desk-top)"
        )
    q1, q2, q3, q4 = pk[:4]
    n = np.cross(q2 - q1, q3 - q1)
    n_len = float(np.linalg.norm(n))
    # 일직선/중복 픽이면 cross=0 -> 0나눗셈 NaN 이 프레임 전체를 오염시킨다
    if n_len < 1e-9:
        sys.exit(
            "[postmesh] ERROR: picks 1-3 are collinear or duplicated "
            "(cannot define the reference plane)"
        )
    n = n / n_len
    # 상판(픽4)은 기준 평면보다 아래 -> n 은 픽4 반대쪽이 위
    if np.dot(q4 - q1, n) > 0:
        n = -n
    if a.flip:
        n = -n
    R = rot_a_to_b(n, np.array([0.0, 0.0, 1.0]))
    e = R @ (q3 - q1)
    yaw = -np.arctan2(e[1], e[0]) + np.radians(a.yaw_deg)
    M = rot_z(yaw) @ R
    mid = M @ (0.5 * (q1 + q3))
    origin = np.array([mid[0], mid[1], (M @ q4)[2]])
    print(
        "[postmesh] pick-plane: up from picks 1-3, z0 from pick 4, "
        "x along pick 1->3, origin xy at picks 1,3 midpoint"
    )
    return M, origin, a.scale


def main():
    a = parse_args()
    mode = resolve_mode(a)
    rng = np.random.default_rng(0)

    src = os.path.expanduser(a.ply)
    print(f"[postmesh] read {src}")
    mesh = o3d.io.read_triangle_mesh(src)
    verts = np.asarray(mesh.vertices)
    n_v0, n_t0 = verts.shape[0], np.asarray(mesh.triangles).shape[0]
    if n_v0 == 0 or n_t0 == 0:
        sys.exit("[postmesh] ERROR: empty mesh")
    print(
        "[postmesh] verts={} tris={} colors={}".format(
            n_v0, n_t0, "yes" if mesh.has_vertex_colors() else "NO"
        )
    )

    # ---- [1] 프레임 계산 ----
    inl_m = None  # ransac 모드에서만 (auto-box 용)
    if mode == "ransac":
        M, origin, scale, inl_m = frame_ransac(verts, a, rng)
    elif mode == "pick":
        M, origin, scale = frame_pick(a)
    else:  # raw
        M, origin = np.eye(3), np.zeros(3)
        scale = a.scale if a.scale > 0 else 1.0
        print(f"[postmesh] raw-frame: alignment skipped, scale {scale}")

    # ---- [2] 프레임 적용 ----
    mesh.vertices = o3d.utility.Vector3dVector(((verts @ M.T) - origin) * scale)

    # ---- [3] 픽 좌표 변환/출력 (최종 프레임 기준) ----
    picks_m = None
    if a.pick:
        picks_m = ((parse_picks(a.pick) @ M.T) - origin) * scale
        for i, p_al in enumerate(picks_m, start=1):
            print(
                "[postmesh] pick {}: aligned ({:.3f},{:.3f},{:.3f}) m".format(i, *p_al)
            )
        lo_p, hi_p = picks_m.min(axis=0), picks_m.max(axis=0)
        print(
            "[postmesh] pick-suggested box (+5cm margin): "
            "--crop-box {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                lo_p[0] - 0.05,
                hi_p[0] + 0.05,
                lo_p[1] - 0.05,
                hi_p[1] + 0.05,
                lo_p[2] - 0.05,
                hi_p[2] + 0.05,
            )
        )

    # ---- [4] 상판 기준 재수평 ----
    # 픽 기준면이 기울었을 때 (예: 옆판이 앞판보다 낮은데 한 평면으로 픽)
    # 진짜 수평 기준인 상판 슬랩에 평면을 다시 맞춰 기울기를 제거한다.
    # 픽 좌표도 같이 변환해 이후 스테이지(cut-above/pick-box)가 유효하다.
    if a.relevel:
        v = np.asarray(mesh.vertices)
        sel = (
            (np.abs(v[:, 2]) < RELEVEL_Z_ABS)
            & (np.abs(v[:, 0]) < RELEVEL_X_ABS)
            & (v[:, 1] > RELEVEL_Y_RANGE[0])
            & (v[:, 1] < RELEVEL_Y_RANGE[1])
        )
        if int(sel.sum()) < 5000:
            sys.exit(
                "[postmesh] ERROR: relevel found too few desk-slab points "
                "(window constants at the top of this file may not fit this scene)"
            )
        n_lv, inl_lv = ransac_plane(v[sel], RELEVEL_TH, 300, rng)
        if n_lv[2] < 0:
            n_lv = -n_lv
        tilt = float(np.degrees(np.arccos(np.clip(n_lv[2], -1, 1))))
        R_lv = rot_a_to_b(n_lv, np.array([0.0, 0.0, 1.0]))
        v = v @ R_lv.T
        z0_lv = float(np.median((v[sel])[inl_lv][:, 2]))
        v[:, 2] -= z0_lv
        mesh.vertices = o3d.utility.Vector3dVector(v)
        if picks_m is not None:
            picks_m = picks_m @ R_lv.T
            picks_m[:, 2] -= z0_lv
        print(
            "[postmesh] relevel: desk-slab tilt {:.2f} deg removed, "
            "z re-zeroed ({:.3f})".format(tilt, z0_lv)
        )

    # ---- [5] 픽 평면 위 절단 (픽들을 지나는 천장면) ----
    if a.cut_above:
        # pick 모드의 픽 4 는 상판 기준점(천장 픽이 아님)이라 피팅에 섞으면
        # 절단면이 상판 쪽으로 기울어 칸막이 상단이 잘린다 -> 픽 1-3 만.
        # 다른 모드에서는 픽 전부가 절단면 픽이므로 전부 쓴다.
        cut_pts = picks_m[:3] if mode == "pick" else picks_m
        c0 = cut_pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(cut_pts - c0, full_matrices=False)
        if sv[1] < 1e-9:
            sys.exit("[postmesh] ERROR: cut-above picks are collinear (no plane)")
        n_c = vt[2]
        if n_c[2] < 0:
            n_c = -n_c
        resid = float(np.abs((cut_pts - c0) @ n_c).max())
        if resid > 0.02:
            print(
                "[postmesh] WARNING: cut-above picks deviate {:.3f} m from a "
                "single plane -- the cut will not follow them exactly".format(resid)
            )
        v = np.asarray(mesh.vertices)
        above = ((v - c0) @ n_c) > 0.005
        mesh.remove_vertices_by_mask(above)
        clean_after_removal(mesh)
        print(
            "[postmesh] cut-above pick plane: removed {} verts, "
            "normal ({:.2f},{:.2f},{:.2f})".format(int(above.sum()), *n_c)
        )

    # ---- [6] 크롭 (정렬 프레임 미터, 우선순위는 --help 참조) ----
    boxes = []
    if a.pick_box > 0:
        lo_p, hi_p = picks_m.min(axis=0), picks_m.max(axis=0)
        ctr, half = 0.5 * (lo_p + hi_p), 0.5 * (hi_p - lo_p) * a.pick_box
        boxes.append(
            [
                float(x)
                for x in np.concatenate(
                    [
                        [ctr[0] - half[0], ctr[0] + half[0]],
                        [ctr[1] - half[1], ctr[1] + half[1]],
                        [ctr[2] - half[2], ctr[2] + half[2]],
                    ]
                )
            ]
        )
        print(
            "[postmesh] pick-box x{:.1f}: --crop-box {:.2f} {:.2f} {:.2f} "
            "{:.2f} {:.2f} {:.2f}".format(a.pick_box, *boxes[0])
        )
    elif a.crop_box is not None:
        boxes.append(list(a.crop_box))
    elif a.auto_box:
        # 상판 인라이어의 발자국(footprint)이 곧 책상 xy 범위다
        inl_s = (inl_m - origin) * scale
        mgn = 0.05
        boxes.append(
            [
                float(np.percentile(inl_s[:, 0], 1) - mgn),
                float(np.percentile(inl_s[:, 0], 99) + mgn),
                float(np.percentile(inl_s[:, 1], 1) - mgn),
                float(np.percentile(inl_s[:, 1], 99) + mgn),
                a.crop_z[0],
                a.crop_z[1],
            ]
        )
        print(
            "[postmesh] auto box from desk footprint: "
            "x[{:.2f},{:.2f}] y[{:.2f},{:.2f}] z[{},{}]".format(*boxes[0])
        )
    elif a.crop_xy > 0:
        boxes.append(
            [-a.crop_xy, a.crop_xy, -a.crop_xy, a.crop_xy, a.crop_z[0], a.crop_z[1]]
        )
    if a.crop_box2 is not None:
        boxes.append(list(a.crop_box2))

    if boxes:
        v = np.asarray(mesh.vertices)  # 불변식: 절단 뒤라 반드시 재독
        n_before = v.shape[0]
        keep = np.zeros(n_before, dtype=bool)
        for b in boxes:
            keep |= box_mask(v, b)
        mesh.remove_vertices_by_mask(~keep)
        clean_after_removal(mesh)
        desc = " + ".join(
            "box x[{:.2f},{:.2f}] y[{:.2f},{:.2f}] z[{:.2f},{:.2f}]".format(*b)
            for b in boxes
        )
        print(
            "[postmesh] crop {}: verts {} -> {}".format(
                desc, n_before, len(mesh.vertices)
            )
        )

    # ---- [7] 데시메이트 ----
    n_t = np.asarray(mesh.triangles).shape[0]
    if a.target_tris > 0 and n_t > a.target_tris:
        mesh = mesh.simplify_quadric_decimation(a.target_tris)
        print(
            "[postmesh] decimate tris {} -> {}".format(
                n_t, np.asarray(mesh.triangles).shape[0]
            )
        )

    # ---- [8] 부유물(구름) 제거: 연결 성분 필터 ----
    # 반사/커버리지 부족이 만든 부유물은 본체와 연결되지 않은 작은 섬이다.
    # 데시메이트 후 기준이라 임계값 스케일이 예측 가능하다.
    if a.keep_comps > 0:
        tc, cn, _ = mesh.cluster_connected_triangles()
        tc, cn = np.asarray(tc), np.asarray(cn)
        top = np.argsort(cn)[::-1][: a.keep_comps]
        drop = ~np.isin(tc, top)
        if drop.any():
            mesh.remove_triangles_by_mask(drop)
            mesh.remove_unreferenced_vertices()
        print(
            "[postmesh] components: {} found, kept largest {} ({} tris)".format(
                cn.shape[0], a.keep_comps, int(cn[top].sum())
            )
        )
    elif a.min_comp_tris > 0:
        tc, cn, _ = mesh.cluster_connected_triangles()
        tc, cn = np.asarray(tc), np.asarray(cn)
        drop = cn[tc] < a.min_comp_tris
        n_small = int((cn < a.min_comp_tris).sum())
        if drop.any():
            mesh.remove_triangles_by_mask(drop)
            mesh.remove_unreferenced_vertices()
        print(
            "[postmesh] components: {} found, dropped {} small (<{} tris)".format(
                cn.shape[0], n_small, a.min_comp_tris
            )
        )

    # ---- [9] 평행이동 (원점 위치 조정) ----
    if any(a.shift):
        mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertices) + np.array(a.shift)
        )
        print("[postmesh] shift ({:.2f},{:.2f},{:.2f}) m".format(*a.shift))

    # ---- [10] 저장 + 사이드카 json (자산 계보 기록) ----
    # 크롭/절단이 전부 지웠으면 여기서 명시적으로 죽는다 (open3d 는 빈
    # 메시 쓰기를 거부해 "failed to write" 만 남아 원인 추적이 어렵다)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        sys.exit(
            "[postmesh] ERROR: all geometry removed before save "
            "(check crop boxes / picks / cut-above plane)"
        )
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if not o3d.io.write_triangle_mesh(out, mesh, write_ascii=False):
        sys.exit(f"[postmesh] ERROR: failed to write {out}")
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    print(
        "[postmesh] final verts={} tris={} bbox x[{:.2f},{:.2f}] "
        "y[{:.2f},{:.2f}] z[{:.2f},{:.2f}] m".format(
            v.shape[0],
            np.asarray(mesh.triangles).shape[0],
            lo[0],
            hi[0],
            lo[1],
            hi[1],
            lo[2],
            hi[2],
        )
    )
    print(f"[postmesh] wrote {out} ({os.path.getsize(out)/1e6:.1f} MB)")

    # 사이드카: 어느 입력/모드/변환으로 만든 자산인지 파일로 남긴다
    # (씬 쪽에서 scale 검증, 픽 재변환, take 세대 추적에 쓰는 계보 기록)
    kst = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M KST")
    meta = {
        "ver": "postprocess_mesh 2026-08-26-r3",
        "date": kst,
        "source_ply": src,
        "mode": mode,
        "M": np.asarray(M).tolist(),
        "origin": np.asarray(origin).tolist(),
        "scale": float(scale),
        "crop_boxes": boxes,
        "shift": list(a.shift),
        "argv": sys.argv[1:],
        "final": {
            "verts": int(v.shape[0]),
            "tris": int(np.asarray(mesh.triangles).shape[0]),
            "bbox": [float(x) for x in np.concatenate([lo, hi])],
        },
    }
    with open(out + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[postmesh] sidecar {out}.json")
    print("[postmesh] next: replica_to_usd.py --up z --floor-pct -1 --no-recenter")


if __name__ == "__main__":
    main()
