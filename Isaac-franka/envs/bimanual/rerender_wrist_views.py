#!/usr/bin/env python
# ======================================
# File: rerender_wrist_views.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] rerender_wrist_views.py 2026-08-11-r1  (ascii-only console/comments)
r"""
기존 씨앗 HDF5 에 손목 카메라 2뷰를 추가하는 rerender 도구 (3뷰 전환용).

궤적 재생성 없음. HDF5 에 기록된 매 frame 의 관절 18개와
cube 상태를 그대로 설정(teleport)하고 손목 카메라 2대만 새로 render 함.
기존 ego_view 는 무수정 (생성 당시 연속 render 산출물 그대로 유지
= 1뷰 checkpoint 에서 warmstart 시 입력 연속성 보존).

동작
    1. 입력 seed.hdf5 를 출력 경로로 통째 복사
    2. demo 마다: frame 상태 설정 -> 물리 1 step(운동학 갱신)
       -> render 3연속(teleport 직후 temporal 누적 수렴) -> 손목 2뷰 capture
    3. obs/left_wrist_view, obs/right_wrist_view (T,H,W,3) uint8 gzip 추가

사용 (container 안)
    미리보기 (장착 각도 확인, 본 render 전 필수):
        isaaclab.sh -p rerender_wrist_views.py --headless \
            --input .../0806_152305_n2/seed.hdf5 --preview 6 \
            --preview-dir /root/project/out/wrist_preview
    본 render (배치 하나):
        isaaclab.sh -p rerender_wrist_views.py --headless \
            --input .../0806_152305_n2/seed.hdf5 --out .../seed_3view.hdf5

시간: frame 당 물리 1 step + render 3회. 704 episode(18.4만 frame)에
대략 2~4시간. 배치 4개를 GPU 별로 나눠 돌리면 그만큼 단축됨.
"""

import argparse
import os
import shutil
import sys
import traceback

# ================================================================== 1. 인자
parser = argparse.ArgumentParser(description="add wrist camera views to seed HDF5")
parser.add_argument("--input", required=True, help="source seed.hdf5")
parser.add_argument("--out", default="", help="output hdf5 (default: <input dir>/seed_3view.hdf5)")
parser.add_argument("--demos", type=int, default=0, help="limit demos (0 = all)")
parser.add_argument(
    "--renders-per-frame",
    type=int,
    default=3,
    help="renders per teleported frame; state is frozen so temporal "
    "accumulation converges to the current pose (same trick as the eval)",
)
parser.add_argument(
    "--warmup-renders",
    type=int,
    default=15,
    help="extra renders after each demo reset (flush stale history)",
)
parser.add_argument("--preview", type=int, default=0, help="dump N preview frames and exit")
parser.add_argument("--preview-dir", default="", help="dir for preview PNGs")
parser.add_argument("--tool-offset", type=float, default=0.1034)

# ================================================================== 2. app 기동
from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bimanual_scene as bs  # noqa: E402

bs.add_scene_args(parser)
parser.set_defaults(cam_w=256, cam_h=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


def grab_rgb(cam):
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)


def set_frame_state(scene, sim, jp, cube_p, cube_q, n):
    """기록된 frame 상태를 그대로 설정함 (속도는 0)."""
    dev = sim.device
    for key, sl in (("robot_l", slice(0, 9)), ("robot_r", slice(9, 18))):
        r = scene[key]
        q = torch.tensor([jp[sl]], dtype=torch.float32, device=dev)
        r.write_joint_state_to_sim(q, torch.zeros_like(q))
        # 물리 1 step 동안 자세가 흐트러지지 않게 목표도 같은 값으로
        r.set_joint_position_target(q)
        r.set_joint_velocity_target(torch.zeros_like(q))

    for i in range(1, bs.MAX_CUBES + 1):
        k = f"cube_{i}"
        obj = scene[k]
        st = obj.data.default_root_state.clone()
        if i <= n:
            st[:, 0:3] = torch.tensor([cube_p[i - 1]], device=dev)
            st[:, 3:7] = torch.tensor([cube_q[i - 1]], device=dev)
        else:
            st[:, 0:3] = torch.tensor([bs.PARK], device=dev) + scene.env_origins
        st[:, 7:] = 0.0
        obj.write_root_state_to_sim(st)
    scene.write_data_to_sim()


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(bs.build(args, with_camera=True))
    sim.reset()
    dt = sim.get_physics_dt()
    cam_l = scene["wrist_cam_l"]
    cam_r = scene["wrist_cam_r"]

    src = args.input
    dst = args.out or os.path.join(os.path.dirname(src), "seed_3view.hdf5")
    preview_only = args.preview > 0

    if not preview_only:
        # 원본은 무수정: 복사본에 뷰를 추가함
        print(f"[rr] copy {src} -> {dst}")
        shutil.copy2(src, dst)
        h = h5py.File(dst, "r+")
    else:
        h = h5py.File(src, "r")
        os.makedirs(args.preview_dir or ".", exist_ok=True)

    grp = h["data"]
    keys = sorted(grp.keys(), key=lambda s: int(s.split("_")[-1]))
    if args.demos:
        keys = keys[: args.demos]
    print(f"[rr] {len(keys)} demos from {src}")

    for di, k in enumerate(keys):
        g = grp[k]
        jp = g["obs/joint_pos"][...]
        n = int(g.attrs["num_cubes"])
        cp = g["obs/cube_pos"][...].reshape(len(jp), n, 3)
        cq = g["obs/cube_quat"][...].reshape(len(jp), n, 4)
        T = len(jp)

        # demo 시작: 첫 frame 상태로 설정 후 warmup render (묵은 history 제거)
        set_frame_state(scene, sim, jp[0], cp[0], cq[0], n)
        sim.step(render=False)
        scene.update(dt)
        for _ in range(max(args.warmup_renders, 1)):
            sim.render()

        if preview_only:
            # 미리보기: 첫 demo 의 "전체 구간"에서 고르게 N frame 을 뽑음.
            # 앞 frame 만 뽑으면 팔이 home 자세라 파지 순간의 손목 뷰
            # 판단 불가. frame 간 jump 가 커서 render 를 넉넉히(5회) 수렴
            import imageio

            picks = np.unique(np.linspace(0, T - 1, num=args.preview).astype(int))
            for t in picks:
                t = int(t)
                set_frame_state(scene, sim, jp[t], cp[t], cq[t], n)
                sim.step(render=False)
                scene.update(dt)
                for _ in range(max(args.renders_per_frame, 5)):
                    sim.render()
                # ego 칸은 render 가 아니라 "기록된 원본 frame"을 씀.
                # scene_cam 의 월드 pose 는 생성기가 실행 중에 잡던 것이라
                # 여기서 다시 render 하면 기본 위치의 엉뚱한 화면이 나옴
                # (본 render 산출물도 ego 는 원본 유지라 이쪽이 실물과 일치)
                row = np.concatenate(
                    [g["obs/ego_view"][t], grab_rgb(cam_l), grab_rgb(cam_r)], axis=1
                )
                imageio.imwrite(
                    os.path.join(args.preview_dir, f"{k}_f{t:03d}.png"), row
                )
                print(f"[rr] preview {k} f{t}/{T} (ego | wrist_l | wrist_r)")
            h.close()
            print(f"[rr] preview done -> {args.preview_dir}")
            return

        wl = np.empty((T, args.cam_h, args.cam_w, 3), dtype=np.uint8)
        wr = np.empty_like(wl)
        for t in range(T):
            set_frame_state(scene, sim, jp[t], cp[t], cq[t], n)
            sim.step(render=False)  # 관절/link 운동학 갱신 (1/120s, 상태 유지)
            scene.update(dt)
            for _ in range(max(args.renders_per_frame, 1)):
                sim.render()
            wl[t] = grab_rgb(cam_l)
            wr[t] = grab_rgb(cam_r)

        for name, arr in (("left_wrist_view", wl), ("right_wrist_view", wr)):
            path = f"obs/{name}"
            if path in g:
                del g[path]
            g.create_dataset(path, data=arr, compression="gzip", compression_opts=1)
        h.flush()
        print(f"[rr] {k} done ({di + 1}/{len(keys)}, T={T})")

    h.close()
    print(f"[rr] all done -> {dst}")


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
        # simulation_app.close() 는 headless 에서 안 돌아오는 경우가 있어
        # 결과 flush 후 바로 종료함 (zombie 방지, 생성기와 동일)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
