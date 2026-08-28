#!/usr/bin/env python
# [ver] isaac_franka_gr00t.py 2026-08-05-r2  (ascii-only console/comments)
r"""
Isaac Lab Franka 태스크를 GR00T 추론 서버로 구동하는 폐루프 하네스.

Isaac Lab 에는 GR00T 평가 하네스가 없다. 이 파일이 그것이다.
LIBERO 로 파인튜닝된 체크포인트(= Franka 단일팔, eef delta 액션)를
Isaac Lab 의 Franka 태스크에 물린다.

구조
    [Isaac Lab env] --obs--> 이 스크립트 --ZMQ--> [GR00T server]
                    <-action------------------------

주의
    - LIBERO 와 Isaac Lab 은 다른 시뮬이다. 규약을 맞춰도 zero-shot 성공률은
      낮을 것으로 예상한다 (sim-to-sim 갭). 이 스크립트의 1차 목적은 배관 검증이다.
    - --dry-run 으로 서버 없이 관측 규약만 먼저 확인할 수 있다.

사용법 (서버가 5555 에 떠 있다고 가정)
    CUDA_VISIBLE_DEVICES=3 isaaclab.sh -p isaac_franka_gr00t.py \
        --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
        --episodes 5 --headless
"""

import argparse
import os
import sys
import traceback

# ---------------------------------------------------------------- 인자 + 앱 기동
# AppLauncher 는 다른 isaaclab 임포트보다 먼저 와야 한다.
parser = argparse.ArgumentParser(
    description="Closed-loop harness: Isaac Lab Franka task driven by a GR00T "
    "inference server (see module docstring)",
)
parser.add_argument(
    "--task",
    default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
    help="Isaac Lab task id; must be an IK-Rel (delta eef) variant",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=5,
    help="rollout episodes",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=400,
    help="max env steps per episode",
)
parser.add_argument(
    "--host",
    default="localhost",
)
parser.add_argument(
    "--port",
    type=int,
    default=5555,
)
parser.add_argument(
    "--text",
    default="stack the cubes",
    help="language instruction; keep close to what the checkpoint saw",
)
parser.add_argument(
    "--cam-size",
    type=int,
    default=256,
    help="camera size; LIBERO checkpoints were trained at 256",
)
parser.add_argument(
    "--cam-key-main",
    default="table_cam",
    help="env observation key of the main camera",
)
parser.add_argument(
    "--cam-key-wrist",
    default="wrist_cam",
    help="env observation key of the wrist camera",
)
parser.add_argument(
    "--action-chunk",
    type=int,
    default=8,
    help="actions executed per inference (<=16)",
)
parser.add_argument(
    "--action-gain",
    type=float,
    default=1.0,
    help="policy-to-env action scale (convention mismatch knob)",
)
parser.add_argument(
    "--gripper-sign",
    type=float,
    default=1.0,
    help="gripper sign; set -1 if open/close is inverted",
)
parser.add_argument(
    "--warmup",
    type=int,
    default=10,
    help="settle steps with zero action (LIBERO num_steps_wait)",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="print observation keys/shapes and exit (no server needed)",
)
parser.add_argument(
    "--video-dir",
    default="",
    help="dir for per-episode rollout mp4 (empty = off)",
)
parser.add_argument(
    "--video-fps",
    type=int,
    default=20,
)
parser.add_argument(
    "--video-wrist",
    action="store_true",
    help="also tile the wrist camera into the video",
)
parser.add_argument(
    "--out-dir",
    default="",
    help="dir for episodes.csv + summary.json (empty = off)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
)

try:
    from isaaclab.app import AppLauncher
except ImportError:
    print("[fatal] isaaclab not importable; run via isaaclab.sh -p", file=sys.stderr)
    raise
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 카메라를 쓰므로 렌더가 필요하다. headless 여도 enable_cameras 는 켜야 한다.
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ------------------------------------------------------------------ 본 임포트
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402  (태스크 등록)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ 유틸
def quat_to_axis_angle(q):
    """(w,x,y,z) -> axis-angle 3벡터. LIBERO 의 quat2axisangle 과 같은 규약.

    LIBERO utils 는 (x,y,z,w) 순서를 받지만 Isaac Lab 은 (w,x,y,z) 를 준다.
    여기서 변환까지 함께 처리한다.
    """
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[0], q[1], q[2], q[3]
    if w < 0.0:  # 이중피복 제거: 항상 짧은 회전
        w, x, y, z = -w, -x, -y, -z
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    return (np.array([x, y, z], dtype=np.float64) / den) * angle


def to_np(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def find_obs(obs, key):
    """관측 dict 가 그룹으로 중첩돼 있어 키를 재귀로 찾는다."""
    if isinstance(obs, dict):
        if key in obs:
            return obs[key]
        for v in obs.values():
            hit = find_obs(v, key)
            if hit is not None:
                return hit
    return None


def describe(obs, indent=0):
    pad = "  " * indent
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, dict):
                print(f"{pad}{k}/")
                describe(v, indent + 1)
            else:
                a = to_np(v)
                print(
                    f"{pad}{k:24s} shape={tuple(a.shape)} dtype={a.dtype} "
                    f"min={a.min():.4f} max={a.max():.4f}"
                )


def prep_image(raw, size):
    """(B,H,W,C) 또는 (H,W,C) -> (1,H,W,3) uint8"""
    a = to_np(raw)
    if a.ndim == 4:
        a = a[0]
    if a.shape[-1] == 4:  # RGBA
        a = a[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
    if a.shape[0] != size or a.shape[1] != size:
        try:
            import cv2

            a = cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
        except ImportError:
            raise RuntimeError(
                f"camera is {a.shape[:2]} but {size}x{size} is required; "
                f"install cv2, or adjust --cam-size / env cfg resolution"
            )
    return a[None, ...]


def save_video(frames, path, fps):
    """mp4 저장. 코덱이 환경마다 달라 폴백 사슬을 둔다."""
    if not frames:
        print(f"[vid] no frames -> skip: {path}")
        return
    try:
        import imageio
    except ImportError:
        print("[vid] imageio not installed -> pip install imageio imageio-ffmpeg")
        return

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    last_err = None
    for codec in ("libx264", "h264", "mpeg4"):
        try:
            imageio.mimsave(path, frames, fps=fps, codec=codec)
            print(f"[vid] {path}  ({len(frames)} frames, codec={codec})")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    # 폴백: PNG 시퀀스
    stem = os.path.splitext(path)[0]
    os.makedirs(stem, exist_ok=True)
    for i, f in enumerate(frames):
        imageio.imwrite(os.path.join(stem, f"{i:05d}.png"), f)
    print(
        f"[vid] mp4 failed ({last_err}) -> PNG sequence: {stem}/ ({len(frames)} frames)"
    )


def write_results(out_dir, cfg, rows, n_success):
    """에피소드 CSV + 실행 설정 JSON. 매 에피소드마다 덮어쓴다."""
    import csv
    import json

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "steps", "success"])
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "task": cfg.task,
                "episodes_done": n,
                "episodes_planned": cfg.episodes,
                "success": n_success,
                "success_rate": (n_success / n) if n else 0.0,
                "mean_steps": (sum(r["steps"] for r in rows) / n) if n else 0.0,
                "action_gain": cfg.action_gain,
                "gripper_sign": cfg.gripper_sign,
                "action_chunk": cfg.action_chunk,
                "warmup": cfg.warmup,
                "cam_size": cfg.cam_size,
                "text": cfg.text,
                "seed": cfg.seed,
                "port": cfg.port,
            },
            f,
            indent=2,
        )


# ------------------------------------------------------------------ 메인
def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    # 카메라 해상도를 체크포인트 학습 해상도에 맞춘다 (기본 84 -> 256)
    for cam_attr in ("table_cam", "wrist_cam"):
        cam = getattr(env_cfg.scene, cam_attr, None)
        if cam is not None and hasattr(cam, "width"):
            cam.width = args.cam_size
            cam.height = args.cam_size
            print(f"[cam] {cam_attr} -> {args.cam_size}x{args.cam_size}")

    env = gym.make(args.task, cfg=env_cfg)
    print(f"[env] {args.task}")
    print(f"[env] action_space={env.action_space}")

    obs, _ = env.reset()
    print("\n===== observation structure =====")
    describe(obs)
    print("=================================\n")

    act_dim = env.action_space.shape[-1]
    print(f"[env] action dim = {act_dim}  (expected: 6 pose delta + 1 gripper = 7)")

    if args.dry_run:
        env.close()
        return

    from groot_client import GrootClient

    client = GrootClient(args.host, args.port)
    print(f"[srv] ping -> {client.ping()}")

    def grab_frame(o):
        """저장용 프레임 한 장. 메인 카메라, 필요하면 손목을 옆에 붙인다."""
        main_img = prep_image(find_obs(o, args.cam_key_main), args.cam_size)[0]
        if not args.video_wrist:
            return main_img
        wrist = prep_image(find_obs(o, args.cam_key_wrist), args.cam_size)[0]
        return np.concatenate([main_img, wrist], axis=1)

    n_success = 0
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        frames = []

        # 물체 안정화 대기 (LIBERO 의 num_steps_wait 과 같은 목적)
        zero = torch.zeros((1, act_dim), device=env.unwrapped.device)
        for _ in range(args.warmup):
            obs, _, _, _, _ = env.step(zero)
            if args.video_dir:
                frames.append(grab_frame(obs))

        done = False
        step = 0
        while step < args.max_steps and not done:
            eef_pos = to_np(find_obs(obs, "eef_pos")).reshape(-1)[:3]
            eef_quat = to_np(find_obs(obs, "eef_quat")).reshape(-1)[:4]
            grip = to_np(find_obs(obs, "gripper_pos")).reshape(-1)
            rot = quat_to_axis_angle(eef_quat)

            cam_main = find_obs(obs, args.cam_key_main)
            cam_wrist = find_obs(obs, args.cam_key_wrist)
            if cam_main is None or cam_wrist is None:
                raise RuntimeError(
                    f"camera obs not found ({args.cam_key_main}, {args.cam_key_wrist}); "
                    f"use a Visuomotor task and make sure cameras are enabled"
                )

            payload = {
                "video.image": prep_image(cam_main, args.cam_size),
                "video.wrist_image": prep_image(cam_wrist, args.cam_size),
                "state.x": np.array([[eef_pos[0]]], dtype=np.float64),
                "state.y": np.array([[eef_pos[1]]], dtype=np.float64),
                "state.z": np.array([[eef_pos[2]]], dtype=np.float64),
                "state.roll": np.array([[rot[0]]], dtype=np.float64),
                "state.pitch": np.array([[rot[1]]], dtype=np.float64),
                "state.yaw": np.array([[rot[2]]], dtype=np.float64),
                # LIBERO 는 gripper 를 2차원(양 손가락)으로 본다
                "state.gripper": np.asarray(grip[:2], dtype=np.float64).reshape(1, -1),
                "annotation.human.action.task_description": [args.text],
            }

            chunk = client.get_action(payload)
            keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
            arr = np.concatenate(
                [np.asarray(chunk[f"action.{k}"]).reshape(-1, 1) for k in keys], axis=1
            )  # (T, 7)

            n_exec = min(args.action_chunk, arr.shape[0])
            for t in range(n_exec):
                a = np.zeros(act_dim, dtype=np.float32)
                a[:6] = arr[t, :6] * args.action_gain
                a[6] = np.sign(arr[t, 6]) * args.gripper_sign
                obs, _, term, trunc, _ = env.step(
                    torch.tensor(a, device=env.unwrapped.device).unsqueeze(0)
                )
                step += 1
                if args.video_dir:
                    frames.append(grab_frame(obs))
                done = bool(to_np(term).any() or to_np(trunc).any())
                if done or step >= args.max_steps:
                    break

        ok = bool(to_np(term).any())  # 성공 종료 텀이 발화하면 term
        n_success += int(ok)
        print(
            f"[ep {ep + 1}/{args.episodes}] steps={step} success={ok} "
            f"(total {n_success}/{ep + 1})"
        )
        rows.append({"episode": ep, "steps": step, "success": int(ok)})

        if args.video_dir:
            tag = "success" if ok else "fail"
            save_video(
                frames,
                os.path.join(args.video_dir, f"ep{ep:03d}_{tag}.mp4"),
                args.video_fps,
            )

        # 매 에피소드마다 덮어써서, 중간에 끊겨도 여기까지가 남는다
        if args.out_dir:
            write_results(args.out_dir, args, rows, n_success)

    print(
        f"\n[result] {n_success}/{args.episodes} = "
        f"{100.0 * n_success / max(args.episodes, 1):.1f}%"
    )
    if args.out_dir:
        print(f"[result] saved -> {args.out_dir}/")
    env.close()


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
        # 호출하면 프로세스가 걸린 채 남아 Ctrl+C 를 눌러도 좀비가 된다.
        # 결과는 이미 디스크에 썼으므로 호출하지 않고 바로 끊는다.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
