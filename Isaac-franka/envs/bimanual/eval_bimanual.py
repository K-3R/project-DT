#!/usr/bin/env python
# ======================================
# File: eval_bimanual.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] eval_bimanual.py 2026-08-27-r2  (ascii-only console/comments) (구명: dual_franka_gr00t_eval.py)
r"""
양팔 Franka 탑쌓기 폐루프 평가 harness (Step 6).

파인튜닝된 GR00T checkpoint 를 추론 server 에 띄운 상태에서, 이 script 가
생성기(gen_bimanual.py)와 같은 scene/랜덤 배치에서 정책을 굴려
task 성공률을 측정함.

구조
    [이 script: Isaac scene] --obs(3뷰+joint18+지시문)--> [GR00T server]
                             <--action chunk(16 x 16dim TCP)---

규약 (학습 데이터와 동일해야 함)
    관측  video.ego_view / video.left_wrist_view / video.right_wrist_view
          (각 1,256,256,3 uint8, 순서 = 카메라 정체성, 3뷰 2026-08-11) /
          state.left_arm(1,7) / state.left_gripper(1,2) /
          state.right_arm(1,7) / state.right_gripper(1,2) /
          annotation.human.task_description
    action  월드 절대 TCP 목표 16차원 = [Lpos3 Lquat4 Lgrip1 | R 동일]
            실행은 action 1개당 물리 6 step(120Hz -> 20Hz), 매 step render
            (생성기와 동일한 일반 render = train/eval 분포 일치)
    성공  생성기와 같은 판정: 모든 cube 가 xy 3.5cm 안 + z 2cm 층 간격

사용 (server 를 먼저 띄운 뒤; run_server_finetuned.sh 참고)
    CUDA_VISIBLE_DEVICES=1 isaaclab.sh -p eval_bimanual.py \
        --headless --episodes-per-n 10 --out-dir /root/project/out/eval_bimanual
"""

import argparse
import os
import sys
import traceback

# ================================================================== 1. 인자
parser = argparse.ArgumentParser(
    description="closed-loop success-rate eval for bimanual Franka stacking"
)

g = parser.add_argument_group("eval")
g.add_argument("--episodes-per-n", type=int, default=10, help="episodes per cube count")
g.add_argument("--num-cubes", default="2,5", help="cube count range min,max")
g.add_argument(
    "--steps-per-cube",
    type=int,
    default=250,
    help="control-step budget per cube to stack; limit = this x (n-1)",
)
g.add_argument(
    "--action-chunk",
    type=int,
    default=16,
    help="actions executed per inference (<= model horizon 16)",
)
g.add_argument("--host", default="localhost")
g.add_argument("--port", type=int, default=5555)
g.add_argument(
    "--instruction",
    default="",
    help="fixed instruction override; empty = per-count auto "
    "('stack K cubes on the blue cube', must match the converter)",
)
g.add_argument("--seed", type=int, default=1000, help="eval seed (train used 100-400)")
g.add_argument(
    "--dry-run",
    action="store_true",
    help="print payload shapes after one reset and exit (no server needed)",
)

g = parser.add_argument_group("output")
g.add_argument("--out-dir", default="", help="dir for episodes.csv + summary.json")
g.add_argument("--video-dir", default="", help="per-episode mp4 dir (empty = off)")
g.add_argument("--stamp", type=int, default=1, help="append KST timestamp to outputs")
g.add_argument("--tag", default="", help="suffix after the timestamp")
g.add_argument("--cam-eye", default="1.9,0.0,1.3")
g.add_argument("--cam-target", default="0.45,0.0,0.10")

# cube 랜덤 배치 knob: 생성기와 같은 기본값 = 같은 초기상태 분포
g = parser.add_argument_group("randomization (must match the generator)")
g.add_argument("--randomize", type=int, default=1)
g.add_argument("--region", default="0.36,0.60,-0.26,0.26")
g.add_argument("--min-sep", type=float, default=0.11)
g.add_argument("--base-clear", type=float, default=0.13)
g.add_argument("--yaw-range", type=float, default=45.0)
g.add_argument("--reach-margin", type=float, default=0.72)
g.add_argument("--rand-tries", type=int, default=400)
g.add_argument("--tool-offset", type=float, default=0.1034)

# ================================================================== 2. app 기동
from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bimanual_scene as bs  # noqa: E402

bs.add_scene_args(parser)
parser.set_defaults(cam_w=256, cam_h=256)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # 관측이 카메라라 항상 켬

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

if args.stamp:
    # container 가 UTC 라 KST 로 고정함 (생성기와 동일)
    from datetime import datetime, timedelta, timezone

    KST = timezone(timedelta(hours=9))
    RUN_ID = datetime.now(KST).strftime("%m%d_%H%M%S") + (
        f"_{args.tag}" if args.tag else ""
    )
    if args.video_dir:
        args.video_dir = os.path.join(args.video_dir, RUN_ID)
    if args.out_dir:
        args.out_dir = os.path.join(args.out_dir, RUN_ID)
    print(f"[run] {RUN_ID}")
    if args.out_dir:
        print(f"[run] results -> {args.out_dir}")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers import (  # noqa: E402
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply, subtract_frame_transforms  # noqa: E402

# ================================================================== 3. 수학
# 생성기(gen_bimanual.py)와 동일한 규약. 생성기는 script 라
# import 불가 -> 필요한 최소만 복사함 (변경 시 양쪽 동기화 필요).
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
GRIP_TH = 0.5 * (GRIP_OPEN + GRIP_CLOSE)  # 정책 출력 이진화 임계


def vec(s):
    return tuple(float(x) for x in s.split(","))


def quat_z(yaw):
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def canonical_quat(q):
    w, x, y, z = (float(v) for v in q)
    if w < 0.0 or (w == 0.0 and x < 0.0):
        return (-w, -x, -y, -z)
    return (w, x, y, z)


def norm_quat(q):
    """정책 출력 quaternion 은 norm 이 정확히 1 이 아닐 수 있어 정규화함."""
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple((q / n).tolist())


def instruction_for(n):
    """개수별 지시문. 변환기(convert_bimanual_lerobot.py)와 동일해야 함.

    n 은 바닥(파란 cube) 포함 총 개수 -> 위에 쌓는 것은 n-1 개.
    """
    k = n - 1
    unit = "cube" if k == 1 else "cubes"
    return f"stack {k} {unit} on the blue cube"


def tower_level(cube_xyz):
    """episode 종료 시점의 탑 층수 (바닥 포함). 부분 성공 계측용.

    n>=3 의 0% 가 "1층은 쌓고 무너지는지 / 시작도 못 하는지"를 가름.
    """
    base = cube_xyz[0]  # cube_1 = 파란 바닥
    lvl = 1
    z = float(base[2])
    others = sorted(
        (p for p in cube_xyz[1:]
         if float(np.hypot(p[0] - base[0], p[1] - base[1])) < 0.035),
        key=lambda p: float(p[2]),
    )
    for p in others:
        if float(p[2]) > z + 0.02:
            lvl += 1
            z = float(p[2])
    return lvl


def check_success(cube_xyz):
    """생성기와 동일: 모든 cube 가 한 자리에 층을 이루면 성공."""
    cp = cube_xyz[np.argsort(cube_xyz[:, 2])]
    xy_ok = bool(np.abs(cp[:, :2] - cp[0, :2]).max() < 0.035)
    z_ok = bool(np.all(np.diff(cp[:, 2]) > 0.02))
    return (xy_ok and z_ok), cp


def grab_rgb(cam):
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)


def save_video(frames, path, fps):
    if not frames:
        return
    try:
        import imageio
    except ImportError:
        print("[vid] imageio not installed -> skip video")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps, codec="libx264", pixelformat="yuv420p")
        print(f"[vid] {path} ({len(frames)} frames)")
    except Exception as e:  # noqa: BLE001
        print(f"[vid] mp4 save failed: {path} ({e})")


# ================================================================== 4. 배치
def sample_layout(rng, a, n, base_l, base_r):
    """생성기와 동일한 cube 배치 sampler (초기상태 분포 일치)."""
    xmin, xmax, ymin, ymax = vec(a.region)
    z = bs.CUBE_HALF
    yr = np.deg2rad(a.yaw_range)

    def d2(p, q):
        return float(np.hypot(p[0] - q[0], p[1] - q[1]))

    def reach(p, base):
        return float(np.linalg.norm(np.array([p[0], p[1], z]) - base))

    for _ in range(a.rand_tries):
        c1 = (
            float(rng.uniform(xmin, xmax)),
            float(rng.uniform(ymin * 0.35, ymax * 0.35)),
            z,
        )
        if max(reach(c1, base_l), reach(c1, base_r)) > a.reach_margin:
            continue
        pts, ok = [c1], True
        for _k in range(n - 1):
            got = None
            for _t in range(60):
                p = (
                    float(rng.uniform(xmin, xmax)),
                    float(rng.uniform(ymin, ymax)),
                    z,
                )
                if d2(p, c1) < a.base_clear:
                    continue
                if any(d2(p, q) < a.min_sep for q in pts[1:]):
                    continue
                near = base_l if reach(p, base_l) <= reach(p, base_r) else base_r
                if reach(p, near) > a.reach_margin:
                    continue
                got = p
                break
            if got is None:
                ok = False
                break
            pts.append(got)
        if not ok:
            continue
        pos = {f"cube_{i+1}": pts[i] for i in range(n)}
        return pos, {k: float(rng.uniform(-yr, yr)) for k in pos}

    print("  [!] layout sampling failed -> default layout")
    pos = {f"cube_{i+1}": bs.cube_home(i + 1) for i in range(n)}
    return pos, {k: 0.0 for k in pos}


def reset_episode(scene, sim, rng, a, n, base_l, base_r, dt):
    """생성기와 동일한 reset: 로봇 상태+PD 목표 초기화, cube 배치, warmup."""
    scene.reset()
    for r in (scene["robot_l"], scene["robot_r"]):
        q = r.data.default_joint_pos.clone()
        r.write_joint_state_to_sim(q, torch.zeros_like(q))
        r.set_joint_position_target(q)
        r.set_joint_velocity_target(torch.zeros_like(q))

    if a.randomize:
        pos, yaws = sample_layout(rng, a, n, base_l, base_r)
    else:
        pos = {f"cube_{i+1}": bs.cube_home(i + 1) for i in range(n)}
        yaws = {k: 0.0 for k in pos}

    for i in range(1, bs.MAX_CUBES + 1):
        k = f"cube_{i}"
        obj = scene[k]
        st = obj.data.default_root_state.clone()
        st[:, 0:3] = (
            torch.tensor([pos.get(k, bs.PARK)], device=sim.device) + scene.env_origins
        )
        st[:, 3:7] = torch.tensor([quat_z(yaws.get(k, 0.0))], device=sim.device)
        st[:, 7:] = 0.0
        obj.write_root_state_to_sim(st)

    scene.write_data_to_sim()
    # 물리 안정화 + temporal 누적 buffer warmup (이전 episode 잔상 제거)
    for _ in range(90):
        sim.step(render=True)
        scene.update(dt)


# ================================================================== 5. 팔 실행기
class ArmExec:
    """정책이 준 월드 TCP 목표를 IK 로 실행하는 축소판 Arm (상태기계 없음)."""

    def __init__(self, scene, key, sim, a):
        self.robot = scene[key]
        self.key = key
        ent = SceneEntityCfg(
            key, joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        ent.resolve(scene)
        self.body_id = ent.body_ids[0]
        self.jac_id = self.body_id - 1  # fixed base
        self.arm_ids = ent.joint_ids
        self.finger_ids = [
            self.robot.find_joints("panda_finger_joint.*")[0][i] for i in range(2)
        ]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=scene.num_envs,
            device=sim.device,
        )
        self.dev = sim.device
        self.grip = GRIP_OPEN
        self.tool = a.tool_offset
        self.target_w = None
        self.hold_here()

    def eef_pose_w(self):
        p = self.robot.data.body_state_w[:, self.body_id, 0:7]
        return p[:, 0:3], p[:, 3:7]

    def root_pose_w(self):
        s = self.robot.data.root_state_w[:, 0:7]
        return s[:, 0:3], s[:, 3:7]

    def aim(self, tcp_pos_w, quat_w):
        """월드 TCP 목표 -> hand 원점 목표 -> root frame IK 명령."""
        quat_w = canonical_quat(norm_quat(quat_w))
        q = torch.tensor([quat_w], dtype=torch.float32, device=self.dev)
        axis = torch.tensor(
            [[0.0, 0.0, self.tool]], dtype=torch.float32, device=self.dev
        )
        back = quat_apply(q, axis)[0].cpu().numpy()
        hand = tuple(float(tcp_pos_w[i] - back[i]) for i in range(3))
        self.target_w = (hand, quat_w)
        rp, rq = self.root_pose_w()
        tp = torch.tensor([hand], dtype=torch.float32, device=self.dev)
        bp, bq = subtract_frame_transforms(rp, rq, tp, q)
        self.ik.set_command(torch.cat([bp, bq], dim=-1))

    def hold_here(self):
        """명령 공백 금지: 지금 자세를 목표로 세움 (생성기와 동일한 함정 대응)."""
        ep, eq = self.eef_pose_w()
        rp, rq = self.root_pose_w()
        bp, bq = subtract_frame_transforms(rp, rq, ep, eq)
        self.ik.set_command(torch.cat([bp, bq], dim=-1))
        self.target_w = (tuple(ep[0].cpu().numpy().tolist()), None)

    def command(self, pos_w, quat_w, grip_val):
        self.aim(pos_w, quat_w)
        self.grip = GRIP_OPEN if float(grip_val) >= GRIP_TH else GRIP_CLOSE

    def apply(self):
        jac = self.robot.root_physx_view.get_jacobians()[
            :, self.jac_id, :, self.arm_ids
        ]
        ep, eq = self.eef_pose_w()
        rp, rq = self.root_pose_w()
        bp, bq = subtract_frame_transforms(rp, rq, ep, eq)
        q = self.robot.data.joint_pos[:, self.arm_ids]
        self.robot.set_joint_position_target(
            self.ik.compute(bp, bq, jac, q), joint_ids=self.arm_ids
        )
        g = torch.full((self.robot.num_instances, 2), self.grip, device=self.dev)
        self.robot.set_joint_position_target(g, joint_ids=self.finger_ids)


# ================================================================== 6. 관측/action
def build_payload(scene, cams, text):
    """학습 데이터(modality.json)와 정확히 같은 key/차원의 관측.

    3뷰 (2026-08-11): key 순서 = ego -> left_wrist -> right_wrist
    (변환기 VIDEO_KEYS / our_configs video_keys 와 동일해야 함)
    """
    jl = scene["robot_l"].data.joint_pos[0].cpu().numpy().astype(np.float64)
    jr = scene["robot_r"].data.joint_pos[0].cpu().numpy().astype(np.float64)
    return {
        "video.ego_view": grab_rgb(cams[0])[None, ...],
        "video.left_wrist_view": grab_rgb(cams[1])[None, ...],
        "video.right_wrist_view": grab_rgb(cams[2])[None, ...],
        "state.left_arm": jl[:7][None, :],
        "state.left_gripper": jl[7:9][None, :],
        "state.right_arm": jr[:7][None, :],
        "state.right_gripper": jr[7:9][None, :],
        "annotation.human.task_description": [text],
    }


def parse_chunk(chunk):
    """server 응답 -> (T,) sequence 6개. T = model horizon(16)."""
    lp = np.asarray(chunk["action.left_eef_pos"]).reshape(-1, 3)
    lq = np.asarray(chunk["action.left_eef_quat"]).reshape(-1, 4)
    lg = np.asarray(chunk["action.left_gripper"]).reshape(-1)
    rp = np.asarray(chunk["action.right_eef_pos"]).reshape(-1, 3)
    rq = np.asarray(chunk["action.right_eef_quat"]).reshape(-1, 4)
    rg = np.asarray(chunk["action.right_gripper"]).reshape(-1)
    return lp, lq, lg, rp, rq, rg


def cube_xyz(scene, n):
    return np.stack(
        [scene[f"cube_{i+1}"].data.root_pos_w[0].cpu().numpy() for i in range(n)]
    )


def write_results(out_dir, a, rows):
    import csv
    import json

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "n", "steps", "success", "level"])
        w.writeheader()
        w.writerows(rows)

    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], [0, 0, 0])
        by_n[r["n"]][0] += r["success"]
        by_n[r["n"]][1] += 1
        by_n[r["n"]][2] += r.get("level", 1)
    n_ok = sum(r["success"] for r in rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "episodes_done": len(rows),
                "success": n_ok,
                "success_rate": (n_ok / len(rows)) if rows else 0.0,
                "by_n": {
                    str(k): {
                        "success": v[0],
                        "total": v[1],
                        "rate": v[0] / v[1],
                        "mean_level": v[2] / v[1],
                    }
                    for k, v in sorted(by_n.items())
                },
                "knobs": {
                    k: getattr(a, k)
                    for k in (
                        "episodes_per_n",
                        "num_cubes",
                        "steps_per_cube",
                        "action_chunk",
                        "seed",
                        "instruction",
                        "port",
                    )
                },
            },
            f,
            indent=2,
        )


# ================================================================== 7. main
def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    nmin, nmax = (int(v) for v in args.num_cubes.split(","))

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(bs.build(args, with_camera=True))
    sim.reset()
    cam = scene["scene_cam"]
    cam.set_world_poses_from_view(
        eyes=torch.tensor([vec(args.cam_eye)], device=sim.device),
        targets=torch.tensor([vec(args.cam_target)], device=sim.device),
    )
    # 관측 카메라 3대, 순서 고정: ego -> left_wrist -> right_wrist
    cams = (cam, scene["wrist_cam_l"], scene["wrist_cam_r"])
    dt = sim.get_physics_dt()
    base_l = scene["robot_l"].data.root_pos_w[0].cpu().numpy()
    base_r = scene["robot_r"].data.root_pos_w[0].cpu().numpy()

    client = None
    if not args.dry_run:
        from groot_client import GrootClient

        client = GrootClient(args.host, args.port)
        print(f"[srv] ping -> {client.ping()}")

    # 개수별 균등 평가 계획 (n 순서 고정 -> 재현 가능)
    plan = [n for n in range(nmin, nmax + 1) for _ in range(args.episodes_per_n)]
    rows = []

    for ep, n in enumerate(plan):
        reset_episode(scene, sim, rng, args, n, base_l, base_r, dt)
        arm_l = ArmExec(scene, "robot_l", sim, args)
        arm_r = ArmExec(scene, "robot_r", sim, args)
        limit = args.steps_per_cube * max(n - 1, 1)
        frames = []
        ok = False
        ctrl = 0

        text = args.instruction or instruction_for(n)

        if args.dry_run:
            payload = build_payload(scene, cams, text)
            print("\n===== payload =====")
            for k, v in payload.items():
                if isinstance(v, np.ndarray):
                    print(f"  {k:40s} {v.shape} {v.dtype}")
                else:
                    print(f"  {k:40s} {v}")
            print("===================\n")
            return

        while ctrl < limit and not ok:
            payload = build_payload(scene, cams, text)
            chunk = client.get_action(payload)
            lp, lq, lg, rp, rq, rg = parse_chunk(chunk)
            n_exec = min(args.action_chunk, len(lp))

            for t in range(n_exec):
                arm_l.command(lp[t], lq[t], lg[t])
                arm_r.command(rp[t], rq[t], rg[t])
                for _ in range(6):  # action 1개 = 물리 6 step (20Hz), 매 step render
                    arm_l.apply()
                    arm_r.apply()
                    scene.write_data_to_sim()
                    sim.step(render=True)
                    scene.update(dt)
                ctrl += 1
                if args.video_dir:
                    # review 영상은 model 이 실제로 본 3뷰를 가로로 붙여 저장
                    # (ego | left_wrist | right_wrist -- 파지 진단의 핵심)
                    frames.append(
                        np.concatenate([grab_rgb(c) for c in cams], axis=1)
                    )
                if ctrl >= limit:
                    break

            ok, _ = check_success(cube_xyz(scene, n))

        lvl = tower_level(cube_xyz(scene, n))
        rows.append(
            {"episode": ep, "n": n, "steps": ctrl, "success": int(ok), "level": lvl}
        )
        n_ok = sum(r["success"] for r in rows)
        print(
            f"[ep {ep + 1}/{len(plan)}] n={n} steps={ctrl} success={ok} "
            f"level={lvl}/{n} (total {n_ok}/{ep + 1})"
        )
        if args.video_dir:
            tag = "success" if ok else "fail"
            save_video(
                frames,
                os.path.join(args.video_dir, f"ep{ep:03d}_n{n}_{tag}.mp4"),
                20,
            )
        if args.out_dir:
            write_results(args.out_dir, args, rows)  # 매 episode 갱신 (중단 안전)

    n_ok = sum(r["success"] for r in rows)
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], [0, 0])
        by_n[r["n"]][0] += r["success"]
        by_n[r["n"]][1] += 1
    print(
        f"\n[result] {n_ok}/{len(rows)} = {100.0 * n_ok / max(len(rows), 1):.1f}%  "
        + "  ".join(f"n={k}:{v[0]}/{v[1]}" for k, v in sorted(by_n.items()))
    )
    if args.out_dir:
        print(f"[result] saved -> {args.out_dir}/")


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
        # simulation_app.close() 는 headless 에서 반환하지 않는 경우가 있어
        # 결과를 disk 에 쓴 뒤 바로 process 를 끝냄 (zombie 방지)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
