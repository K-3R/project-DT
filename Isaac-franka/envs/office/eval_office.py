#!/usr/bin/env python
# ======================================
# File: eval_office.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] eval_office.py 2026-08-15-r1  (ascii-only console/comments) (구명: dual_franka_office_eval.py)
r"""
사무실 마커 task 폐루프 평가 harness (env2).

finetune 된 GR00T checkpoint 를 추론 server 에 띄워 두고, 이 script 가
생성기(gen_office.py)와 같은 scene/random 배치에서 정책을 굴려
"마커를 연필꽂이에 꽂기" 성공률을 측정함. 구조는 1번 환경 harness
(bimanual/eval_bimanual.py)를 그대로 따르고, scene/reset/성공판정만
office 생성기에서 가져옴 (격리 원칙: bimanual 파일은 import 하지 않음).

구조
    [이 script: Isaac scene] --obs(3 view+joint18+지시문)--> [GR00T server]
                             <--action chunk(16 x 16dim TCP)---

규약 (학습 데이터와 동일해야 함)
    관측  video.ego_view / video.left_wrist_view / video.right_wrist_view
          (각 1,256,256,3 uint8, 순서 = 카메라 정체성) + state.left_arm(1,7)
          state.left_gripper(1,2) / state.right_arm(1,7) /
          state.right_gripper(1,2) / annotation.human.task_description
    지시문 개수별 "put {n} marker(s) into the pencil holder"
          (convert_office_lerobot.py 의 instruction_for 와 동일해야 함)
    action  world 절대 TCP 목표 16차원 = [Lpos3 Lquat4 Lgrip1 | R 동일]
            실행은 action 1개당 물리 6 step (120Hz -> 20Hz), 매 step render
    성공  생성기와 같은 판정: 모든 마커가 연필꽂이 안
    부분  level = 종료 시점에 통 안에 든 마커 수 (0..n)

주의 (office 고유 함정)
    이 scene 은 로봇 base 가 z180 으로 돌아 있어 PhysX world Jacobian 을
    root frame 으로 되돌리는 회전이 필수 (생성기 Arm.apply 와 동일).
    bimanual harness 의 ArmExec 를 그대로 쓰면 팔이 반대로 감.

사용 (server 를 먼저 띄운 뒤; run_server_finetuned.sh CKPT=ckpt_office_3view)
    CUDA_VISIBLE_DEVICES=1 isaaclab.sh -p eval_office.py \
        --headless --episodes-per-n 10 --out-dir /root/project/out/eval_office
"""

import argparse
import os
import sys
import traceback

# 1. arguments (argparse) ------------------------------------------------
parser = argparse.ArgumentParser(
    description="closed-loop success-rate eval for office marker task"
)

# argument groups (argparse)
g = parser.add_argument_group("eval")
g.add_argument(
    "--episodes-per-n", type=int, default=10, help="episodes per marker count"
)
g.add_argument(
    "--num-markers",
    default="1,2",
    help="marker count range min,max (training data covers 1-2 so far)",
)
g.add_argument(
    "--steps-per-marker",
    type=int,
    default=400,
    help="control-step budget per marker; limit = this x n. Generator "
    "physics budget was 2200/marker = 367 control steps, so 400 gives "
    "the same margin ratio as env1 (250 vs 233)",
)
g.add_argument(
    "--action-chunk",
    type=int,
    default=16,
    help="actions executed per inference (<= model horizon 16)",
)
g.add_argument(
    "--host",
    default="localhost",
)
g.add_argument(
    "--port",
    type=int,
    default=5555,
)
g.add_argument(
    "--instruction",
    default="",
    help="fixed instruction override; empty = per-count auto "
    "('put N marker(s) into the pencil holder', must match the converter)",
)
g.add_argument(
    "--seed",
    type=int,
    default=1000,
    help="eval seed (train used 100-300)",
)
g.add_argument(
    "--dry-run",
    action="store_true",
    help="print payload shapes after one reset and exit (no server needed)",
)

# argument groups (argparse)
g = parser.add_argument_group("output")
g.add_argument(
    "--out-dir",
    default="",
    help="dir for episodes.csv + summary.json",
)
g.add_argument(
    "--video-dir",
    default="",
    help="per-episode mp4 dir (empty = off)",
)
g.add_argument(
    "--stamp",
    type=int,
    default=1,
    help="append KST timestamp to outputs",
)
g.add_argument(
    "--tag",
    default="",
    help="suffix after the timestamp",
)

# 마커 random 배치 knob: 생성기와 같은 기본값 = 같은 초기상태 분포
# argument groups (argparse)
g = parser.add_argument_group("randomization (must match the generator)")
g.add_argument(
    "--region",
    default="0.06,0.26,-0.26,0.38",
)
g.add_argument(
    "--min-sep",
    type=float,
    default=0.11,
)
g.add_argument(
    "--rand-tries",
    type=int,
    default=400,
)
g.add_argument(
    "--tool-offset",
    type=float,
    default=0.1034,
)

# 2. IsaacLab 시작 ---------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import office_scene as osc  # noqa: E402

osc.add_scene_args(parser)
# 학습 데이터와 동일: 카메라 256, 로봇/연필꽂이 켬, 마커는 scene 최대 생성
parser.set_defaults(cam_w=256, cam_h=256, robots=1, items=osc.MAX_ITEMS, holder=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # 관측이 카메라이므로 항상 켬
args.robots = 1

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
from isaaclab.utils.math import (  # noqa: E402
    matrix_from_quat,
    quat_apply,
    quat_inv,
    subtract_frame_transforms,
)

# 3. 수학 (자세/판정 유틸) -------------------------------------------------------
# 생성기(gen_office.py)와 동일한 규약. 생성기는 script 라
# import 할 수 없어 필요한 최소만 복사함 (변경 시 양쪽 동기화 필요).
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0
GRIP_TH = 0.5 * (GRIP_OPEN + GRIP_CLOSE)  # 정책 출력 이진화 임계


def vec(s):
    return tuple(float(x) for x in s.split(","))


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
    """개수별 지시문. convert_office_lerobot.py 와 글자 단위로 동일해야 함."""
    unit = "marker" if n == 1 else "markers"
    return f"put {n} {unit} into the pencil holder"


def count_inserted(scene, n, hx, hy):
    """연필꽂이 안에 들어간 마커 수 (생성기와 동일 판정)."""
    ix, iy, wall_h = osc.HOLDER_INNER
    cnt = 0
    for i in range(1, n + 1):
        p = scene[f"item_{i}"].data.root_pos_w[0].cpu().numpy()
        if (
            abs(p[0] - hx) < ix / 2 + 0.01
            and abs(p[1] - hy) < iy / 2 + 0.01
            and 0.005 < p[2] < wall_h + 0.06
        ):
            cnt += 1
    return cnt


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


# 4. 배치/reset (생성기와 동일 규약) -----------------------------------------------
# 생성기와 동일한 마커 배치 sampler (초기상태 분포 일치)
def forbidden_rects(a):
    hx, hy = vec(a.holder_xy)
    kx, ky, _ = osc.KEYBOARD_SIZE
    px, py = osc.MOUSEPAD_XY
    mx, my, _ = osc.MOUSEPAD_SIZE
    pad = 0.05
    return [
        # (xmin, xmax, ymin, ymax)
        (
            osc.KEYBOARD_X - kx / 2 - pad,
            osc.KEYBOARD_X + kx / 2 + pad,
            -ky / 2 - pad,
            ky / 2 + pad,
        ),
        (px - mx / 2 - pad, px + mx / 2 + pad, py - my / 2 - pad, py + my / 2 + pad),
        (hx - 0.24, hx + 0.24, hy - 0.24, hy + 0.24),
    ]


def sample_layout(rng, a, n):
    xmin, xmax, ymin, ymax = vec(a.region)
    rects = forbidden_rects(a)

    def bad(p):
        return any(r[0] <= p[0] <= r[1] and r[2] <= p[1] <= r[3] for r in rects)

    for _ in range(a.rand_tries):
        pts, ok = [], True
        for _k in range(n):
            got = None
            for _t in range(80):
                p = (float(rng.uniform(xmin, xmax)), float(rng.uniform(ymin, ymax)))
                if bad(p):
                    continue
                if any(np.hypot(p[0] - q[0], p[1] - q[1]) < a.min_sep for q in pts):
                    continue
                got = p
                break
            if got is None:
                ok = False
                break
            pts.append(got)
        if ok:
            yaws = [float(rng.uniform(-np.pi, np.pi)) for _ in pts]
            return pts, yaws
    print("  [!] layout sampling failed -> line layout")
    return [(0.14 + 0.0 * i, -0.30 + 0.25 * i) for i in range(n)], [0.0] * n


def reset_episode(scene, sim, rng, a, n, dt):
    """생성기와 동일한 reset: 로봇 상태+PD 목표 초기화, 마커 배치, warmup."""
    scene.reset()
    for r in (scene["robot_l"], scene["robot_r"]):
        q = r.data.default_joint_pos.clone()
        r.write_joint_state_to_sim(q, torch.zeros_like(q))
        r.set_joint_position_target(q)  # PD 목표도 초기화 (검증된 함정)
        r.set_joint_velocity_target(torch.zeros_like(q))

    pts, yaws = sample_layout(rng, a, n)
    for i in range(1, osc.MAX_ITEMS + 1):
        k = f"item_{i}"
        obj = scene[k]
        st = obj.data.default_root_state.clone()
        if i <= n:
            pos = (pts[i - 1][0], pts[i - 1][1], osc.item_half_h(i))
            lay = osc.quat_mul(
                osc.quat_axis("z", np.degrees(yaws[i - 1])), osc.quat_axis("y", 90.0)
            )
        else:
            pos, lay = osc.PARK, (1.0, 0.0, 0.0, 0.0)
        st[:, 0:3] = torch.tensor([pos], device=sim.device) + scene.env_origins
        st[:, 3:7] = torch.tensor([lay], device=sim.device)
        st[:, 7:] = 0.0
        obj.write_root_state_to_sim(st)

    scene.write_data_to_sim()
    for _ in range(90):  # 안정화 + temporal 누적 buffer warmup (이전 잔상 제거)
        sim.step(render=True)
        scene.update(dt)


# 5. 팔 실행기 (policy action 을 IK 로 실행) -------------------------------------
class ArmExec:
    """정책이 준 world TCP 목표를 IK 로 실행하는 축소판 Arm (상태기계 없음).

    apply() 의 Jacobian 회전이 office 고유 필수 사항임: 이 scene 은 로봇
    base 가 z180 이라 world frame Jacobian 을 root frame 으로 되돌리지
    않으면 IK 방향이 반전됨 (생성기 2차 실행에서 실측, 공식
    task_space_actions.py jacobian_b 와 동일한 변환).
    """

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
        """world TCP 목표 -> hand 원점 목표 -> root frame IK 명령."""
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
        """명령 공백 금지: 지금 자세를 목표로 세움 (생성기와 동일 함정 대응)."""
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
        # world Jacobian -> root frame (office 필수, 위 class docstring 참조)
        rot_t = matrix_from_quat(quat_inv(rq))
        jac = jac.clone()
        jac[:, :3, :] = torch.bmm(rot_t, jac[:, :3, :])
        jac[:, 3:, :] = torch.bmm(rot_t, jac[:, 3:, :])
        bp, bq = subtract_frame_transforms(rp, rq, ep, eq)
        q = self.robot.data.joint_pos[:, self.arm_ids]
        self.robot.set_joint_position_target(
            self.ik.compute(bp, bq, jac, q), joint_ids=self.arm_ids
        )
        g = torch.full((self.robot.num_instances, 2), self.grip, device=self.dev)
        self.robot.set_joint_position_target(g, joint_ids=self.finger_ids)


# 6. 관측/action (server 와의 payload 계약) ------------------------------------
def build_payload(scene, cams, text):
    """학습 데이터(modality.json)와 정확히 같은 key/차원의 관측.

    3 view key 순서 = ego -> left_wrist -> right_wrist
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
        by_n[r["n"]][2] += r.get("level", 0)
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
                        "num_markers",
                        "steps_per_marker",
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


# 7. main (episode 반복 + 채점) ----------------------------------------------
def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    nmin, nmax = (int(v) for v in args.num_markers.split(","))
    hx, hy = vec(args.holder_xy)

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(osc.build(args, with_camera=True))
    sim.reset()
    cam = scene["scene_cam"]
    # ego 시점은 생성기와 동일한 상수 (학습/평가 분포 일치)
    cam.set_world_poses_from_view(
        eyes=torch.tensor([osc.EGO_EYE], device=sim.device),
        targets=torch.tensor([osc.EGO_TARGET], device=sim.device),
    )
    # 관측 카메라 3대, 순서 고정: ego -> left_wrist -> right_wrist
    cams = (cam, scene["wrist_cam_l"], scene["wrist_cam_r"])
    dt = sim.get_physics_dt()

    client = None
    if not args.dry_run:
        from groot_client import GrootClient

        client = GrootClient(args.host, args.port)
        print(f"[srv] ping -> {client.ping()}")

    # 개수별 균등 평가 계획 (n 순서 고정 -> 재현 가능)
    plan = [n for n in range(nmin, nmax + 1) for _ in range(args.episodes_per_n)]
    rows = []

    for ep, n in enumerate(plan):
        reset_episode(scene, sim, rng, args, n, dt)
        arm_l = ArmExec(scene, "robot_l", sim, args)
        arm_r = ArmExec(scene, "robot_r", sim, args)
        limit = args.steps_per_marker * n
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
                    # 확인용 영상은 model 이 실제로 본 3 view 를 가로로 붙여 저장
                    frames.append(np.concatenate([grab_rgb(c) for c in cams], axis=1))
                if ctrl >= limit:
                    break

            ok = count_inserted(scene, n, hx, hy) == n

        lvl = count_inserted(scene, n, hx, hy)
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
