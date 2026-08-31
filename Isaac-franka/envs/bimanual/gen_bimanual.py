#!/usr/bin/env python
# ======================================
# File: gen_bimanual.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] gen_bimanual.py 2026-08-05-r5  (training data: 20Hz camera + HDF5 v2) (구명: dual_franka_stack_sm.py)
r"""
양팔 Franka -- 흩어진 cube 를 하나의 탑으로 쌓는 정답(씨앗) 궤적 생성기.

teleop 없이 script 상태기계로 만듦. GR00T finetuning 용 학습 데이터를
직접 출력함 (카메라 frame 포함, 20Hz). 정본 계획은
docs/bimanual_training_pipeline.md 참조.


task
    cube_1        탑의 바닥. 그대로 둠
    나머지 N-1개   base 가 가까운 팔이 집어 순서대로 그 위에 쌓음
    N             --num-cubes 범위에서 episode 마다 무작위 (기본 2~5)

    집기는 두 팔이 동시에 수행함 (담당 cube 가 서로 반대편이라 동선이 안 겹침).
    쌓는 자리는 하나뿐이므로 놓기만 한 번에 한 팔씩 -- Stack 이 lock 으로 조율함.
    로봇 link(link2~7)에 접촉력이 감지되면 그 episode 는 실패로 버림.


제어와 기록 주기
    물리 120Hz, 기록은 record-every(6) step 마다 = 20Hz.
    render 는 매 step 수행함 (일반 방식). 기록 시점에만 render 하는 절약
    방식은 temporal 누적 buffer(AA/denoiser)에 이전 pose 가 섞여 팔 잔상이
    생기는 문제로 폐기함 (FXAA 전환으로도 안 사라짐, 08-06 검증).

    주의: IK 가 제어하는 것은 panda_hand 원점이고, 실제 파지점(TCP)은
    거기서 손의 +z 로 tool_offset(0.1034) 만큼 나간 지점임.


상태 흐름 (한 cube 당)
    REST -> ABOVE_PICK -> AT_PICK -> CLOSE -> LIFT -> HOLD
         -> ABOVE_PLACE -> AT_PLACE -> OPEN -> UP -> RETREAT
         -> (다음 cube or FINISHED)


기록 (--out)  HDF5 v2
    data/demo_i/
     ├ actions   (T,16)  [Lpos3 Lquat4 Lgrip1 | Rpos3 Rquat4 Rgrip1]
     │                   world 절대 TCP 목표, quat 은 qw>=0 정규화
     ├ obs/
     │   ├ ego_view  (T,256,256,3) uint8   학습 카메라 (gzip)
     │   └ joint_pos(T,18) eef_pos_l/r eef_quat_l/r grip_l/r cube_pos cube_quat
     └ subtask/  place_done (T,) cube 하나를 쌓을 때마다 +1 되는 누적값
    attrs: num_samples(=T, 20Hz), physics_steps, success, num_cubes
    data attrs: fps, instruction


사용법
    isaaclab.sh -p gen_bimanual.py --headless --demos 20 \
        --out /root/project/datasets/franka_bimanual/seed.hdf5 \
        --video-dir /root/project/out/sm
"""

import argparse
import os
import sys
import traceback

# 1. arguments (argparse) ------------------------------------------------
parser = argparse.ArgumentParser(
    description="Bimanual Franka cube-tower seed trajectory generator "
    "(see module docstring for details)",
)

# argument groups (argparse)
g = parser.add_argument_group("run")
g.add_argument(
    "--demos",
    type=int,
    default=5,
    help="number of trajectories",
)
g.add_argument(
    "--max-steps",
    type=int,
    default=1400,
    help="physics step budget PER CUBE; actual limit = this x cubes to stack",
)
g.add_argument(
    "--seed",
    type=int,
    default=0,
)

# argument groups (argparse)
g = parser.add_argument_group("recording")
g.add_argument(
    "--record-every",
    type=int,
    default=6,
    help="record every N physics steps (120Hz/6 = 20Hz data); "
    "rendering itself happens every step",
)
g.add_argument(
    "--instruction",
    default="stack all the cubes on the blue cube",
    help="language instruction stored with the dataset",
)
g.add_argument(
    "--collision-th",
    type=float,
    default=1.0,
    help="net contact force threshold [N] on link2-7; above = episode failed",
)

# argument groups (argparse)
g = parser.add_argument_group("output")
g.add_argument(
    "--out",
    default="",
    help="HDF5 output path (empty = no save)",
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
    help="append run timestamp to output paths so reruns never overwrite",
)
g.add_argument(
    "--tag",
    default="",
    help="suffix after the timestamp",
)
g.add_argument(
    "--cam-eye",
    default="1.9,0.0,1.3",
)
g.add_argument(
    "--cam-target",
    default="0.45,0.0,0.10",
)

# argument groups (argparse)
g = parser.add_argument_group("randomization")
g.add_argument(
    "--randomize",
    type=int,
    default=1,
    help="re-place cubes in the region every episode (seed diversity)",
)
g.add_argument(
    "--num-cubes",
    default="2,5",
    help="cube count range min,max incl. the base cube; 2 means base + 1 "
    "to stack. 6 was too unstable, fixed to 2~5 (2026-08-05)",
)
g.add_argument(
    "--region",
    default="0.36,0.60,-0.26,0.26",
    help="cube spawn region xmin,xmax,ymin,ymax [m] (env frame)",
)
g.add_argument(
    "--min-sep",
    type=float,
    default=0.11,
    help="min distance between cubes [m] so the gripper clears neighbors",
)
g.add_argument(
    "--base-clear",
    type=float,
    default=0.13,
    help="min distance from the base cube (tower site) to others [m]",
)
g.add_argument(
    "--yaw-range",
    type=float,
    default=45.0,
    help="cube yaw range +/- deg; the gripper rotates to match",
)
g.add_argument(
    "--reach-margin",
    type=float,
    default=0.72,
    help="cube must be within this distance of its arm base [m] "
    "(Franka reach 0.855 minus margin)",
)
g.add_argument(
    "--rand-tries",
    type=int,
    default=400,
    help="sampling retries",
)

# argument groups (argparse)
g = parser.add_argument_group("motion")
g.add_argument(
    "--hover",
    type=float,
    default=0.12,
    help="approach height above object [m]",
)
g.add_argument(
    "--standby-y",
    type=float,
    default=0.32,
    help="sideways offset from the tower while waiting/retreating [m]",
)
g.add_argument(
    "--standby-h",
    type=float,
    default=0.28,
    help="standby height [m]",
)
g.add_argument(
    "--up-h",
    type=float,
    default=0.12,
    help="vertical exit height right after placing [m]",
)
g.add_argument(
    "--grasp-z",
    type=float,
    default=0.005,
    help="grasp z offset from cube center (replaced by measured value after CLOSE)",
)
g.add_argument(
    "--tool-offset",
    type=float,
    default=0.1034,
    help="panda_hand origin to TCP distance [m]; same value as the "
    "ee_frame offset in the Franka stack task",
)
g.add_argument(
    "--place-clear",
    type=float,
    default=0.005,
    help="clearance above the tower top for the place target [m]",
)
g.add_argument(
    "--pos-tol",
    type=float,
    default=0.006,
    help="reach tolerance [m]",
)
g.add_argument(
    "--dwell",
    type=int,
    default=25,
    help="settle steps before transition",
)
g.add_argument(
    "--state-timeout",
    type=int,
    default=250,
    help="force-advance after this many steps in one state "
    "(safety net when contact blocks the reach check; 0 = off)",
)

# 2. IsaacLab 시작 ---------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# bimanual_scene 은 최상단에서 isaaclab 를 import 하지 않게 만들어 둠.
# (app 기동 전에는 omni.client 가 없어 import 자체가 실패함)
import bimanual_scene as bs  # noqa: E402

# bs에서도 parser로 인자를 받으므로
# 인자를 넘겨줘서 등록을 해주어야 한다
bs.add_scene_args(parser)

# 학습 카메라는 256x256 (GR00T 전처리 crop0.95->resize224 에 맞춘 값)
parser.set_defaults(cam_w=256, cam_h=256)

# AppLauncher 에서도 parser를 받아서 등록을 해주어야 한다
AppLauncher.add_app_launcher_args(parser)

# arg를 받고 파싱
args = parser.parse_args()

# 해당 설정은 강제로 덮어쓴다
# 학습 데이터에 카메라 frame 이 반드시 들어가므로 항상 켬
args.enable_cameras = True

# 앱을 부팅한다
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 산출물 경로에 실행 시각을 붙임 (여러 번 돌려도 안 덮이도록) ----
if args.stamp:
    # container 가 UTC 라 local 시간 기준이면 9시간 이르게 찍힘 -> KST 고정
    from datetime import datetime, timedelta, timezone

    KST = timezone(timedelta(hours=9))
    RUN_ID = datetime.now(KST).strftime("%m%d_%H%M%S") + (
        f"_{args.tag}" if args.tag else ""
    )
    if args.video_dir:
        args.video_dir = os.path.join(args.video_dir, RUN_ID)
    if args.out:
        d, fn = os.path.split(args.out)
        stem, ext = os.path.splitext(fn)
        args.out = os.path.join(d, RUN_ID, f"{stem}{ext or '.hdf5'}")
    print(f"[run] {RUN_ID}")
    if args.video_dir:
        print(f"[run] video -> {args.video_dir}")
    if args.out:
        print(f"[run] hdf5  -> {args.out}")


# 앱이 부팅 된 다음에 import 해야 함 (IsaacLab 의 scene import 규약)

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

# 3. 수학 ------------------------------------------------------------------
DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)  # x축 180도 회전. gripper 가 아래를 향함
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0


def vec(s):
    return tuple(float(x) for x in s.split(","))


def quat_z(yaw):
    """z축 회전 quaternion (w,x,y,z)."""
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def canonical_quat(q):
    """quaternion 이중피복 제거: qw>=0 (w==0 이면 x>=0) 반구로 통일.

    q 와 -q 는 같은 회전이지만 기록에서 부호가 섞이면 학습 시
    per-component min_max 정규화가 망가짐 (검증에서 확인된 함정).
    """
    w, x, y, z = (float(v) for v in q)
    if w < 0.0 or (w == 0.0 and x < 0.0):
        return (-w, -x, -y, -z)
    return (w, x, y, z)


def yaw_of(q):
    """quaternion 에서 z축 회전각만 뽑음."""
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def grasp_quat(yaw):
    """아래를 향하되 z축으로 yaw 만큼 돌아간 파지 자세.

    cube 가 돌아가 있으면 gripper 도 같이 돌려야 면을 뭄.
    정사각 단면이라 90도 주기로 같으므로 회전량을 +/-45도 안으로 접음.
    """
    y = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return canonical_quat(quat_mul(quat_z(y), DOWN_QUAT))


# 4. 배치 ------------------------------------------------------------------
def sample_layout(rng, a, n, base_l, base_r):
    """구역 안에 cube n 개를 뿌림. cube_1 이 탑의 바닥.

    제약
      - 바닥과 나머지는 base_clear 이상 (쌓는 중 간섭 방지)
      - 나머지끼리는 min_sep 이상 (gripper 가 옆 cube 를 안 건드리게)
      - 각 cube 는 "가까운 팔"의 reach 안
      - 바닥은 양팔 모두 닿아야 함 (누가 쌓을지 모르므로)
    """
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

    print(
        "  [!] layout sampling failed -> using default layout "
        "(widen --region or lower --min-sep/--base-clear)"
    )
    pos = {f"cube_{i+1}": bs.cube_home(i + 1) for i in range(n)}
    return pos, {k: 0.0 for k in pos}


# 5. 조율 ------------------------------------------------------------------
class Stack:
    """쌓는 자리 하나를 두 팔이 나눠 쓰기 위한 공용 상태.

    놓을 높이는 "바닥 + level x 4cm" 가정 대신 현재 탑의 실측 높이에서
    구함. 앞 cube 가 조금 눌리거나 뜬 채로 놓이면 오차가 누적되기 때문.
    """

    def __init__(self, scene, base_key):
        p = scene[base_key].data.root_pos_w[0].cpu().numpy()
        self.xy = (float(p[0]), float(p[1]))
        self.yaw = yaw_of(scene[base_key].data.root_quat_w[0].cpu().numpy())
        self.placed = [base_key]  # 이미 탑을 이루는 cube 들
        self.level = 1  # 다음에 놓을 층
        self.owner = None  # 지금 놓고 있는 팔

    def top_z(self, scene):
        """지금 탑 꼭대기 cube 의 중심 z (실측)."""
        return max(float(scene[k].data.root_pos_w[0, 2]) for k in self.placed)

    def acquire(self, arm, scene):
        """놓을 cube 의 목표 중심 z. 다른 팔이 쓰는 중이면 None."""
        if self.owner is not None:
            return None
        self.owner = arm
        return self.top_z(scene) + bs.CUBE_SIZE

    def release(self, arm, cube_key):
        if self.owner is arm:
            self.owner = None
            self.placed.append(cube_key)
            self.level += 1


# 6. 팔 -------------------------------------------------------------------
(
    REST,
    ABOVE_PICK,
    AT_PICK,
    CLOSE,
    LIFT,
    HOLD,
    ABOVE_PLACE,
    AT_PLACE,
    OPEN,
    UP,
    RETREAT,
    FINISHED,
) = range(12)
SNAME = {
    REST: "REST",
    ABOVE_PICK: "ABOVE_PICK",
    AT_PICK: "AT_PICK",
    CLOSE: "CLOSE",
    LIFT: "LIFT",
    HOLD: "HOLD",
    ABOVE_PLACE: "ABOVE_PLACE",
    AT_PLACE: "AT_PLACE",
    OPEN: "OPEN",
    UP: "UP",
    RETREAT: "RETREAT",
    FINISHED: "FINISHED",
}


class Arm:
    """한 팔의 IK, gripper, 상태기계."""

    def __init__(self, scene, key, sim, stack, queue, a):
        self.robot = scene[key]
        self.key = key
        self.stack = stack
        self.queue = list(queue)  # 이 팔이 담당할 cube 들
        self.cube_key = None
        self.state = REST if self.queue else FINISHED
        self.timer = 0  # 도달 후 체류 counter
        self.elapsed = 0  # 현재 상태에 머문 step (시간초과용)
        self.just_placed = False  # 이번 step 에 한 cube 를 쌓아 올렸는지

        self.pick = None  # 파지 지점 (REST 에서 고정)
        self.pick_q = DOWN_QUAT
        self.place_q = grasp_quat(stack.yaw)  # 탑 각도에 맞춰 놓음
        self.place_z = None  # 놓을 cube 의 목표 중심 z
        self.hold_off = a.grasp_z  # TCP-cube 중심 실측 간격 (CLOSE 에서 갱신)
        self.side = 1.0 if float(self.robot.data.root_pos_w[0, 1]) >= 0 else -1.0

        ent = SceneEntityCfg(
            key, joint_names=["panda_joint.*"], body_names=["panda_hand"]
        )
        ent.resolve(scene)
        self.body_id = ent.body_ids[0]
        self.jac_id = self.body_id - 1  # fixed base 이므로 -1
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
        self.target_w = None  # hand 원점 목표 (world)
        self.tcp_target_w = None  # 파지점 목표 (world, 기록용)
        self.hold_here()  # 시작부터 IK 명령을 채워 둠

    # ---- 기하 --------------------------------------------------------
    def eef_pose_w(self):
        p = self.robot.data.body_state_w[:, self.body_id, 0:7]
        return p[:, 0:3], p[:, 3:7]

    def root_pose_w(self):
        s = self.robot.data.root_state_w[:, 0:7]
        return s[:, 0:3], s[:, 3:7]

    def tcp_now(self):
        """지금 파지점의 world 위치. hand 원점에서 손 +z 로 tool 만큼."""
        ep, eq = self.eef_pose_w()
        axis = torch.tensor(
            [[0.0, 0.0, self.tool]], dtype=torch.float32, device=self.dev
        )
        return (ep + quat_apply(eq, axis))[0].cpu().numpy()

    def dist(self):
        if self.target_w is None:
            return 1e9
        ep, _ = self.eef_pose_w()
        t = torch.tensor([self.target_w[0]], device=self.dev)
        return float(torch.norm(ep - t, dim=-1)[0])

    # ---- 명령 --------------------------------------------------------
    def aim(self, tcp_pos_w, quat_w=DOWN_QUAT):
        """world TCP 목표 -> hand 원점 목표 -> root frame IK 명령.

        quat 은 canonical_quat 로 반구를 통일해 기록의 부호 일관성을 지킴.
        """
        quat_w = canonical_quat(quat_w)
        q = torch.tensor([quat_w], dtype=torch.float32, device=self.dev)
        axis = torch.tensor(
            [[0.0, 0.0, self.tool]], dtype=torch.float32, device=self.dev
        )
        back = quat_apply(q, axis)[0].cpu().numpy()
        hand = tuple(float(tcp_pos_w[i] - back[i]) for i in range(3))
        self.tcp_target_w = tuple(float(v) for v in tcp_pos_w)
        self.target_w = (hand, quat_w)
        rp, rq = self.root_pose_w()
        tp = torch.tensor([hand], dtype=torch.float32, device=self.dev)
        bp, bq = subtract_frame_transforms(rp, rq, tp, q)
        self.ik.set_command(torch.cat([bp, bq], dim=-1))

    def hold_here(self):
        """지금 자세를 목표로 잡아 그 자리에 세움.

        IK controller 는 set_command 없이 compute 하면 목표가 0 으로 잡혀
        팔이 root 원점 쪽으로 끌려감. 명령이 없는 구간에서 반드시 부를 것.
        """
        ep, eq = self.eef_pose_w()
        rp, rq = self.root_pose_w()
        bp, bq = subtract_frame_transforms(rp, rq, ep, eq)
        self.ik.set_command(torch.cat([bp, bq], dim=-1))
        self.target_w = (
            tuple(ep[0].cpu().numpy().tolist()),
            canonical_quat(eq[0].cpu().numpy().tolist()),
        )
        self.tcp_target_w = self.target_w[0]

    def apply(self):
        """IK 를 풀어 관절 목표를 씀. 매 물리 step 호출."""
        if self.target_w is None:
            self.hold_here()
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

    # ---- 상태기계 ----------------------------------------------------
    def _to(self, s):
        self.state = s
        self.timer = 0
        self.elapsed = 0

    def _reached(self, a):
        """목표에 dwell step 이상 머물면 True. 시간초과도 True (안전장치)."""
        self.elapsed += 1
        if self.dist() < a.pos_tol:
            self.timer += 1
            if self.timer > a.dwell:
                return True
        else:
            self.timer = 0
        # 접촉으로 막히면 거리가 영영 안 줄어듦 (예: 놓을 때 아래 cube 에 닿음)
        if a.state_timeout and self.elapsed > a.state_timeout:
            print(
                f"  [!] {self.key} {SNAME[self.state]} timeout "
                f"d={self.dist():.4f} -> forcing next state"
            )
            return True
        return False

    def step(self, scene, a):
        self.just_placed = False
        if self.state == FINISHED:
            self.hold_here()
            return

        sx, sy = self.stack.xy
        standby = (
            sx,
            sy + self.side * a.standby_y,
            self.stack.top_z(scene) + a.standby_h,
        )

        # ---- 다음 cube 를 집으러 감 ----
        if self.state == REST:
            self.cube_key = self.queue.pop(0)
            c = scene[self.cube_key].data.root_pos_w[0].cpu().numpy()
            self.pick = (float(c[0]), float(c[1]), float(c[2]))
            self.pick_q = grasp_quat(
                yaw_of(scene[self.cube_key].data.root_quat_w[0].cpu().numpy())
            )
            self.grip = GRIP_OPEN
            self.aim((self.pick[0], self.pick[1], self.pick[2] + a.hover), self.pick_q)
            self._to(ABOVE_PICK)
            return

        # 파지 지점은 REST 에서 고정한 값을 씀.
        # 매 step cube 를 다시 읽으면 쥔 뒤에 cube 가 손을 따라와 목표가 도망감.
        hover_pick = (self.pick[0], self.pick[1], self.pick[2] + a.hover)
        at_pick = (self.pick[0], self.pick[1], self.pick[2] + a.grasp_z)
        s = self.state

        if s == ABOVE_PICK:
            self.aim(hover_pick, self.pick_q)
            if self._reached(a):
                self.aim(at_pick, self.pick_q)
                self._to(AT_PICK)

        elif s == AT_PICK:
            self.aim(at_pick, self.pick_q)
            if self._reached(a):
                self._to(CLOSE)

        elif s == CLOSE:
            self.grip = GRIP_CLOSE
            self.timer += 1
            if self.timer > a.dwell * 2:  # 손가락이 닫힐 시간
                # cube 가 gripper 안에서 어디에 물렸는지 실측함.
                # grasp_z 를 그대로 믿으면 놓을 때 누르거나 뜸.
                cz = float(scene[self.cube_key].data.root_pos_w[0, 2])
                self.hold_off = float(self.tcp_now()[2]) - cz
                self.aim(hover_pick, self.pick_q)
                self._to(LIFT)

        elif s == LIFT:
            self.aim(hover_pick, self.pick_q)  # 고정 지점이라 도망가지 않음
            if self._reached(a):
                self.aim(standby, self.place_q)
                self._to(HOLD)

        elif s == HOLD:
            # 쥔 채 자기 쪽 바깥에서 대기. 쌓는 자리를 잡으면 진행함.
            self.aim(standby, self.place_q)
            z = self.stack.acquire(self, scene)
            if z is not None:
                self.place_z = z
                self.aim((sx, sy, z + self.hold_off + a.hover), self.place_q)
                self._to(ABOVE_PLACE)

        elif s == ABOVE_PLACE:
            self.aim((sx, sy, self.place_z + self.hold_off + a.hover), self.place_q)
            if self._reached(a):
                self.aim(
                    (sx, sy, self.place_z + self.hold_off + a.place_clear),
                    self.place_q,
                )
                self._to(AT_PLACE)

        elif s == AT_PLACE:
            self.aim(
                (sx, sy, self.place_z + self.hold_off + a.place_clear), self.place_q
            )
            if self._reached(a):
                self._to(OPEN)

        elif s == OPEN:
            self.grip = GRIP_OPEN
            self.timer += 1
            if self.timer > a.dwell * 2:
                # 먼저 수직으로만 빠짐. 바로 옆으로 가면 방금 쌓은 탑을 스침.
                self.aim((sx, sy, self.place_z + a.up_h), self.place_q)
                self._to(UP)

        elif s == UP:
            self.aim((sx, sy, self.place_z + a.up_h), self.place_q)
            if self._reached(a):
                self.aim(standby, self.place_q)
                self._to(RETREAT)

        elif s == RETREAT:
            self.aim(standby, self.place_q)
            if self._reached(a):
                self.stack.release(self, self.cube_key)  # 다음 팔이 들어올 수 있게
                self.just_placed = True  # subtask 경계
                self._to(REST if self.queue else FINISHED)


# 7. 기록 (Recorder + HDF5) ------------------------------------------------
class Recorder:
    """한 episode 의 action, 관측(카메라 포함), subtask 경계를 20Hz 로 모음."""

    KEYS = (
        "actions",
        "joint_pos",
        "eef_pos_l",
        "eef_quat_l",
        "eef_pos_r",
        "eef_quat_r",
        "grip_l",
        "grip_r",
        "cube_pos",
        "cube_quat",
        "place_done",
        "ego_view",
    )

    def __init__(self, n_cubes):
        self.n = n_cubes
        self.buf = {k: [] for k in self.KEYS}

    def add(self, scene, arm_l, arm_r, frame):
        al = arm_l.tcp_target_w or (0.0, 0.0, 0.0)
        ar = arm_r.tcp_target_w or (0.0, 0.0, 0.0)
        ql = arm_l.target_w[1] if arm_l.target_w else DOWN_QUAT
        qr = arm_r.target_w[1] if arm_r.target_w else DOWN_QUAT
        self.buf["actions"].append(
            list(al) + list(ql) + [arm_l.grip] + list(ar) + list(qr) + [arm_r.grip]
        )
        self.buf["joint_pos"].append(
            np.concatenate(
                [
                    scene["robot_l"].data.joint_pos[0].cpu().numpy(),
                    scene["robot_r"].data.joint_pos[0].cpu().numpy(),
                ]
            )
        )
        for nm, arm in (("l", arm_l), ("r", arm_r)):
            p, q = arm.eef_pose_w()
            self.buf[f"eef_pos_{nm}"].append(p[0].cpu().numpy())
            self.buf[f"eef_quat_{nm}"].append(q[0].cpu().numpy())
            self.buf[f"grip_{nm}"].append(
                arm.robot.data.joint_pos[0, arm.finger_ids].cpu().numpy()
            )
        keys = [f"cube_{i+1}" for i in range(self.n)]
        self.buf["cube_pos"].append(
            np.concatenate([scene[k].data.root_pos_w[0].cpu().numpy() for k in keys])
        )
        self.buf["cube_quat"].append(
            np.concatenate([scene[k].data.root_quat_w[0].cpu().numpy() for k in keys])
        )
        self.buf["place_done"].append(0.0)
        self.buf["ego_view"].append(frame)

    def mark_place(self):
        """기록 구간 사이에 발생한 놓기 완료를 최근 기록 step 에 반영함."""
        if self.buf["place_done"]:
            self.buf["place_done"][-1] = 1.0

    def arrays(self):
        out = {}
        for k, v in self.buf.items():
            if k == "ego_view":
                out[k] = np.asarray(v, dtype=np.uint8)
            else:
                out[k] = np.array(v, np.float32)
        # 경계 신호는 "그 시점 이후 증가" 누적 형태로 (Mimic 규약과 동일)
        out["place_done"] = np.cumsum(out["place_done"]).astype(np.float32)
        return out

    def cube_xyz_last(self):
        return np.array(self.buf["cube_pos"][-1]).reshape(self.n, 3)

    def frames(self):
        return self.buf["ego_view"]


def save_video(frames, path, fps):
    """mp4 저장. codec 이 환경마다 달라 fallback 사슬을 둠."""
    if not frames:
        return
    try:
        import imageio
    except ImportError:
        print("[vid] imageio not installed -> skip video")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # yuv420p 명시: player 호환성 (기본값이 바뀌는 환경 대비), codec 은 log 로 확인
    err = None
    for codec in ("libx264", "h264", "mpeg4"):
        try:
            imageio.mimsave(path, frames, fps=fps, codec=codec, pixelformat="yuv420p")
            print(f"[vid] {path} ({len(frames)} frames, codec={codec})")
            return
        except Exception as e:
            err = e
            continue
    print(f"[vid] mp4 save failed: {path} (last error: {err})")


def grab_rgb(cam):
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)


def collided_now(scene, th):
    """두 로봇의 link2~7 net 접촉력이 임계값을 넘었는지 여부.

    history_length=6 이라 최근 6 물리 step 의 순간 접촉까지 cover 함.
    self-collision 은 spawn 에서 꺼 두었고 hand/finger 는 sensor 에서 제외했으므로
    여기 걸리는 접촉은 로봇-로봇 또는 로봇-table 충돌임.
    """
    for k in ("contact_l", "contact_r"):
        f = scene[k].data.net_forces_w_history  # (N, H, B, 3)
        if f is not None and float(torch.norm(f, dim=-1).max()) > th:
            return True
    return False


def open_hdf5(path, a):
    """점진 저장용 HDF5 를 엶 (batch 시작 시 1회).

    episode 를 RAM 에 모아뒀다 끝에 쓰는 방식은 batch 210개 기준 peak 가
    수십 GB 라 OOM 위험 -> 끝날 때마다 바로 쓰는 구조로 변경 (08-06).
    """
    import h5py

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = h5py.File(path, "w")
    grp = f.create_group("data")
    grp.attrs["total"] = 0
    grp.attrs["fps"] = 120.0 / a.record_every
    grp.attrs["instruction"] = a.instruction
    grp.attrs["action_layout"] = (
        "Lpos3 Lquat4 Lgrip1 | Rpos3 Rquat4 Rgrip1 "
        "(world-frame absolute TCP target, qw>=0)"
    )
    return f, grp


def append_demo(grp, rec, steps, n):
    """성공 episode 하나를 즉시 기록하고 flush 함.

    flush 까지 해 두면 도중에 process 가 죽어도 완료분은 파일에 남음.
    """
    i = int(grp.attrs["total"])
    arr = rec.arrays()
    d = grp.create_group(f"demo_{i}")
    d.create_dataset("actions", data=arr["actions"])
    d.attrs["num_samples"] = int(arr["actions"].shape[0])  # 20Hz step 수
    d.attrs["physics_steps"] = steps
    d.attrs["success"] = True
    d.attrs["num_cubes"] = n
    obs = d.create_group("obs")
    for k, v in arr.items():
        if k in ("actions", "place_done"):
            continue
        if k == "ego_view":
            obs.create_dataset(k, data=v, compression="gzip", compression_opts=1)
        else:
            obs.create_dataset(k, data=v)
    sub = d.create_group("subtask")
    sub.create_dataset("place_done", data=arr["place_done"])
    grp.attrs["total"] = i + 1
    grp.file.flush()


def write_meta(meta_dir, a, stats, n_ok):
    """실행 meta 를 기록함. episode 마다 갱신되므로 진행 확인용으로도 씀."""
    import json

    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "run_meta.json"), "w") as f:
        json.dump(
            {
                "success": n_ok,
                "demos": a.demos,
                "per_episode": [
                    {"n": n, "ok": bool(o), "steps": s, "collided": bool(c)}
                    for o, s, n, c in stats
                ],
                "fps": 120.0 / a.record_every,
                "instruction": a.instruction,
                "knobs": {
                    k: getattr(a, k)
                    for k in (
                        "num_cubes",
                        "region",
                        "min_sep",
                        "base_clear",
                        "yaw_range",
                        "reach_margin",
                        "hover",
                        "grasp_z",
                        "up_h",
                        "standby_y",
                        "standby_h",
                        "place_clear",
                        "pos_tol",
                        "dwell",
                        "state_timeout",
                        "max_steps",
                        "seed",
                        "tool_offset",
                        "record_every",
                        "collision_th",
                    )
                },
                "scene": {
                    k: getattr(a, k)
                    for k in (
                        "table",
                        "table_usd",
                        "table_mirror",
                        "mirror_side",
                        "layout",
                        "base_sep",
                        "table_dx",
                        "cam_w",
                        "cam_h",
                    )
                },
            },
            f,
            indent=2,
        )


# 8. episode (물리 루프 한 판) -------------------------------------------------
def reset_episode(scene, sim, rng, a, n, base_l, base_r, dt):
    """로봇을 초기 자세로, cube 를 구역에 배치하고 카메라를 warmup 함."""
    scene.reset()
    for r in (scene["robot_l"], scene["robot_r"]):
        q = r.data.default_joint_pos.clone()
        r.write_joint_state_to_sim(q, torch.zeros_like(q))
        # PD 목표도 초기 자세로 재설정. 상태만 reset 하면 안정화 구간 동안
        # 이전 episode 의 마지막 IK 목표로 팔이 끌려감 (충돌 중단 뒤
        # 다음 episode 가 저공 자세로 시작하던 문제, 08-06 검증에서 발견)
        r.set_joint_position_target(q)
        r.set_joint_velocity_target(torch.zeros_like(q))

    if a.randomize:
        pos, yaws = sample_layout(rng, a, n, base_l, base_r)
    else:
        pos = {f"cube_{i+1}": bs.cube_home(i + 1) for i in range(n)}
        yaws = {k: 0.0 for k in pos}

    # 쓰는 cube 는 구역에, 안 쓰는 cube 는 table 밖으로
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
    # 물리 안정화 + 카메라 warmup: reset 직후에는 이전 episode 장면이
    # temporal 누적 buffer 에 남아 첫 frame 에 흐리게 비침 (frame 당 감쇠
    # ~0.86 실측). 30 frame 이면 ~1% 잔여가 보여서 90 frame 을 흘려보냄
    for _ in range(90):
        sim.step(render=True)
        scene.update(dt)
    return pos, yaws


def assign_cubes(scene, n, base_l, base_r):
    """cube_1 을 뺀 나머지를 base 가 가까운 팔에게 배정함."""
    q_l, q_r = [], []
    for i in range(2, n + 1):
        k = f"cube_{i}"
        p = scene[k].data.root_pos_w[0].cpu().numpy()
        near_l = np.linalg.norm(p - base_l) <= np.linalg.norm(p - base_r)
        (q_l if near_l else q_r).append(k)
    return q_l, q_r


def check_success(cube_xyz):
    """모든 cube 가 한 자리에 층을 이루면 성공."""
    cp = cube_xyz[np.argsort(cube_xyz[:, 2])]
    xy_ok = bool(np.abs(cp[:, :2] - cp[0, :2]).max() < 0.035)
    z_ok = bool(np.all(np.diff(cp[:, 2]) > 0.02))
    return (xy_ok and z_ok), cp


def run_episode(scene, sim, cam, a, ep, n, base_l, base_r, rng, dt):
    pos, yaws = reset_episode(scene, sim, rng, a, n, base_l, base_r, dt)
    stack = Stack(scene, "cube_1")
    q_l, q_r = assign_cubes(scene, n, base_l, base_r)
    arm_l = Arm(scene, "robot_l", sim, stack, q_l, a)
    arm_r = Arm(scene, "robot_r", sim, stack, q_r, a)

    print(f"  cubes n={n}  L={q_l}  R={q_r}")
    print(
        "  layout "
        + "  ".join(
            f"{k[-1]}=({v[0]:.3f},{v[1]:+.3f},{np.rad2deg(yaws[k]):+.0f}deg)"
            for k, v in sorted(pos.items())
        )
    )

    rec = Recorder(n)
    step, prev = 0, None
    collided = False
    placed_since_record = False
    limit = a.max_steps * max(n - 1, 1)

    while step < limit and not (arm_l.state == FINISHED and arm_r.state == FINISHED):
        arm_l.step(scene, a)
        arm_r.step(scene, a)
        # 놓기 완료가 기록 step 사이에 발생해도 잃지 않도록 latch 함
        if arm_l.just_placed or arm_r.just_placed:
            placed_since_record = True
        arm_l.apply()
        arm_r.apply()
        scene.write_data_to_sim()
        sim.step(render=True)  # 매 step render (일반 방식, 잔상 없음)
        scene.update(dt)
        step += 1

        if step % max(a.record_every, 1) == 0:
            rec.add(scene, arm_l, arm_r, grab_rgb(cam))
            if placed_since_record:
                rec.mark_place()
                placed_since_record = False
            # 접촉 검사: history=6 이 기록 사이 물리 step 을 cover 함
            if collided_now(scene, a.collision_th):
                collided = True
                print(f"  [{step:5d}] COLLISION detected -> episode failed")
                break

        cur = (arm_l.state, arm_r.state)
        if cur != prev:
            print(
                f"  [{step:5d}] L={SNAME[arm_l.state]:12s} "
                f"R={SNAME[arm_r.state]:12s} level={stack.level}"
            )
            prev = cur
        elif step % 400 == 0:
            print(
                f"  [{step:5d}] ... L={SNAME[arm_l.state]:12s} d={arm_l.dist():.4f}  "
                f"R={SNAME[arm_r.state]:12s} d={arm_r.dist():.4f}"
            )

    ok, cp = check_success(rec.cube_xyz_last())
    ok = ok and not collided
    print(
        f"[ep {ep+1}/{a.demos}] n={n} steps={step} success={ok} "
        f"collided={collided} z={cp[:,2].round(3).tolist()}"
    )

    if a.video_dir:
        save_video(
            rec.frames(),
            os.path.join(
                a.video_dir, f"ep{ep:03d}_n{n}_{'success' if ok else 'fail'}.mp4"
            ),
            120.0 / a.record_every,
        )
    return rec, ok, step, collided


# 9. main (episode 반복 + 집계) ----------------------------------------------
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

    dt = sim.get_physics_dt()
    base_l = scene["robot_l"].data.root_pos_w[0].cpu().numpy()
    base_r = scene["robot_r"].data.root_pos_w[0].cpu().numpy()

    # 점진 저장: 파일을 먼저 열고, 성공 episode 는 즉시 쓰고 RAM 에서 버림
    h5f = grp = None
    if args.out:
        h5f, grp = open_hdf5(args.out, args)
    meta_dir = os.path.dirname(args.out) if args.out else args.video_dir

    stats = []
    n_ok = 0
    for ep in range(args.demos):
        n = int(rng.integers(nmin, nmax + 1))
        rec, ok, steps, col = run_episode(
            scene, sim, cam, args, ep, n, base_l, base_r, rng, dt
        )
        if ok:
            if grp is not None:
                append_demo(grp, rec, steps, n)
            n_ok += 1
        stats.append((ok, steps, n, col))
        del rec  # frame buffer 즉시 해제 (RAM 을 episode 1개분으로 유지)
        if meta_dir:
            write_meta(meta_dir, args, stats, n_ok)  # 매 episode 갱신 = 진행 확인용

    n_col = sum(1 for _, _, _, c in stats if c)
    print(
        f"\n[result] {n_ok}/{args.demos} = {100.0*n_ok/max(args.demos,1):.1f}%  "
        f"(collisions: {n_col})"
    )
    by_n = {}
    for ok, _, n, _ in stats:
        by_n.setdefault(n, [0, 0])
        by_n[n][0] += int(ok)
        by_n[n][1] += 1
    print(
        "  by-n: " + "  ".join(f"n={k}:{v[0]}/{v[1]}" for k, v in sorted(by_n.items()))
    )

    if h5f is not None:
        h5f.close()
        print(f"[hdf5] {args.out}  saved {n_ok}/{args.demos} successful demos")
    if meta_dir:
        print(f"[meta] {meta_dir}/run_meta.json")


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
        # simulation_app.close() 는 headless 에서 반환하지 않는 경우가 있음.
        # 호출하면 그 뒤 줄에 도달하지 못해 process 가 걸린 채 남음.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
