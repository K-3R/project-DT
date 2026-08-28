#!/usr/bin/env python
# [ver] dual_franka_office_sm.py 2026-08-10-r1  (ascii-only console/comments)
r"""
사무실 책상 태스크 -- 마커를 연필꽂이에 꽂는 씨앗 궤적 생성기.

태스크
    책상에 눕혀 흩어진 마커(지름 2.6cm, 길이 14cm) 1~3개를 연필꽂이에
    세워 꽂는다. 연필꽂이는 왼쪽 모니터 앞이라 왼팔만 닿는다.

    왼쪽 마커   왼팔이 집어서 세워 꽂는다
    오른쪽 마커 오른팔이 집어 공중에서 왼팔에 넘긴다 (mid-air handover,
               사용자 선택) -> 왼팔이 세워 꽂는다

핸드오버 기하
    오른팔이 마커를 수평(축=+x)으로 들고 핸드오버 지점에 대기.
    그립은 중심에서 +GRIP_OFF (긴 쪽이 -x). 왼팔이 -x 쪽 끝을 잡으면
    두 손목이 마커 축 방향으로 2*GRIP_OFF 떨어져 충돌하지 않는다.
    왼팔이 닫은 뒤 오른팔이 열고 물러난다.

세워 꽂기
    왼팔 그립은 항상 "중심-GRIP_OFF, 긴 쪽 +hand-x" 로 통일된다
    (직접 집기/핸드오버 모두). 손목을 world-y 로 +90도 돌리면(ERECT)
    hand-x 가 아래를 향해 마커가 수직으로 매달린다. 연필꽂이 위에서
    내려 놓으면 벽이 가이드가 되어 들어간다.

기록 (1번 환경과 동일 규약 + 3뷰)
    물리 120Hz + 매 스텝 렌더, 기록 20Hz (6스텝마다), 카메라 3뷰 256x256
    (ego_view + left/right_wrist_view -- docs/threeview_camera_upgrade.md,
    키 이름/순서는 변환기 VIDEO_KEYS 와 정확히 일치해야 한다),
    액션 16차원 [Lpos3 Lquat4 Lgrip1 | R 동일] = 월드 절대 TCP 명령(qw>=0),
    state joint 18, HDF5 점진 저장 (성공분만, flush), 충돌은 실패 처리.

사용
    CUDA_VISIBLE_DEVICES=1 isaaclab.sh -p dual_franka_office_sm.py --headless \
        --demos 2 --out /root/project/datasets/office_markers/seed.hdf5
"""

import argparse
import os
import sys
import traceback

# ================================================================== 1. 인자
parser = argparse.ArgumentParser(
    description="scripted seed-trajectory generator: markers into pencil holder"
)

g = parser.add_argument_group("run")
g.add_argument("--demos", type=int, default=5, help="number of attempts")
g.add_argument("--num-markers", default="1,3", help="marker count range min,max")
g.add_argument("--seed", type=int, default=0)
g.add_argument(
    "--steps-per-marker",
    type=int,
    default=2200,
    help="physics step budget per marker; limit = this x n (handover is slow)",
)
g.add_argument(
    "--record-every",
    type=int,
    default=6,
    help="record every N physics steps (120Hz/6 = 20Hz data); "
    "rendering itself happens every step",
)
g.add_argument(
    "--instruction",
    default="put all the markers into the pencil holder",
    help="language instruction stored with the dataset",
)
g.add_argument(
    "--collision-th",
    type=float,
    default=1.0,
    help="net contact force threshold [N] on link2-7; above = episode failed",
)

g = parser.add_argument_group("output")
g.add_argument("--out", default="", help="HDF5 output path (empty = no save)")
g.add_argument("--video-dir", default="", help="per-episode mp4 dir (empty = off)")
g.add_argument(
    "--stamp",
    type=int,
    default=1,
    help="append run timestamp to output paths so reruns never overwrite",
)
g.add_argument("--tag", default="", help="suffix after the timestamp")

g = parser.add_argument_group("marker layout")
g.add_argument(
    "--region",
    default="0.06,0.26,-0.26,0.38",
    help="marker spawn region xmin,xmax,ymin,ymax [m]; clustered near the "
    "desk center -- far corners are at the vertical-grasp reach limit "
    "(the wrist sits ~16cm above the TCP, shrinking usable radius) and "
    "the ego_view strip between the robots stays fully visible",
)
g.add_argument("--min-sep", type=float, default=0.11, help="min marker spacing [m]")
g.add_argument("--rand-tries", type=int, default=400, help="sampling retries")

g = parser.add_argument_group("motion")
g.add_argument("--hover", type=float, default=0.10, help="approach height [m]")
g.add_argument(
    "--grasp-z",
    type=float,
    default=0.0,
    help="TCP z offset from marker center at grasp. Round objects must be "
    "pinched at the equator: +4mm put the contact above center and the "
    "closing normal squeezed the marker downward out of the grip",
)
g.add_argument(
    "--grip-off",
    type=float,
    default=0.035,
    help="grip offset from marker center along its axis [m]; "
    "handover wrist separation = 2x this. Smaller = less gravity torque "
    "(0.05 made the marker tilt in the grip)",
)
g.add_argument(
    "--handover",
    default="0.20,0.00,0.28",
    help="handover point x,y,z (both arms reach it)",
)
g.add_argument(
    "--handover-off",
    type=float,
    default=0.055,
    help="left-arm grip offset at handover [m]; larger than grip-off so "
    "the two wrists are 9cm apart (7cm made the hands collide)",
)
g.add_argument(
    "--stage",
    default="0.24,-0.30,0.28",
    help="staging point before erecting the marker (left arm)",
)
g.add_argument(
    "--insert-bottom",
    type=float,
    default=0.05,
    help="marker bottom height above desk when released into the holder",
)
g.add_argument(
    "--standby-l",
    default="0.36,-0.34,0.30",
    help="left arm idle TCP",
)
g.add_argument(
    "--standby-r",
    default="0.36,0.34,0.30",
    help="right arm idle TCP",
)
g.add_argument(
    "--split-y",
    type=float,
    default=0.0,
    help="markers with y above this go to the right arm (handover), "
    "the rest to the left arm (desk split in half)",
)
g.add_argument("--pos-tol", type=float, default=0.008, help="reach tolerance [m]")
g.add_argument(
    "--ang-tol",
    type=float,
    default=0.12,
    help="orientation tolerance [rad]; without this the erect rotation "
    "'completed' instantly (position unchanged) and the descent started "
    "while the marker was still swinging to vertical",
)
g.add_argument("--dwell", type=int, default=25, help="settle steps at each waypoint")
g.add_argument(
    "--state-timeout",
    type=int,
    default=300,
    help="force-advance after this many steps in one primitive (0 = off)",
)
g.add_argument("--tool-offset", type=float, default=0.1034)

# ================================================================== 2. 앱 기동
from isaaclab.app import AppLauncher  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import office_scene as osc  # noqa: E402

osc.add_scene_args(parser)
# 학습 카메라 256x256, 로봇/연필꽂이 항상 켬. 마커 수는 씬에 최대로 만들어
# 두고 에피소드마다 필요한 만큼만 배치한다 (안 쓰는 것은 PARK).
parser.set_defaults(cam_w=256, cam_h=256, robots=1, items=osc.MAX_ITEMS, holder=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.robots = 1

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

if args.stamp:
    # 컨테이너가 UTC 라 KST 로 고정한다 (1번 환경과 동일)
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
    if args.out:
        print(f"[run] hdf5 -> {args.out}")

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

# ================================================================== 3. 수학
# 주의 (1차 실행에서 잡은 함정): 이 씬의 로봇 베이스는 z 축 180도로 돌아
# -x(책상)를 본다. 1번 환경의 DOWN_QUAT(Rx180, hand-x=+x)를 그대로 쓰면
# 손목이 베이스 정면 기준 180도 비틀린 자세를 요구받아 관절한계로 도달
# 불가 -> IK 가 팔을 휘둘러 책상에 충돌한다 (영상으로 확인).
# 그래서 "아래보기"를 베이스 기준으로 재정의한다:
#   DOWN_QUAT = Ry180 = (0,0,1,0): hand-z 아래, hand-x = world -x (팔 정면)
DOWN_QUAT = (0.0, 0.0, 1.0, 0.0)
GRIP_OPEN, GRIP_CLOSE = 0.04, 0.0


def vec(s):
    return tuple(float(x) for x in s.split(","))


def canonical_quat(q):
    """이중피복 제거: qw>=0 (w==0 이면 x>=0). 기록 부호 일관성용."""
    w, x, y, z = (float(v) for v in q)
    if w < 0.0 or (w == 0.0 and x < 0.0):
        return (-w, -x, -y, -z)
    return (w, x, y, z)


# 운반 자세 (팔끼리 교차 방지를 위해 좌우가 거울이다):
#   핸드오버에서 각 팔이 마커의 "자기 쪽" 끝을 잡아야 팔이 안 겹친다
#   (3차 실행에서 교차 설계로 팔끼리 충돌 -> 거울 규약으로 수정).
#   왼팔  CARRY_L = Rz(-90)*DOWN: hand-x = +y. 마커의 -y 쪽 끝을 잡고
#         긴 쪽(+hand-x = +y)이 오른팔 방향 -> ERECT_L 에서 아래로 간다
#   오른팔 CARRY_R = Rz(+90)*DOWN: hand-x = -y. +y 쪽 끝을 잡는다
# 세우기(왼팔만): ERECT_L = Rx(-90)*CARRY_L -> +hand-x(+y) 가 아래.
#   hand-z 는 -y 가 되어 손 몸통이 TCP 의 +y 쪽(중앙, 빈 공간)에 선다.
CARRY_L = canonical_quat(osc.quat_mul(osc.quat_axis("z", -90.0), DOWN_QUAT))
CARRY_R = canonical_quat(osc.quat_mul(osc.quat_axis("z", 90.0), DOWN_QUAT))
ERECT_L = canonical_quat(osc.quat_mul(osc.quat_axis("x", -90.0), CARRY_L))


def grasp_quat(theta):
    """축 방향 theta 인 마커를 가로질러 무는 자세.

    Rz(theta)*DOWN_QUAT 의 hand-x 는 -u(theta) 라 축과 평행(반대방향)이고,
    theta 를 (-90,90] 로 접으므로 손목 비틀림이 90도를 넘지 않는다.
    """
    return canonical_quat(
        osc.quat_mul(osc.quat_axis("z", np.degrees(theta)), DOWN_QUAT)
    )


def marker_axis(scene, key):
    """마커 축의 월드 방향각 (실린더 축 = 바디 z). (-90,90] 로 접는다."""
    w, x, y, z = scene[key].data.root_quat_w[0].cpu().numpy().tolist()
    # (0,0,1) 을 쿼터니언으로 회전한 결과의 xy 성분
    ax = 2.0 * (x * z + w * y)
    ay = 2.0 * (y * z - w * x)
    th = float(np.arctan2(ay, ax))
    if th > np.pi / 2:
        th -= np.pi
    elif th <= -np.pi / 2:
        th += np.pi
    return th


def count_inserted(scene, n, hx, hy):
    """연필꽂이 안에 들어간 마커 수."""
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


def check_success(scene, n, hx, hy):
    """모든 마커가 연필꽂이 안이면 성공."""
    return count_inserted(scene, n, hx, hy) == n


def grab_rgb(cam):
    rgb = cam.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    return rgb if rgb.dtype == np.uint8 else np.clip(rgb, 0, 255).astype(np.uint8)


def collided_now(scene, th):
    """link2~7 net 접촉력이 임계값을 넘었는가 (history 6 이 기록 사이 커버)."""
    for k in ("contact_l", "contact_r"):
        f = scene[k].data.net_forces_w_history
        if f is not None and float(torch.norm(f, dim=-1).max()) > th:
            return True
    return False


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


# ================================================================== 4. 팔
class Arm:
    """한 팔의 IK 와 그리퍼 (1번 환경 생성기에서 검증된 구현)."""

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
        self.target_w = None  # (hand pos, quat)
        self.tcp_target_w = None
        self.hold_here()

    def eef_pose_w(self):
        p = self.robot.data.body_state_w[:, self.body_id, 0:7]
        return p[:, 0:3], p[:, 3:7]

    def root_pose_w(self):
        s = self.robot.data.root_state_w[:, 0:7]
        return s[:, 0:3], s[:, 3:7]

    def tcp_now(self):
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

    def ang_err(self):
        """목표 자세와의 회전각 오차 [rad]."""
        if self.target_w is None or self.target_w[1] is None:
            return 0.0
        _, eq = self.eef_pose_w()
        q = eq[0].cpu().numpy()
        t = np.asarray(self.target_w[1])
        dot = abs(float(np.dot(q, t)))
        return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))

    def aim(self, tcp_pos_w, quat_w=DOWN_QUAT):
        """월드 TCP 목표 -> hand 원점 목표 -> 루트 프레임 IK 명령."""
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
        """명령 공백 금지: 지금 자세를 목표로 세운다."""
        ep, eq = self.eef_pose_w()
        rp, rq = self.root_pose_w()
        bp, bq = subtract_frame_transforms(rp, rq, ep, eq)
        self.ik.set_command(torch.cat([bp, bq], dim=-1))
        self.target_w = (
            tuple(ep[0].cpu().numpy().tolist()),
            canonical_quat(eq[0].cpu().numpy().tolist()),
        )
        # 라벨은 TCP 로 기록한다. hand 원점을 넣으면 정지 팔의 액션 라벨이
        # tool_offset(0.1034) 만큼 z 로 밀린다 (env1 0.4절 결함의 상속 --
        # env2 는 생성 전이라 여기서 바로잡는다. IK 명령은 hand 기준 그대로)
        self.tcp_target_w = tuple(float(v) for v in self.tcp_now())

    def apply(self):
        jac = self.robot.root_physx_view.get_jacobians()[
            :, self.jac_id, :, self.arm_ids
        ]
        ep, eq = self.eef_pose_w()
        rp, rq = self.root_pose_w()
        # PhysX 자코비안은 "월드" 프레임 기준인데 IK 오차는 "루트" 프레임에서
        # 계산한다. 베이스가 z180 으로 돌아 있는 이 씬에서 회전을 안 되돌리면
        # 방향이 반전돼 팔이 엉뚱한 곳으로 간다 (2차 실행에서 실측 확인).
        # 공식 구현과 동일한 변환: task_space_actions.py 의 jacobian_b.
        # 1번 환경은 베이스가 identity 라 생략해도 우연히 맞았던 것이다.
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


# ================================================================== 5. 프로그램
class Program:
    """순차 프리미티브 실행기.

    이 태스크는 팔 둘이 교대로 움직이는 순차 흐름이라 (핸드오버 순간만
    맞물림), 팔별 상태기계 대신 프리미티브 목록이 단순하고 안전하다.
    각 프리미티브: init(1회) -> done() 이 dwell 동안 참 -> 다음.
    state-timeout 이 지나면 경고 후 강제 전진한다 (1번 환경과 동일 안전장치).
    """

    def __init__(self, a):
        self.a = a
        self.items = []  # (name, init_fn, done_fn, dwell)
        self.i = -1
        self.timer = 0
        self.elapsed = 0
        self.done = False

    def add(self, name, init_fn, done_fn, dwell=None):
        self.items.append((name, init_fn, done_fn, dwell))

    def move(self, name, arm, pos, quat=DOWN_QUAT):
        """pos/quat 는 값이거나 () -> 값 함수 (런타임 계산용).

        완료 = 위치 AND 자세 도달. 자세를 안 보면 제자리 회전(세우기 등)이
        즉시 통과돼 다음 동작이 회전 중에 시작된다 (실측: 하강 중 세워짐).
        """

        def init():
            p = pos() if callable(pos) else pos
            q = quat() if callable(quat) else quat
            arm.aim(p, q)

        self.add(
            name,
            init,
            lambda: arm.dist() < self.a.pos_tol and arm.ang_err() < self.a.ang_tol,
        )

    def gripper(self, name, arm, val, wait):
        def init():
            arm.grip = val

        # 손가락이 움직일 시간만 준다 (조건 없음)
        self.add(name, init, lambda: True, dwell=wait)

    def step(self):
        """물리 스텝마다 호출. 프로그램이 끝나면 True."""
        if self.done:
            return True
        if self.i < 0:
            self._advance()
        name, _init, done_fn, dwell = self.items[self.i]
        need = self.a.dwell if dwell is None else dwell
        self.elapsed += 1
        if done_fn():
            self.timer += 1
            if self.timer >= need:
                self._advance()
        else:
            self.timer = 0
        if (
            not self.done
            and self.a.state_timeout
            and self.elapsed > self.a.state_timeout
        ):
            print(f"  [!] timeout in '{self.items[self.i][0]}' -> forcing next")
            self._advance()
        return self.done

    def _advance(self):
        self.i += 1
        self.timer = 0
        self.elapsed = 0
        if self.i >= len(self.items):
            self.done = True
            return
        name, init_fn, _d, _w = self.items[self.i]
        print(f"  [prim {self.i:2d}] {name}")
        init_fn()


def build_program(a, scene, arm_l, arm_r, order, bases):
    """에피소드 프로그램을 만든다. order = [(marker_key, need_handover)].

    부호 규약 (DOWN_QUAT = Ry180 기준, 거울 운반 규약):
      grasp_quat(th) 의 hand-x = -u(th). 두 팔 모두 그립 = 중심 + off*u.
      reorient 로 왼팔은 CARRY_L(hand-x=+y), 오른팔은 CARRY_R(hand-x=-y).
        회전 후 물리 그립 위치: 왼팔 = 중심의 -y 쪽 / 오른팔 = +y 쪽
        = 핸드오버에서 각자 자기 진영의 끝을 잡는다 (팔 교차 없음).
      왼팔의 긴 쪽은 +hand-x(+y) -> ERECT_L 에서 아래로 늘어진다.
    """
    prog = Program(a)
    hx, hy = vec(a.holder_xy)
    hoz = vec(a.handover)
    stage = vec(a.stage)
    sb_l, sb_r = vec(a.standby_l), vec(a.standby_r)
    off = a.grip_off
    half = osc.MARKER_L / 2.0
    wall_top = osc.HOLDER_WALL + osc.HOLDER_INNER[2]
    hang = half + off  # ERECT 후 그립 아래로 늘어지는 길이
    bb = {}

    def pick_info(key, sign):
        """집기 좌표: 중심 +- off 를 축 방향으로. 리셋 후 고정값을 쓴다."""

        def calc():
            p = scene[key].data.root_pos_w[0].cpu().numpy()
            th = marker_axis(scene, key)
            u = np.array([np.cos(th), np.sin(th), 0.0])
            g = p[:3] + sign * off * u
            bb[key] = (g, th)

        return calc

    def pick(tag, arm, key, carry, lift_z):
        """공통 집기: 측정 -> 위 -> 파지점 -> 닫기 -> 들어올려 운반자세."""
        prog.add(f"{tag}:measure", pick_info(key, +1.0), lambda: True, dwell=1)
        prog.move(
            f"{tag}:above_pick",
            arm,
            lambda k=key: (bb[k][0][0], bb[k][0][1], osc.MARKER_R + a.hover),
            lambda k=key: grasp_quat(bb[k][1]),
        )
        prog.move(
            f"{tag}:at_pick",
            arm,
            lambda k=key: (bb[k][0][0], bb[k][0][1], osc.MARKER_R + a.grasp_z),
            lambda k=key: grasp_quat(bb[k][1]),
        )
        prog.gripper(f"{tag}:close", arm, GRIP_CLOSE, a.dwell * 2)
        prog.move(
            f"{tag}:lift",
            arm,
            lambda k=key: (bb[k][0][0], bb[k][0][1], lift_z),
            lambda k=key: grasp_quat(bb[k][1]),
        )
        # 같은 자리에서 운반 자세로 재정렬 (마커 축 -> world y)
        prog.move(
            f"{tag}:reorient",
            arm,
            lambda k=key: (bb[k][0][0], bb[k][0][1], lift_z),
            carry,
        )

    # 삽입 슬롯: 매번 정중앙에 꽂으면 3개째부터 먼저 선 마커와 부딪힌다
    # (실측: 2개 성공 후 3개째 실패). 통 안 네 구석을 순서대로 쓴다.
    # 안쪽 17cm(2배 확대) 기준: +-3.5cm 면 벽(여유 3cm)과 이웃(7cm 간격)
    # 모두 넉넉하다
    SLOTS = [(-0.035, -0.035), (0.035, 0.035), (-0.035, 0.035), (0.035, -0.035)]

    def insert(tag, l_off, slot_i):
        """왼팔이 (그립 = 중심 -y 쪽, 긴쪽 +y) 로 쥔 마커를 세워 꽂는다.

        l_off = 왼팔 그립의 중심 오프셋. 직접 집기는 grip_off, 핸드오버는
        handover_off 라 hang(그립 아래로 늘어지는 길이)이 경로마다 다르다.
        slot_i = 몇 번째 삽입인가 (슬롯 선택).
        """
        hang2 = half + l_off
        sx_, sy_ = SLOTS[slot_i % len(SLOTS)]
        tx, ty = hx + sx_, hy + sy_
        prog.move(f"{tag}:stage", arm_l, stage, CARRY_L)
        # 세우기: +hand-x(+y) 가 아래로. 손 몸통은 TCP 의 +y 쪽(중앙 빈 공간)
        prog.move(f"{tag}:erect", arm_l, stage, ERECT_L)
        prog.move(
            f"{tag}:above_holder",
            arm_l,
            (tx, ty, wall_top + hang2 + 0.04),
            ERECT_L,
        )
        prog.move(f"{tag}:insert", arm_l, (tx, ty, a.insert_bottom + hang2), ERECT_L)
        prog.gripper(f"{tag}:open", arm_l, GRIP_OPEN, a.dwell)
        # 수직 이탈: 마커가 아래에 서 있으므로 곧장 위로만 뺀다
        prog.move(f"{tag}:up", arm_l, (tx, ty, wall_top + hang2 + 0.08), ERECT_L)
        prog.move(f"{tag}:standby", arm_l, sb_l, DOWN_QUAT)

    # 시작 전 두 팔을 각자 대기 위치로 (기본 자세가 작업 구역 위에 있어서
    # 상대 팔 경로를 막는 것을 방지)
    prog.move("init:l_standby", arm_l, sb_l, DOWN_QUAT)
    prog.move("init:r_standby", arm_r, sb_r, DOWN_QUAT)

    for idx, (key, handover) in enumerate(order):
        tag = key
        if not handover:
            # ---- 왼팔 직접: 집고 세워 꽂는다 ----
            pick(tag, arm_l, key, CARRY_L, 0.22)
            insert(tag, off, idx)
        else:
            # ---- 오른팔이 집어 핸드오버 -> 왼팔이 꽂는다 ----
            pick(f"{tag}:R", arm_r, key, CARRY_R, 0.26)
            # 핸드오버: 마커 축 = y, 중심이 hoz. 오른팔 그립은 +y 쪽(자기 진영)
            prog.move(
                f"{tag}:r_handover",
                arm_r,
                (hoz[0], hoz[1] + off, hoz[2]),
                CARRY_R,
            )
            # 왼팔: -y 쪽(자기 진영)에서 접근해 반대쪽 끝을 잡는다.
            # 파지점은 handover_off 로 더 바깥 -> 손목 간격 off+handover_off
            prog.move(
                f"{tag}:l_approach",
                arm_l,
                (hoz[0], hoz[1] - a.handover_off - 0.10, hoz[2] + 0.07),
                CARRY_L,
            )
            prog.move(
                f"{tag}:l_at_handover",
                arm_l,
                (hoz[0], hoz[1] - a.handover_off, hoz[2]),
                CARRY_L,
            )
            prog.gripper(f"{tag}:l_close", arm_l, GRIP_CLOSE, a.dwell * 2)
            prog.gripper(f"{tag}:r_open", arm_r, GRIP_OPEN, a.dwell)
            # 오른팔 이탈: 마커 축을 따라 자기 쪽(+y)으로 빠진 뒤 대기 위치로
            prog.move(
                f"{tag}:r_retreat",
                arm_r,
                (hoz[0], hoz[1] + off + 0.16, hoz[2] + 0.03),
                CARRY_R,
            )
            prog.move(f"{tag}:r_standby", arm_r, sb_r, DOWN_QUAT)
            insert(tag, a.handover_off, idx)

    return prog, bb


# ================================================================== 6. 기록
class Recorder:
    # 카메라 키 이름은 변환기(convert_bimanual_lerobot.py VIDEO_KEYS)와
    # 정확히 일치해야 한다. 순서 = 카메라 정체성 (GR00T 는 ID 임베딩 없음)
    VIEW_KEYS = ("ego_view", "left_wrist_view", "right_wrist_view")
    KEYS = (
        "actions",
        "joint_pos",
        "eef_pos_l",
        "eef_quat_l",
        "grip_l",
        "eef_pos_r",
        "eef_quat_r",
        "grip_r",
        "item_pos",
        "item_quat",
        "insert_done",
    ) + VIEW_KEYS

    def __init__(self, n):
        self.n = n
        self.buf = {k: [] for k in self.KEYS}

    def add(self, scene, arm_l, arm_r, frame, wrist_l, wrist_r, n_inserted):
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
        keys = [f"item_{i+1}" for i in range(self.n)]
        self.buf["item_pos"].append(
            np.concatenate([scene[k].data.root_pos_w[0].cpu().numpy() for k in keys])
        )
        self.buf["item_quat"].append(
            np.concatenate([scene[k].data.root_quat_w[0].cpu().numpy() for k in keys])
        )
        self.buf["insert_done"].append(float(n_inserted))
        self.buf["ego_view"].append(frame)
        self.buf["left_wrist_view"].append(wrist_l)
        self.buf["right_wrist_view"].append(wrist_r)

    def arrays(self):
        out = {}
        for k, v in self.buf.items():
            if k in self.VIEW_KEYS:
                out[k] = np.asarray(v, dtype=np.uint8)
            else:
                out[k] = np.array(v, np.float32)
        return out

    def frames(self, key="ego_view"):
        return self.buf[key]


def open_hdf5(path, a):
    """점진 저장용 HDF5 (배치 시작 시 1회). RAM 상수 + 크래시 내성."""
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
    """성공 에피소드 하나를 즉시 기록하고 flush 한다."""
    i = int(grp.attrs["total"])
    arr = rec.arrays()
    d = grp.create_group(f"demo_{i}")
    d.create_dataset("actions", data=arr["actions"])
    d.attrs["num_samples"] = int(arr["actions"].shape[0])
    d.attrs["physics_steps"] = steps
    d.attrs["success"] = True
    d.attrs["num_markers"] = n
    obs = d.create_group("obs")
    for k, v in arr.items():
        if k in ("actions", "insert_done"):
            continue
        if k in Recorder.VIEW_KEYS:
            obs.create_dataset(k, data=v, compression="gzip", compression_opts=1)
        else:
            obs.create_dataset(k, data=v)
    sub = d.create_group("subtask")
    sub.create_dataset("insert_done", data=arr["insert_done"])
    grp.attrs["total"] = i + 1
    grp.file.flush()


def write_meta(meta_dir, a, stats, n_ok):
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
                        "num_markers",
                        "region",
                        "min_sep",
                        "hover",
                        "grasp_z",
                        "grip_off",
                        "handover",
                        "stage",
                        "insert_bottom",
                        "pos_tol",
                        "dwell",
                        "state_timeout",
                        "steps_per_marker",
                        "seed",
                        "tool_offset",
                        "record_every",
                        "collision_th",
                        "holder_xy",
                        "base_x",
                        "base_sep",
                        "base_z",
                        "wrist_fov",
                    )
                },
            },
            f,
            indent=2,
        )


# ================================================================== 7. 배치/리셋
# 마커를 뿌리면 안 되는 자리 (키보드/마우스패드/연필꽂이 주변, 여유 포함)
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
        # 연필꽂이 주변은 넓게 비운다: 그리퍼 손 폭(~9cm) + 파지 오프셋 때문에
        # 가까이 놓인 마커는 손이 연필꽂이에 걸려 못 집는다 (실측).
        # 통이 2배(17cm)로 커져서 금지 반경도 넓혔다
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
    """로봇/마커 리셋 + 물리 안정화 + 렌더 워밍업 (1번 환경 규약)."""
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
    for _ in range(90):  # 안정화 + temporal 누적버퍼 워밍업 (이전 잔상 제거)
        sim.step(render=True)
        scene.update(dt)


# ================================================================== 8. 에피소드
class HdWriters:
    """쇼케이스 HD 영상 스트리밍 기록 (버퍼링하면 수 GB 라 바로 파일에 쓴다)."""

    def __init__(self, a, ep, n, hd_cams):
        self.cams = hd_cams
        self.writers = []
        if not hd_cams or not a.video_dir:
            return
        import imageio

        os.makedirs(a.video_dir, exist_ok=True)
        fps = 120.0 / a.record_every
        for i in range(len(hd_cams)):
            path = os.path.join(a.video_dir, f"ep{ep:03d}_n{n}_hd_view{i}.mp4")
            self.writers.append(
                imageio.get_writer(
                    path, fps=fps, codec="libx264", pixelformat="yuv420p"
                )
            )

    def add(self):
        for cam, w in zip(self.cams, self.writers):
            w.append_data(grab_rgb(cam))

    def close(self):
        for w in self.writers:
            w.close()
        if self.writers:
            print(f"[vid] {len(self.writers)} hd view(s) saved")


def run_episode(scene, sim, cam, wcam_l, wcam_r, a, ep, n, bases, rng, dt, hd_cams=()):
    reset_episode(scene, sim, rng, a, n, dt)
    arm_l = Arm(scene, "robot_l", sim, a)
    arm_r = Arm(scene, "robot_r", sim, a)

    # 담당 배정 (사용자 규칙): 책상을 반으로 나눠 y > split_y 는 오른팔
    # (핸드오버 경유), 나머지는 왼팔 직접. 왼팔 리치 밖이면 무조건 핸드오버.
    order = []
    bl = np.array(bases[0])
    for i in range(1, n + 1):
        p = scene[f"item_{i}"].data.root_pos_w[0].cpu().numpy()
        reach_l = float(np.linalg.norm(p - bl))
        handover = (p[1] > a.split_y) or (reach_l > 0.72)
        order.append((f"item_{i}", handover))
    order.sort(key=lambda t: t[1])  # 직접 집는 것 먼저 (실패 시 빨리 끝난다)
    print(
        f"  markers n={n}  "
        + "  ".join(f"{k}:{'handover' if h else 'direct'}" for k, h in order)
    )

    prog, _bb = build_program(a, scene, arm_l, arm_r, order, bases)
    hx, hy = vec(a.holder_xy)

    rec = Recorder(n)
    hd = HdWriters(a, ep, n, hd_cams)
    limit = a.steps_per_marker * n
    step = 0
    collided = False
    while step < limit:
        finished = prog.step()
        arm_l.apply()
        arm_r.apply()
        scene.write_data_to_sim()
        sim.step(render=True)  # 매 스텝 렌더 (일반 방식, 잔상 없음)
        scene.update(dt)
        step += 1

        if step % max(a.record_every, 1) == 0:
            n_in = count_inserted(scene, n, hx, hy)
            rec.add(
                scene,
                arm_l,
                arm_r,
                grab_rgb(cam),
                grab_rgb(wcam_l),
                grab_rgb(wcam_r),
                n_in,
            )
            hd.add()
            if collided_now(scene, a.collision_th):
                collided = True
                break
        if finished:
            break

    hd.close()
    ok = check_success(scene, n, hx, hy) and not collided
    print(f"[ep {ep+1}/{a.demos}] n={n} steps={step} success={ok} collided={collided}")
    if a.video_dir:
        # ego + 손목 2뷰 전부 저장 (손목 화각 육안 판정용)
        stem = f"ep{ep:03d}_n{n}_{'success' if ok else 'fail'}"
        fps = 120.0 / a.record_every
        save_video(rec.frames(), os.path.join(a.video_dir, f"{stem}.mp4"), fps)
        for key, suffix in (
            ("left_wrist_view", "wristL"),
            ("right_wrist_view", "wristR"),
        ):
            save_video(
                rec.frames(key),
                os.path.join(a.video_dir, f"{stem}_{suffix}.mp4"),
                fps,
            )
    return rec, ok, step, collided


# ================================================================== 9. main
def main():
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    nmin, nmax = (int(v) for v in args.num_markers.split(","))

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device))
    scene = InteractiveScene(osc.build(args, with_camera=True))
    sim.reset()
    cam = scene["scene_cam"]
    cam.set_world_poses_from_view(
        eyes=torch.tensor([osc.EGO_EYE], device=sim.device),
        targets=torch.tensor([osc.EGO_TARGET], device=sim.device),
    )
    # 손목 카메라는 panda_hand 에 OffsetCfg 로 붙어 있어 포즈 세팅 불필요
    wcam_l = scene["wrist_cam_l"]
    wcam_r = scene["wrist_cam_r"]
    dt = sim.get_physics_dt()
    bases = (
        scene["robot_l"].data.root_pos_w[0].cpu().numpy(),
        scene["robot_r"].data.root_pos_w[0].cpu().numpy(),
    )

    # 쇼케이스 HD 카메라 배치 (있을 때만)
    hd_cams = []
    if args.hd_views.strip():
        tgt = torch.tensor([osc.EGO_TARGET], device=sim.device)
        for i, v in enumerate([s for s in args.hd_views.split("|") if s.strip()]):
            c = scene[f"hd_cam_{i}"]
            c.set_world_poses_from_view(
                eyes=torch.tensor([vec(v)], device=sim.device), targets=tgt
            )
            hd_cams.append(c)
        print(f"[hd] {len(hd_cams)} showcase camera(s) at {args.hd_w}x{args.hd_h}")

    h5f = grp = None
    if args.out:
        h5f, grp = open_hdf5(args.out, args)
    meta_dir = os.path.dirname(args.out) if args.out else args.video_dir

    stats = []
    n_ok = 0
    for ep in range(args.demos):
        n = int(rng.integers(nmin, nmax + 1))
        rec, ok, steps, col = run_episode(
            scene, sim, cam, wcam_l, wcam_r, args, ep, n, bases, rng, dt, hd_cams
        )
        if ok:
            if grp is not None:
                append_demo(grp, rec, steps, n)
            n_ok += 1
        stats.append((ok, steps, n, col))
        del rec  # 프레임 버퍼 즉시 해제 (RAM 상수 유지)
        if meta_dir:
            write_meta(meta_dir, args, stats, n_ok)

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
        # simulation_app.close() 는 헤드리스에서 반환하지 않는 경우가 있어
        # 결과를 디스크에 쓴 뒤 바로 프로세스를 끝낸다 (좀비 방지)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
