#!/usr/bin/env bash
# [ver] run_eval.sh 2026-08-26-r2
# r2: GPU 기본값 제거 (공용 서버 규약 -- 명시 필수)
#
# Isaac Lab Franka 태스크 폐루프. run_server.sh 를 먼저 띄운 뒤 실행.
# Isaac Lab 은 컨테이너 안에 있으므로 docker exec 로 들어간다.
#
#   CLIENT_GPU=3 EPISODES=5 ./run_eval.sh
#   DRY_RUN=1 CLIENT_GPU=3 ./run_eval.sh   # 서버 없이 관측 키/shape 만 확인
set -euo pipefail

CLIENT_GPU="${CLIENT_GPU:-}"
if [ -z "$CLIENT_GPU" ]; then
  echo "[eval] ERROR: set CLIENT_GPU explicitly (shared server), e.g. CLIENT_GPU=3 ..."
  exit 1
fi
EPISODES="${EPISODES:-5}"
PORT="${PORT:-5555}"
TASK="${TASK:-Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0}"
TEXT="${TEXT:-stack the cubes}"
GAIN="${GAIN:-1.0}"
GRIP_SIGN="${GRIP_SIGN:-1.0}"
CAM_SIZE="${CAM_SIZE:-256}"
CONTAINER="${CONTAINER:-gr00t_isaac}"
# 컨테이너의 /root/project 는 호스트 ~/project/gr00t_Isaacsim 의 바인드 마운트다.
# 따라서 호스트에서 파일을 고치면 컨테이너에 즉시 반영된다 (docker cp 불필요).
ISAACLAB="${ISAACLAB:-/root/project/IsaacLab}"
SCRIPT_DIR_IN_CONTAINER="${SCRIPT_DIR_IN_CONTAINER:-/root/project/isaac_franka/legacy_libero}"

# 산출물. 컨테이너의 /root/project 는 호스트 마운트라 호스트에서 바로 열린다.
RUN_TAG="${RUN_TAG:-run}"
VIDEO_DIR="${VIDEO_DIR:-/root/project/out/legacy_libero/$RUN_TAG/videos}"
OUT_DIR="${OUT_DIR:-/root/project/out/legacy_libero/$RUN_TAG}"
VIDEO_WRIST="${VIDEO_WRIST:-0}"

EXTRA=""
if [ "${DRY_RUN:-0}" = "1" ]; then
  EXTRA="--dry-run"
else
  EXTRA="--video-dir $VIDEO_DIR --out-dir $OUT_DIR"
  [ "$VIDEO_WRIST" = "1" ] && EXTRA="$EXTRA --video-wrist"
fi

echo "[eval] gpu=$CLIENT_GPU task=$TASK episodes=$EPISODES port=$PORT"

# -e TERM=xterm : docker exec 에 TTY 가 없으면 TERM 이 비어
#   로그인 셸 초기화의 tabs/tput 이 "'ansi+tabs': unknown terminal type" 로 죽는다.
#   (우리 스크립트가 시작도 못 하고 즉시 리턴하는 증상)
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc "
umask 000 &&
CUDA_VISIBLE_DEVICES=$CLIENT_GPU $ISAACLAB/isaaclab.sh -p \
$SCRIPT_DIR_IN_CONTAINER/isaac_franka_gr00t.py \
  --task '$TASK' \
  --episodes $EPISODES \
  --port $PORT \
  --text '$TEXT' \
  --cam-size $CAM_SIZE \
  --action-gain $GAIN \
  --gripper-sign $GRIP_SIGN \
  --headless $EXTRA
"
