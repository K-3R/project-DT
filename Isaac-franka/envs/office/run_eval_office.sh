#!/usr/bin/env bash
# ======================================
# File: run_eval_office.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_eval_office.sh 2026-08-31
# =============================================================================
# 사무실 마커 task 폐루프 평가 실행 (host 에서 실행; client 는 container 안)
#
# 순서: host 에서 server 먼저 (bimanual server script 재사용, CKPT 만 교체):
#   SERVER_GPU=2 PORT=5561 CKPT=/data1/huggingface/sslunder54/checkpoints/office_3view \
#     bash <클론루트>/Isaac-franka/train/run_server_finetuned.sh
# 그 다음 이것:
#   CLIENT_GPU=2 EPISODES_PER_N=10 VIDEO=1 PORT=5561 bash run_eval_office.sh
#   DRY_RUN=1 CLIENT_GPU=2 bash run_eval_office.sh  (server 없이 관측 규약만 확인)
#
# 산출물: container /root/project/out/eval_office/<KST스탬프>/
#         (host <클론루트>/out/eval_office/... 로 보임)
# =============================================================================
set -euo pipefail

CLIENT_GPU="${CLIENT_GPU:-}"
EPISODES_PER_N="${EPISODES_PER_N:-10}"
PORT="${PORT:-5555}"
SEED="${SEED:-1000}"
CHUNK="${CHUNK:-16}"
NUM_MARKERS="${NUM_MARKERS:-1,2}"
STEPS_PER_MARKER="${STEPS_PER_MARKER:-400}"
CONTAINER="${CONTAINER:-gr00t_isaac}"
ISAACLAB="${ISAACLAB:-/root/project/IsaacLab}"
SCRIPT_DIR="${SCRIPT_DIR:-/root/project/Isaac-franka/envs/office}"
OUT_DIR="${OUT_DIR:-/root/project/out/eval_office}"
VIDEO="${VIDEO:-0}"

# 공용 server 규약: GPU 는 반드시 명시 (조용한 기본값 금지. DRY_RUN 도
# container 에서 Isaac 을 기동하므로 GPU 를 점유함)
if [ -z "$CLIENT_GPU" ]; then
    echo "[eval] ERROR: set CLIENT_GPU explicitly (shared server), e.g. CLIENT_GPU=2 ..."
    exit 1
fi

EXTRA=""
if [ "${DRY_RUN:-0}" = "1" ]; then
    EXTRA="--dry-run"
else
    EXTRA="--out-dir $OUT_DIR"
    [ "$VIDEO" = "1" ] && EXTRA="$EXTRA --video-dir $OUT_DIR/videos"
fi

echo "[eval] gpu=$CLIENT_GPU episodes_per_n=$EPISODES_PER_N port=$PORT seed=$SEED"
echo "[eval] markers=$NUM_MARKERS steps_per_marker=$STEPS_PER_MARKER"

docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc "
umask 000 &&
cd $SCRIPT_DIR &&
CUDA_VISIBLE_DEVICES=$CLIENT_GPU $ISAACLAB/isaaclab.sh -p \
eval_office.py \
  --headless \
  --episodes-per-n $EPISODES_PER_N \
  --num-markers $NUM_MARKERS \
  --steps-per-marker $STEPS_PER_MARKER \
  --port $PORT \
  --seed $SEED \
  --action-chunk $CHUNK \
  $EXTRA
"
