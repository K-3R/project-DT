#!/usr/bin/env bash
# ======================================
# File: run_eval_bimanual.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_eval_bimanual.sh 2026-08-31
# =============================================================================
# 양팔 Franka 폐루프 평가 실행 (host 에서 실행; client 는 container 안)
#
# 순서: host 에서 server 먼저 (train/run_server_finetuned.sh), 그 다음 이것.
#
#   CLIENT_GPU=1 EPISODES_PER_N=10 bash run_eval_bimanual.sh
#   DRY_RUN=1 CLIENT_GPU=1 bash run_eval_bimanual.sh  (server 없이 관측 규약만 확인)
#
# 산출물: container /root/project/out/eval_bimanual/<KST stamp>/
#         (host ~/project/gr00t_Isaacsim/out/eval_bimanual/... 로 보임)
# =============================================================================
set -euo pipefail

CLIENT_GPU="${CLIENT_GPU:-}"
EPISODES_PER_N="${EPISODES_PER_N:-10}"
PORT="${PORT:-5555}"
SEED="${SEED:-1000}"
CHUNK="${CHUNK:-16}"
NUM_CUBES="${NUM_CUBES:-2,5}"
STEPS_PER_CUBE="${STEPS_PER_CUBE:-250}"
CONTAINER="${CONTAINER:-gr00t_isaac}"
ISAACLAB="${ISAACLAB:-/root/project/IsaacLab}"
SCRIPT_DIR="${SCRIPT_DIR:-/root/project/Isaac-franka/envs/bimanual}"
OUT_DIR="${OUT_DIR:-/root/project/out/eval_bimanual}"
VIDEO="${VIDEO:-0}"

# 공용 server 규약: GPU 는 반드시 명시 (조용한 기본값 금지. DRY_RUN 도
# container 에서 Isaac 을 기동하므로 GPU 를 점유함)
if [ -z "$CLIENT_GPU" ]; then
    echo "[eval] ERROR: set CLIENT_GPU explicitly (shared server), e.g. CLIENT_GPU=1 ..."
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

docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc "
umask 000 &&
cd $SCRIPT_DIR &&
CUDA_VISIBLE_DEVICES=$CLIENT_GPU $ISAACLAB/isaaclab.sh -p \
eval_bimanual.py \
  --headless \
  --episodes-per-n $EPISODES_PER_N \
  --num-cubes $NUM_CUBES \
  --steps-per-cube $STEPS_PER_CUBE \
  --port $PORT \
  --seed $SEED \
  --action-chunk $CHUNK \
  $EXTRA
"
