#!/usr/bin/env bash
# ======================================
# File: run_eval_office_scan.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_eval_office_scan.sh 2026-08-31
# =============================================================================
# scan 책상판 office marker task closed-loop 평가 (host 에서 실행)
#
# run_eval_office.sh 의 scan 배경판. server 는 office 것 그대로:
#   SERVER_GPU=2 PORT=5561 CKPT=/data1/huggingface/sslunder54/checkpoints/office_3view \
#     bash <클론루트>/Isaac-franka/train/run_server_finetuned.sh
# 그 다음 이것:
#   CLIENT_GPU=2 EPISODES_PER_N=10 VIDEO=1 PORT=5561 bash run_eval_office_scan.sh
#   DRY_RUN=1 CLIENT_GPU=2 bash run_eval_office_scan.sh  (server 없이 관측 규약만)
#
# scan/batch/색 확정값(protocol v1)은 office_scan_scene.py 상단 상수가
# 정본 -- 이 runner 는 그 값들을 전달하지 않음 (바꾸려면 scene 모듈 상수
# 수정 = benchmark 버전업). 실험용 1회 변경은 EXTRA_ARGS 로:
#   EXTRA_ARGS="--scan-usd /root/project/datasets/other.usd" bash ...
# 산출물: container /root/project/out/eval_office_scan/<KST stamp>/
#   (host <클론루트>/out/eval_office_scan/... 로 보임)
#   scene log 의 "[office-scan] effective: {...}" 한 줄이 그 run 의
#   protocol/자산/batch 기록임
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
SCRIPT_DIR="${SCRIPT_DIR:-/root/project/Isaac-franka/envs/office_scan}"
OUT_DIR="${OUT_DIR:-/root/project/out/eval_office_scan}"
VIDEO="${VIDEO:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # 1회성 실험용 추가 flag (header 참조)

# 공용 server 규약: GPU 는 반드시 명시 (조용한 기본값 금지. DRY_RUN 도
# container 에서 Isaac 을 기동하므로 GPU 점유 발생)
if [ -z "$CLIENT_GPU" ]; then
    echo "[eval-office-scan] ERROR: set CLIENT_GPU explicitly (shared server), e.g. CLIENT_GPU=2 ..."
    exit 1
fi

EXTRA=""
if [ "${DRY_RUN:-0}" = "1" ]; then
    EXTRA="--dry-run"
else
    EXTRA="--out-dir $OUT_DIR"
    [ "$VIDEO" = "1" ] && EXTRA="$EXTRA --video-dir $OUT_DIR/videos"
fi

echo "[eval-office-scan] gpu=$CLIENT_GPU episodes_per_n=$EPISODES_PER_N port=$PORT seed=$SEED"
echo "[eval-office-scan] markers=$NUM_MARKERS steps_per_marker=$STEPS_PER_MARKER"

docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc "
umask 000 &&
cd $SCRIPT_DIR &&
CUDA_VISIBLE_DEVICES=$CLIENT_GPU $ISAACLAB/isaaclab.sh -p \
eval_office_scan.py \
  --headless \
  --episodes-per-n $EPISODES_PER_N \
  --num-markers $NUM_MARKERS \
  --steps-per-marker $STEPS_PER_MARKER \
  --port $PORT \
  --seed $SEED \
  --action-chunk $CHUNK \
  $EXTRA_ARGS \
  $EXTRA
"
