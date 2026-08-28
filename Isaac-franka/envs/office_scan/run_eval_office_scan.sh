#!/usr/bin/env bash
# =============================================================================
# 스캔 책상판 office 마커 태스크 폐루프 평가 (호스트에서 실행)
#
# run_eval_office.sh 의 스캔 배경판. 서버는 office 것 그대로:
#   SERVER_GPU=2 PORT=5561 CKPT=/data1/huggingface/sslunder54/checkpoints/office_3view \
#     bash ~/project/gr00t_Isaacsim/isaac_franka/train/run_server_finetuned.sh
# 그 다음 이것:
#   CLIENT_GPU=2 EPISODES_PER_N=10 VIDEO=1 PORT=5561 bash run_eval_office_scan.sh
#   DRY_RUN=1 CLIENT_GPU=2 bash run_eval_office_scan.sh  (서버 없이 관측 규약만)
#
# 스캔/배치/색 확정값(프로토콜 v1)은 office_scan_scene.py 상단 상수가
# 정본이다 -- 이 러너는 그 값들을 전달하지 않는다 (바꾸려면 씬 모듈 상수
# 수정 = 벤치마크 버전업). 실험용 1회 변경은 EXTRA_ARGS 로:
#   EXTRA_ARGS="--scan-usd /root/project/datasets/other.usd" bash ...
# 산출물: 컨테이너 /root/project/out/eval_office_scan/<KST스탬프>/
#   (호스트 ~/project/gr00t_Isaacsim/out/eval_office_scan/... 로 보임)
#   씬 로그의 "[office-scan] effective: {...}" 한 줄이 그 런의
#   프로토콜/자산/배치 기록이다
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
SCRIPT_DIR="${SCRIPT_DIR:-/root/project/isaac_franka/envs/office_scan}"
OUT_DIR="${OUT_DIR:-/root/project/out/eval_office_scan}"
VIDEO="${VIDEO:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # 1회성 실험용 추가 플래그 (헤더 참조)

# 공용 서버 규약: GPU 는 반드시 명시 (조용한 기본값 금지. DRY_RUN 도
# 컨테이너에서 Isaac 을 기동하므로 GPU 를 문다)
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
