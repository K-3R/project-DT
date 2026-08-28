#!/usr/bin/env bash
# [ver] run_evalauto_office_scan.sh 2026-08-28-r1
# =============================================================================
# office_scan 원샷 평가: 서버 자동 기동 -> eval -> 서버 자동 종료
# (run_scripts/run_robocasa_baseline.sh 의 자동화 방식을 이 트랙에 이식)
#
# 사용 (호스트, gr00t_sh 활성화 상태에서):
#   SERVER_GPU=5 CLIENT_GPU=4 \
#   CKPT=/data1/huggingface/sslunder54/checkpoints/office_scan_3view \
#     nohup bash run_evalauto_office_scan.sh \
#     > ~/project/gr00t_Isaacsim/out/evalauto_office_scan.log 2>&1 &
#
# 노브: EPISODES_PER_N(기본 50 = 본평가), PORT(5561), VIDEO(0), DENOISE(4),
#       OUT_DIR(컨테이너 경로; 기본 /root/project/out/eval_office_scan --
#       런마다 그 아래 KST 스탬프 폴더가 생기니 구분용으로만 바꾸면 된다)
#       SEED/NUM_MARKERS/EXTRA_ARGS 등 run_eval_office_scan.sh 의 노브는
#       환경변수로 그대로 상속된다 (이 스크립트는 통로만 제공)
# CKPT 는 필수 -- run_server_finetuned.sh 의 기본값(bimanual_full)이
# 조용히 뜨는 함정을 원샷 러너에서는 원천 차단한다.
#
# 중단: pkill -f run_evalauto_office_scan.sh
#   (EXIT trap 이 서버와 컨테이너 클라이언트까지 정리한다. trap 이 못 돈
#    경우의 수동 정리: pkill -f "inference_service.py.*--port <PORT>" +
#    docker exec -u 0 -e TERM=xterm gr00t_isaac bash -lc
#      "pkill -9 -f '[e]val_office_scan.py'")
# =============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRANKA_DIR="$(cd "$HERE/../.." && pwd)"

SERVER_GPU="${SERVER_GPU:-}"
CLIENT_GPU="${CLIENT_GPU:-}"
CKPT="${CKPT:-}"
PORT="${PORT:-5561}"
EPISODES_PER_N="${EPISODES_PER_N:-50}"
VIDEO="${VIDEO:-0}"
DENOISE="${DENOISE:-4}"
CONTAINER="${CONTAINER:-gr00t_isaac}"
OUT_ROOT="$HOME/project/gr00t_Isaacsim/out"
# 결과 위치 (컨테이너 경로 -- 기저 러너에 그대로 전달). 호스트에서는
# /root/project -> ~/project/gr00t_Isaacsim 매핑으로 보인다
OUT_DIR="${OUT_DIR:-/root/project/out/eval_office_scan}"
HOST_OUT="${OUT_DIR/#\/root\/project/$HOME/project/gr00t_Isaacsim}"

# 공용 서버 규약: GPU 반드시 명시. CKPT 도 명시 필수 (헤더 참조)
if [ -z "$SERVER_GPU" ] || [ -z "$CLIENT_GPU" ]; then
    echo "[evalauto] ERROR: set SERVER_GPU and CLIENT_GPU explicitly (shared server)"
    exit 1
fi
if [ -z "$CKPT" ]; then
    echo "[evalauto] ERROR: set CKPT explicitly, e.g."
    echo "  CKPT=/data1/huggingface/sslunder54/checkpoints/office_scan_3view"
    exit 1
fi
if [ ! -d "$CKPT" ]; then
    echo "[evalauto] ERROR: checkpoint not found: $CKPT"
    exit 1
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "gr00t_sh" ]; then
    echo "[evalauto] WARNING: conda env is '${CONDA_DEFAULT_ENV:-none}' (server needs gr00t_sh)"
fi

# 포트 선점 검사: 이미 물려 있으면 남의 서버일 수 있으니 죽이지 않고 거부
if nc -z localhost "$PORT" 2>/dev/null; then
    echo "[evalauto] ERROR: port $PORT already in use"
    echo "  check: pgrep -af \"inference_service.py.*--port $PORT\""
    exit 1
fi

STAMP=$(date +%m%d_%H%M%S)
mkdir -p "$OUT_ROOT"
SERVER_LOG="$OUT_ROOT/evalauto_${STAMP}_server.log"
CLIENT_LOG="$OUT_ROOT/evalauto_${STAMP}_client.log"

echo "[evalauto] ckpt=$CKPT"
echo "[evalauto] port=$PORT episodes_per_n=$EPISODES_PER_N video=$VIDEO"
echo "[evalauto] server gpu=$SERVER_GPU client gpu=$CLIENT_GPU stamp=$STAMP"

# ---- 서버 기동 (백그라운드; 파이프 없이 리다이렉트라 $! = 러너 bash) ----
SERVER_GPU="$SERVER_GPU" PORT="$PORT" CKPT="$CKPT" DENOISE="$DENOISE" \
    bash "$FRANKA_DIR/train/run_server_finetuned.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    # eval 이 어떻게 끝나든 (정상/에러/Ctrl+C) 서버는 반드시 내린다.
    # 포트 조준 pkill 이라 다른 포트의 서버는 건드리지 않는다.
    echo "[evalauto] stopping server (pid $SERVER_PID)..."
    pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 3
    pkill -9 -f "inference_service.py.*--port $PORT" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
    # 중단 시 컨테이너 안 클라이언트가 좀비로 남는 것도 같이 정리
    # (정상 종료 후에는 매칭이 없어 no-op)
    docker exec -u 0 -e TERM=xterm "$CONTAINER" bash -lc \
        "pkill -9 -f '[e]val_office_scan.py'" 2>/dev/null || true
    echo "[evalauto] cleanup done"
}
trap cleanup EXIT

# ---- 서버 대기: 포트가 열리거나, 서버가 죽거나, 타임아웃 ----
echo "[evalauto] waiting for server on port $PORT (ckpt load takes ~minutes)..."
waited=0
while ! nc -z localhost "$PORT" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[evalauto] ERROR: server died during startup -- see $SERVER_LOG"
        exit 1
    fi
    sleep 3
    waited=$((waited + 3))
    if [ "$waited" -ge 600 ]; then
        echo "[evalauto] ERROR: server not up after ${waited}s -- see $SERVER_LOG"
        exit 1
    fi
done
echo "[evalauto] server up after ~${waited}s"

# ---- eval (포그라운드; 기존 러너 재사용, 나머지 노브는 env 상속) ----
CLIENT_GPU="$CLIENT_GPU" EPISODES_PER_N="$EPISODES_PER_N" PORT="$PORT" VIDEO="$VIDEO" \
    OUT_DIR="$OUT_DIR" \
    bash "$HERE/run_eval_office_scan.sh" 2>&1 | tee "$CLIENT_LOG"
rc=${PIPESTATUS[0]}

echo "[evalauto] eval finished rc=$rc"
LATEST=$(ls -dt "$HOST_OUT"/*/ 2>/dev/null | head -1)
if [ -n "$LATEST" ] && [ -f "$LATEST/summary.json" ]; then
    echo "[evalauto] summary: $LATEST/summary.json"
    cat "$LATEST/summary.json"
    echo ""
fi
exit "$rc"
