#!/usr/bin/env bash
# ======================================
# File: run_evalauto_office_scan.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_evalauto_office_scan.sh 2026-08-28-r2
# r2: 경로 자기상대화 (OUT_ROOT/HOST_OUT = 클론 루트 기준) + 서버에
#     GR00T_DIR 명시 전달 (옛 레포 기본값 발동 차단) + ckpt 예시 현행화
# =============================================================================
# office_scan one-shot 평가: server 자동 기동 -> eval -> server 자동 종료
#
# 사용 (host, gr00t_sh 활성화 상태에서, clone root 기준):
#   SERVER_GPU=5 CLIENT_GPU=4 CKPT=<클론>/checkpoints/lab_office_sim \
#     nohup bash Isaac-franka/envs/office_scan/run_evalauto_office_scan.sh \
#     > out/evalauto_office_scan.log 2>&1 &
#
# knob: EPISODES_PER_N(기본 50 = 본평가), PORT(5561), VIDEO(0), DENOISE(4),
#       OUT_DIR(container 경로; 기본 /root/project/out/eval_office_scan --
#       run 마다 그 아래 KST stamp 폴더가 생기니 구분용으로만 바꾸면 됨)
#       SEED/NUM_MARKERS/EXTRA_ARGS 등 run_eval_office_scan.sh 의 knob 은
#       환경변수로 그대로 상속됨 (이 script 는 통로만 제공)
# CKPT 는 필수 (run_server_finetuned.sh 도 08-31 부터 기본값 없이 명시 필수)
#
# 중단: pkill -f run_evalauto_office_scan.sh
#   (EXIT trap 이 server 와 container client 까지 정리함. trap 이 못 돈
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
# repo 자기상대: 이 clone root = container 의 /root/project mount
PROJ_ROOT="$(cd "$FRANKA_DIR/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/out}"
GR00T_DIR="${GR00T_DIR:-$PROJ_ROOT/Isaac-GR00T}"
# 결과 위치 (container 경로 -- 기저 runner 에 그대로 전달). host 에서는
# /root/project -> clone root mapping 으로 보임
OUT_DIR="${OUT_DIR:-/root/project/out/eval_office_scan}"
HOST_OUT="${OUT_DIR/#\/root\/project/$PROJ_ROOT}"

# 공용 server 규약: GPU 반드시 명시. CKPT 도 명시 필수 (header 참조)
if [ -z "$SERVER_GPU" ] || [ -z "$CLIENT_GPU" ]; then
    echo "[evalauto] ERROR: set SERVER_GPU and CLIENT_GPU explicitly (shared server)"
    exit 1
fi
if [ -z "$CKPT" ]; then
    echo "[evalauto] ERROR: set CKPT explicitly, e.g."
    echo "  CKPT=$PROJ_ROOT/checkpoints/lab_office_sim"
    exit 1
fi
if [ ! -d "$CKPT" ]; then
    echo "[evalauto] ERROR: checkpoint not found: $CKPT"
    exit 1
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "gr00t_sh" ]; then
    echo "[evalauto] WARNING: conda env is '${CONDA_DEFAULT_ENV:-none}' (server needs gr00t_sh)"
fi

# port 선점 검사: 이미 물려 있으면 남의 server 일 수 있으니 죽이지 않고 거부
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

# ---- server 기동 (background; pipe 없이 redirect 라 $! = runner bash) ----
# GR00T_DIR 명시 전달: server script 의 기본값에 기대지 않음 (repo 이원화 방지)
SERVER_GPU="$SERVER_GPU" PORT="$PORT" CKPT="$CKPT" DENOISE="$DENOISE" \
    GR00T_DIR="$GR00T_DIR" \
    bash "$FRANKA_DIR/train/run_server_finetuned.sh" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
    # eval 이 어떻게 끝나든 (정상/error/Ctrl+C) server 는 반드시 내림.
    # port 조준 pkill 이라 다른 port 의 server 는 건드리지 않음.
    echo "[evalauto] stopping server (pid $SERVER_PID)..."
    pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 3
    pkill -9 -f "inference_service.py.*--port $PORT" 2>/dev/null || true
    kill -9 "$SERVER_PID" 2>/dev/null || true
    # 중단 시 container 안 client 가 zombie 로 남는 것도 같이 정리
    # (정상 종료 후에는 match 가 없어 no-op)
    docker exec -u 0 -e TERM=xterm "$CONTAINER" bash -lc \
        "pkill -9 -f '[e]val_office_scan.py'" 2>/dev/null || true
    echo "[evalauto] cleanup done"
}
trap cleanup EXIT

# ---- server 대기: port 가 열리거나, server 가 죽거나, timeout ----
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

# ---- eval (foreground; 기존 runner 재사용, 나머지 knob 은 env 상속) ----
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
