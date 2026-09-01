#!/usr/bin/env bash
# ======================================
# File: run_eval_bimanual.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_eval_bimanual.sh 2026-08-31-r2
# r2: 원샷 통합 -- server 자동 기동/종료 흡수 (구 r1 = client 전용, server
#     수동 기동 필요했음). mode 3종: 기본 / DRY_RUN / EXTERNAL_SERVER.
#     로그 prefix [eval-bimanual] 로 분화 (구 [eval] 은 office 와 충돌)
# =============================================================================
# 양팔 Franka 큐브 탑쌓기 원샷 평가:
# server 자동 기동 -> port 대기 -> eval -> server 자동 종료 -> summary 출력
#
# 사용 (host, gr00t_sh 활성화 상태, clone root 기준):
#   SERVER_GPU=2 CLIENT_GPU=1 CKPT=/data1/huggingface/sslunder54/checkpoints/bimanual_full \
#     nohup bash Isaac-franka/envs/bimanual/run_eval_bimanual.sh \
#     > out/eval_bimanual.log 2>&1 &
#   본평가는 EPISODES_PER_N=50 을 명시할 것 (기본 10 = 빠른 확인용)
#
# mode:
#   기본               server 기동 + EXIT trap 정리 (CKPT/SERVER_GPU 필수)
#   DRY_RUN=1          server/CKPT 없이 관측 규약만 확인 (client --dry-run)
#   EXTERNAL_SERVER=1  이미 떠 있는 server 재사용 (기동/정리 없음, CKPT 불필요.
#                      sweep 등 server 1회 load 로 여러 run 돌릴 때)
#
# 산출물: container /root/project/out/eval_bimanual/<KST stamp>/
#   (host <클론루트>/out/eval_bimanual/... 로 보임)
#
# 중단: pkill -f run_eval_bimanual.sh
#   (EXIT trap 이 server 와 container client 까지 정리함. trap 이 못 돈
#    경우의 수동 정리: pkill -f "inference_service.py.*--port <PORT>" +
#    docker exec -u 0 -e TERM=xterm gr00t_dt bash -lc
#      "pkill -9 -f '[e]val_bimanual.py'")
# =============================================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRANKA_DIR="$(cd "$HERE/../.." && pwd)"
# repo 자기상대: 이 clone root = container 의 /root/project mount
PROJ_ROOT="$(cd "$FRANKA_DIR/.." && pwd)"

# ---- knob ----
SERVER_GPU="${SERVER_GPU:-}"
CLIENT_GPU="${CLIENT_GPU:-}"
CKPT="${CKPT:-}"
PORT="${PORT:-5555}"
EPISODES_PER_N="${EPISODES_PER_N:-10}"
SEED="${SEED:-1000}"
CHUNK="${CHUNK:-16}"
NUM_CUBES="${NUM_CUBES:-2,5}"
STEPS_PER_CUBE="${STEPS_PER_CUBE:-250}"
VIDEO="${VIDEO:-0}"
DENOISE="${DENOISE:-4}"
CONTAINER="${CONTAINER:-gr00t_dt}"
ISAACLAB="${ISAACLAB:-/root/project/IsaacLab}"
SCRIPT_DIR="${SCRIPT_DIR:-/root/project/Isaac-franka/envs/bimanual}"
OUT_DIR="${OUT_DIR:-/root/project/out/eval_bimanual}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # 1회성 실험용 추가 flag (client 에 그대로 전달)
DRY_RUN="${DRY_RUN:-0}"
EXTERNAL_SERVER="${EXTERNAL_SERVER:-0}"

GR00T_DIR="${GR00T_DIR:-$PROJ_ROOT/Isaac-GR00T}"
OUT_ROOT="${OUT_ROOT:-$PROJ_ROOT/out}"
# 결과 위치 (container 경로). host 에서는 /root/project -> clone root 로 보임
HOST_OUT="${OUT_DIR/#\/root\/project/$PROJ_ROOT}"

# ---- gate ----
if [ "$DRY_RUN" = "1" ] && [ "$EXTERNAL_SERVER" = "1" ]; then
    echo "[eval-bimanual] ERROR: DRY_RUN and EXTERNAL_SERVER are exclusive"
    exit 1
fi
# 공용 server 규약: GPU 반드시 명시 (조용한 기본값 금지. DRY_RUN 도
# container 에서 Isaac 을 기동하므로 GPU 점유 발생)
if [ -z "$CLIENT_GPU" ]; then
    echo "[eval-bimanual] ERROR: set CLIENT_GPU explicitly (shared server), e.g. CLIENT_GPU=2 ..."
    exit 1
fi
if [ "$DRY_RUN" = "0" ] && [ "$EXTERNAL_SERVER" = "0" ]; then
    if [ -z "$SERVER_GPU" ]; then
        echo "[eval-bimanual] ERROR: set SERVER_GPU explicitly (shared server)"
        exit 1
    fi
    # CKPT 도 GPU 처럼 명시 필수 (조용한 기본값 = 엉뚱한 ckpt 평가 함정)
    if [ -z "$CKPT" ]; then
        echo "[eval-bimanual] ERROR: set CKPT explicitly, e.g."
        echo "  CKPT=/data1/huggingface/sslunder54/checkpoints/bimanual_full"
        exit 1
    fi
    if [ ! -d "$CKPT" ]; then
        echo "[eval-bimanual] ERROR: checkpoint not found: $CKPT"
        exit 1
    fi
    if [ "${CONDA_DEFAULT_ENV:-}" != "gr00t_sh" ]; then
        echo "[eval-bimanual] WARNING: conda env is '${CONDA_DEFAULT_ENV:-none}' (server needs gr00t_sh)"
    fi
fi

# ---- client 실행 flag ----
EXTRA=""
if [ "$DRY_RUN" = "1" ]; then
    EXTRA="--dry-run"
else
    EXTRA="--out-dir $OUT_DIR"
    [ "$VIDEO" = "1" ] && EXTRA="$EXTRA --video-dir $OUT_DIR/videos"
fi

# client 는 container 안에서 실행 (host 는 server 만)
run_client() {
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
  $EXTRA_ARGS \
  $EXTRA
"
}

echo "[eval-bimanual] gpu=$CLIENT_GPU episodes_per_n=$EPISODES_PER_N port=$PORT seed=$SEED"
echo "[eval-bimanual] cubes=$NUM_CUBES steps_per_cube=$STEPS_PER_CUBE"

# ---- mode: DRY_RUN (server 없이 관측 규약만) ----
if [ "$DRY_RUN" = "1" ]; then
    run_client
    exit $?
fi

STAMP=$(date +%m%d_%H%M%S)
mkdir -p "$OUT_ROOT"
CLIENT_LOG="$OUT_ROOT/eval_bimanual_${STAMP}_client.log"

# ---- mode: EXTERNAL_SERVER (떠 있는 server 재사용) ----
if [ "$EXTERNAL_SERVER" = "1" ]; then
    if ! nc -z localhost "$PORT" 2>/dev/null; then
        echo "[eval-bimanual] ERROR: no server on port $PORT (EXTERNAL_SERVER=1)"
        exit 1
    fi
    echo "[eval-bimanual] using external server on port $PORT"
    # 중단 시 container 안 client zombie 정리 (정상 종료 후에는 no-op)
    trap 'docker exec -u 0 -e TERM=xterm "$CONTAINER" bash -lc "pkill -9 -f \"[e]val_bimanual.py\"" 2>/dev/null || true' EXIT
else
    # ---- mode: 기본 (server 자동 기동/종료) ----
    # port 선점 검사: 이미 물려 있으면 남의 server 일 수 있으니 죽이지 않고 거부
    if nc -z localhost "$PORT" 2>/dev/null; then
        echo "[eval-bimanual] ERROR: port $PORT already in use"
        echo "  reuse it with EXTERNAL_SERVER=1, or check: pgrep -af \"inference_service.py.*--port $PORT\""
        exit 1
    fi
    SERVER_LOG="$OUT_ROOT/eval_bimanual_${STAMP}_server.log"
    echo "[eval-bimanual] ckpt=$CKPT"
    echo "[eval-bimanual] server gpu=$SERVER_GPU stamp=$STAMP"

    # server 기동 (background; pipe 없이 redirect 라 $! = runner bash).
    # GR00T_DIR 명시 전달: server script 의 기본값에 기대지 않음
    SERVER_GPU="$SERVER_GPU" PORT="$PORT" CKPT="$CKPT" DENOISE="$DENOISE" \
        GR00T_DIR="$GR00T_DIR" \
        bash "$FRANKA_DIR/train/run_server_finetuned.sh" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    cleanup() {
        # eval 이 어떻게 끝나든 (정상/error/Ctrl+C) server 는 반드시 내림.
        # port 조준 pkill 이라 다른 port 의 server 는 건드리지 않음.
        echo "[eval-bimanual] stopping server (pid $SERVER_PID)..."
        pkill -TERM -P "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 3
        pkill -9 -f "inference_service.py.*--port $PORT" 2>/dev/null || true
        kill -9 "$SERVER_PID" 2>/dev/null || true
        # 중단 시 container 안 client 가 zombie 로 남는 것도 같이 정리
        # (정상 종료 후에는 match 가 없어 no-op)
        docker exec -u 0 -e TERM=xterm "$CONTAINER" bash -lc \
            "pkill -9 -f '[e]val_bimanual.py'" 2>/dev/null || true
        echo "[eval-bimanual] cleanup done"
    }
    trap cleanup EXIT

    # server 대기: port 가 열리거나, server 가 죽거나, timeout
    echo "[eval-bimanual] waiting for server on port $PORT (ckpt load takes ~minutes)..."
    waited=0
    while ! nc -z localhost "$PORT" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[eval-bimanual] ERROR: server died during startup -- see $SERVER_LOG"
            exit 1
        fi
        sleep 3
        waited=$((waited + 3))
        if [ "$waited" -ge 600 ]; then
            echo "[eval-bimanual] ERROR: server not up after ${waited}s -- see $SERVER_LOG"
            exit 1
        fi
    done
    echo "[eval-bimanual] server up after ~${waited}s"
fi

# ---- eval (foreground) ----
run_client 2>&1 | tee "$CLIENT_LOG"
rc=${PIPESTATUS[0]}

echo "[eval-bimanual] eval finished rc=$rc"
LATEST=$(ls -dt "$HOST_OUT"/*/ 2>/dev/null | head -1)
if [ -n "$LATEST" ] && [ -f "$LATEST/summary.json" ]; then
    echo "[eval-bimanual] summary: $LATEST/summary.json"
    cat "$LATEST/summary.json"
    echo ""
fi
exit "$rc"
