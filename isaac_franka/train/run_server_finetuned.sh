#!/usr/bin/env bash
# =============================================================================
# 양팔 Franka 파인튜닝 체크포인트용 GR00T 추론 서버 (호스트 gr00t 환경)
#
#   SERVER_GPU=0 bash run_server_finetuned.sh
#
# 주의: data-config / embodiment-tag 는 학습과 동일해야 한다 (정규화 통계가
#       체크포인트 metadata 에 태그 키로 저장돼 있음). our_configs.py 가
#       레포 루트에 있어야 하고, 실행 위치도 레포 루트여야 한다.
# 중단: Ctrl+C (포그라운드) 또는 pkill -f inference_service
# =============================================================================
set -euo pipefail

SERVER_GPU="${SERVER_GPU:-}"
# 공용 서버 규약: GPU 는 반드시 명시 (조용한 기본값 금지)
if [ -z "$SERVER_GPU" ]; then
    echo "[server] ERROR: set SERVER_GPU explicitly (shared server), e.g. SERVER_GPU=2 ..."
    exit 1
fi
PORT="${PORT:-5555}"
DENOISE="${DENOISE:-4}"
GR00T_DIR="${GR00T_DIR:-$HOME/project/gr00t_Isaacsim/Isaac-GR00T}"
CKPT="${CKPT:-/data1/huggingface/sslunder54/checkpoints/bimanual_full}"
DCFG="${DCFG:-our_configs:BimanualFrankaConfig}"

if [ ! -d "$CKPT" ]; then
    echo "[server] ERROR: checkpoint not found: $CKPT"
    exit 1
fi
if [ ! -f "$GR00T_DIR/our_configs.py" ]; then
    echo "[server] ERROR: our_configs.py missing in $GR00T_DIR"
    exit 1
fi

echo "[server] gpu=$SERVER_GPU port=$PORT denoise=$DENOISE"
echo "[server] ckpt=$CKPT"
echo "[server] dcfg=$DCFG"

cd "$GR00T_DIR"
CUDA_VISIBLE_DEVICES="$SERVER_GPU" python scripts/inference_service.py \
    --model_path "$CKPT" \
    --server \
    --data_config "$DCFG" \
    --denoising-steps "$DENOISE" \
    --port "$PORT" \
    --embodiment-tag new_embodiment
