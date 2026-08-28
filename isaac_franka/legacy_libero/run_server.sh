#!/usr/bin/env bash
# [ver] run_server.sh 2026-08-26-r2
# r2: GPU 기본값 제거 (공용 서버 규약 -- 명시 필수)
#
# GR00T 추론 서버. 체크포인트를 로컬 경로에서 읽는다.
#
#   SERVER_GPU=4 CKPT=libero-spatial ./run_server.sh
#
# CKPT 는 CKPT_ROOT 아래의 디렉터리 이름이거나 절대경로.
#   libero-spatial / libero-goal / libero-object / libero-90 / libero-long
#   n1.5-3b / robocasa-tabletop  (기존 보유분)
set -euo pipefail

SERVER_GPU="${SERVER_GPU:-}"
if [ -z "$SERVER_GPU" ]; then
  echo "[fatal] set SERVER_GPU explicitly (shared server), e.g. SERVER_GPU=4 ..."
  exit 1
fi
CKPT="${CKPT:-libero-spatial}"
PORT="${PORT:-5555}"
DENOISE="${DENOISE:-8}"
GR00T_DIR="${GR00T_DIR:-$HOME/Isaac-GR00T}"
CKPT_ROOT="${CKPT_ROOT:-/home/sslunder54/project/gr00t_Isaacsim/checkpoints}"

case "$CKPT" in
  /*) CKPT_PATH="$CKPT" ;;
  *)  CKPT_PATH="$CKPT_ROOT/$CKPT" ;;
esac

# goal 만 데이터 컨피그가 다르다 (MeanStd 정규화로 학습됨)
case "$CKPT" in
  *libero-goal*) DCFG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd" ;;
  *libero*)      DCFG="examples.Libero.custom_data_config:LiberoDataConfig" ;;
  *)             DCFG="${DCFG:-examples.Libero.custom_data_config:LiberoDataConfig}" ;;
esac

if [ ! -d "$CKPT_PATH" ]; then
  echo "[fatal] checkpoint not found: $CKPT_PATH"
  echo "  to download:"
  echo "  huggingface-cli download youliangtan/gr00t-n1.5-libero-spatial-posttrain \\"
  echo "      --local-dir $CKPT_ROOT/libero-spatial"
  exit 1
fi

echo "[server] gpu=$SERVER_GPU port=$PORT denoise=$DENOISE"
echo "[server] ckpt=$CKPT_PATH"
echo "[server] dcfg=$DCFG"

cd "$GR00T_DIR"
CUDA_VISIBLE_DEVICES="$SERVER_GPU" python scripts/inference_service.py \
  --model_path "$CKPT_PATH" \
  --server \
  --data_config "$DCFG" \
  --denoising-steps "$DENOISE" \
  --port "$PORT" \
  --embodiment-tag new_embodiment
