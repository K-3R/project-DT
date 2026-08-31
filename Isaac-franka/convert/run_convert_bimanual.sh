#!/usr/bin/env bash
# ======================================
# File: run_convert_bimanual.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_convert_bimanual.sh 2026-08-31
# =============================================================================
# 씨앗 HDF5 4벌 -> LeRobot dataset 1개 변환 (host gr00t 환경에서 실행)
#
# 입력: NAS 의 batch 폴더들 (<IN>/*/seed_3view.hdf5, n2/n3/n4/n5)
# 출력: home 의 LeRobot dataset 폴더 (학습이 읽는 위치, 총 ~200MB 수준)
#
# 실행:
#   bash run_convert_bimanual.sh
# 완료 후 검증 gate (GR00T repo root 에서):
#   python scripts/load_dataset.py --dataset-path <OUT> \
#       --embodiment-tag new_embodiment --plot-state-action
# =============================================================================
set -u
# tee 등 pipe 에 물려도 log 가 실시간으로 나오게 buffering 해제
export PYTHONUNBUFFERED=1

# 3-view (2026-08-11): 입력은 각 batch 의 seed_3view.hdf5, 출력은 _3view dataset
# (1-view dataset franka_bimanual_lerobot 은 비교용으로 보존)
IN="${IN:-/data1/huggingface/sslunder54/datasets/franka_bimanual}"
OUT="${OUT:-/data1/huggingface/sslunder54/datasets/franka_bimanual_lerobot_3view}"

# 의존성 확인 (없으면 pip install 안내 후 중단)
if ! python -c "import h5py, pandas, pyarrow, imageio, imageio_ffmpeg" 2>/dev/null; then
    echo "[conv] ERROR: missing python deps in this env"
    echo "[conv] run: pip install h5py pandas pyarrow imageio imageio-ffmpeg"
    exit 1
fi

python "$(dirname "$0")/convert_bimanual_lerobot.py" \
    --input-root "$IN" \
    --output-dir "$OUT"
