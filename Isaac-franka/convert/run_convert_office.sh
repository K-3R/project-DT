#!/usr/bin/env bash
# ======================================
# File: run_convert_office.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_convert_office.sh 2026-08-31
# =============================================================================
# office marker HDF5 -> LeRobot dataset 변환 (host gr00t 환경에서 실행)
#
# 입력: NAS 의 office batch 폴더들 (<IN>/*/seed.hdf5, 3-view 직접 기록본)
# 출력: <OUT> LeRobot dataset (지시문 = marker 개수별)
#
# 실행:
#   bash run_convert_office.sh
# =============================================================================
set -u
export PYTHONUNBUFFERED=1

IN="${IN:-/data1/huggingface/sslunder54/datasets/office_markers}"
OUT="${OUT:-/data1/huggingface/sslunder54/datasets/office_markers_lerobot}"

if ! python -c "import h5py, pandas, pyarrow, imageio, imageio_ffmpeg" 2>/dev/null; then
    echo "[conv] ERROR: missing python deps in this env"
    echo "[conv] run: conda activate gr00t_sh  (or pip install h5py pandas pyarrow imageio imageio-ffmpeg)"
    exit 1
fi

python "$(dirname "$0")/convert_office_lerobot.py" \
    --input-root "$IN" \
    --output-dir "$OUT"
