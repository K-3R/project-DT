#!/usr/bin/env bash
# =============================================================================
# office 마커 HDF5 -> LeRobot 데이터셋 변환 (호스트 gr00t 환경에서 실행)
#
# 입력: NAS 의 office 배치 폴더들 (<IN>/*/seed.hdf5, 3뷰 직접 기록본)
# 출력: <OUT> LeRobot 데이터셋 (지시문 = 마커 개수별)
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
