#!/usr/bin/env bash
# =============================================================================
# 씨앗 HDF5 4벌 -> LeRobot 데이터셋 1개 변환 (호스트 gr00t 환경에서 실행)
#
# 입력: NAS 의 배치 폴더들 (<IN>/*/seed.hdf5, n2/n3/n4/n5)
# 출력: 홈의 LeRobot 데이터셋 폴더 (학습이 읽는 위치, 총 ~200MB 수준)
#
# 실행:
#   bash run_convert_bimanual.sh
# 완료 후 검증 게이트 (GR00T 레포 루트에서):
#   python scripts/load_dataset.py --dataset-path <OUT> \
#       --embodiment-tag new_embodiment --plot-state-action
# =============================================================================
set -u
# tee 등 파이프에 물려도 로그가 실시간으로 나오게 버퍼링 해제
export PYTHONUNBUFFERED=1

# 3뷰 (2026-08-11): 입력은 각 배치의 seed_3view.hdf5, 출력은 _3view 데이터셋
# (1뷰 데이터셋 franka_bimanual_lerobot 은 비교용으로 보존)
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
