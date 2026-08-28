#!/usr/bin/env bash
# ======================================
# File: install_2dgs.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-28
# ======================================
# [ver] install_2dgs.sh 2026-08-28-r2
# =============================================================================
# 2DGS(surfel_splatting) conda env 구축 -- 모노레포 기준.
# upstream 사본이 이 레포의 2d-gaussian-splatting/ 에 동봉돼 있으므로
# 별도 clone 이 필요 없다.
#
# 사용 (2d-gaussian-splatting/ 에서):
#   bash scan/setup/install_2dgs.sh
#
# 함정 (전부 실측 -- 순서를 바꾸지 말 것):
#   - PATH 의 nvcc(12.x)와 env 의 torch(cu118)가 어긋나면 서브모듈 CUDA
#     커널 빌드가 깨진다 -> CUDA_HOME 을 11.8 로 고정하고 같은 셸에서 빌드
#   - environment.yml 은 의존성이 부족하다 (pip 절 중단 이력) -> [4] 보충
# =============================================================================
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 2d-gaussian-splatting/
cd "$HERE"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"

# ---- [0] 선검사: CUDA 11.8 부재는 느린 env 생성 전에 걸러낸다 ----
if [ ! -d "$CUDA_HOME" ]; then
    echo "[install] ERROR: CUDA_HOME not found: $CUDA_HOME"
    echo "  set CUDA_HOME=<cuda-11.8 install path> and rerun"
    exit 1
fi

# ---- [1] conda env (이미 있으면 건너뜀 -- 재실행 안전) ----
eval "$(conda shell.bash hook)"
# 정확 일치 검사: 부분 문자열 매칭이면 유사명 env 가 생성을 건너뛰게 한다
if ! conda env list | awk '{print $1}' | grep -qx "surfel_splatting"; then
    echo "[install] creating conda env surfel_splatting..."
    conda env create --file environment.yml
fi
conda activate surfel_splatting

# ---- [2] CUDA 11.8 고정 (torch cu118 과 일치해야 커널 빌드 성립) ----
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[install] nvcc: $(nvcc --version | tail -n 1)"

# ---- [3] CUDA 서브모듈 빌드 (수 분 소요) ----
MAX_JOBS=8 pip install submodules/diff-surfel-rasterization
MAX_JOBS=8 pip install submodules/simple-knn

# ---- [4] environment.yml 부족분 보충 (없으면 후처리/렌더에서 즉사) ----
pip install open3d==0.18.0 mediapy lpips scikit-image tqdm trimesh plyfile \
    opencv-python matplotlib

# ---- [5] 검증 ----
python -c "import torch, diff_surfel_rasterization, simple_knn, open3d; print('build OK | torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"
echo "[install] done -- next: GPU=<n> bash scan/run_scan.sh data/raw/<video>.mp4"
