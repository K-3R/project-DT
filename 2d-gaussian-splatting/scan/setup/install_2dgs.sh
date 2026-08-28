#!/usr/bin/env bash
# ======================================
# File: install_2dgs.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-28
# ======================================
# [ver] install_2dgs.sh 2026-08-28-r2
# =============================================================================
# 2DGS(surfel_splatting) conda env 구축. upstream 사본이 repo 에 동봉 --
# 별도 clone 불필요.
#
# 사용 (2d-gaussian-splatting/ 에서): bash scan/setup/install_2dgs.sh
#
# 순서 변경 금지: PATH 의 nvcc(12.x)와 torch(cu118)가 어긋나면 submodule
# build 가 깨짐 -> CUDA_HOME 11.8 고정 후 같은 shell 에서 build
# =============================================================================
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # 2d-gaussian-splatting/
cd "$HERE"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"

# ---- [0] 선검사: CUDA 11.8 ----
if [ ! -d "$CUDA_HOME" ]; then
    echo "[install] ERROR: CUDA_HOME not found: $CUDA_HOME"
    echo "  set CUDA_HOME=<cuda-11.8 install path> and rerun"
    exit 1
fi

# ---- [1] conda env (있으면 건너뜀) ----
eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "surfel_splatting"; then
    echo "[install] creating conda env surfel_splatting..."
    conda env create --file environment.yml
fi
conda activate surfel_splatting

# ---- [2] CUDA 11.8 고정 ----
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[install] nvcc: $(nvcc --version | tail -n 1)"

# ---- [3] CUDA submodule build (수 분 소요) ----
MAX_JOBS=8 pip install submodules/diff-surfel-rasterization
MAX_JOBS=8 pip install submodules/simple-knn

# ---- [4] environment.yml 부족분 보충 ----
pip install open3d==0.18.0 mediapy lpips scikit-image tqdm trimesh plyfile \
    opencv-python matplotlib

# ---- [5] 검증 ----
python -c "import torch, diff_surfel_rasterization, simple_knn, open3d; print('build OK | torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"
echo "[install] done -- next: GPU=<n> bash scan/run_scan.sh data/raw/<video>.mp4"
