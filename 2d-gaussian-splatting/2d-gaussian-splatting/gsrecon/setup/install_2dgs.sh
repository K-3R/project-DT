# 2DGS 설치
cd ~/project
git clone https://github.com/hbb1/2d-gaussian-splatting.git 2d-gaussian-splatting --recursive

# 2DGS env 세팅
cd 2d-gaussian-splatting
conda env create --file environment.yml

## cuda 11.8 <-> PATH 의 nvcc 가 12.1
## 따라서 추가로 cuda 11.8 path에서 다운로드 할 수 있도록
conda activate surfel_splatting && \
export CUDA_HOME=/usr/local/cuda-11.8 && \
export PATH=$CUDA_HOME/bin:$PATH && \
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
nvcc --version | tail -n 2

## 같은 창에서
cd ~/project/2d-gaussian-splatting && \
MAX_JOBS=8 pip install submodules/diff-surfel-rasterization && \
MAX_JOBS=8 pip install submodules/simple-knn
## cuda 커널 컴파일

# 확인
python -c "import torch, diff_surfel_rasterization, simple_knn; print('build OK | torch', torch.__version__, '| cuda', torch.version.cuda, '| avail', torch.cuda.is_available())"