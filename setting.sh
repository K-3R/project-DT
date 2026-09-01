# 환경 구축 명령 모음 -- 자세한 설명과 함정은 SETUP.md 참조
# host tool 전제: COLMAP, MeshLab, netcat, docker(NVIDIA runtime), CUDA 11.8

# step 1: clone (여기가 container 의 /root/project 로 mount 됨)
git clone <this-repo> project-DT && cd project-DT

# step 2: host env surfel_splatting (2DGS -- conda env + submodule build)
cd 2d-gaussian-splatting && bash scan/setup/install_2dgs.sh && cd ..

# step 3: host env gr00t_sh (system 전제: ffmpeg libsm6 libxext6)
conda create -n gr00t_sh python=3.10 -y && conda activate gr00t_sh
pip install --upgrade setuptools
pip install -e ./Isaac-GR00T[base]
pip install --no-build-isolation flash-attn==2.7.1.post4    # CUDA 11.8 이면 2.8.2
pip install h5py pandas pyarrow imageio imageio-ffmpeg
python -c "import gr00t, os; print(os.path.dirname(os.path.dirname(gr00t.__file__)))"

## step 3.1: repo 를 옮겼거나 다시 clone 했을 때만 (editable 재바인딩)
pip install -e ./Isaac-GR00T --no-deps

# step 4: container 생성
docker run -d --name gr00t_dt \
  --network host --gpus all --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v <clone-root>:/root/project \
  -v <cache-dir>/ov-cache:/root/.cache/ov \
  -v <cache-dir>/ov-local:/root/.local/share/ov \
  -v <cache-dir>/omniverse:/root/.nvidia-omniverse \
  --entrypoint /bin/bash \
  nvcr.io/nvidia/isaac-sim:4.5.0 -c "sleep infinity"

# step 5: container 안 Isaac Lab 설치 (a~e 순서 지킬 것)
docker exec gr00t_dt bash -lc "apt-get update -qq && apt-get install -y -qq git"
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install "setuptools<81"
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install \
    --no-build-isolation flatdict==4.0.1
docker exec -e TERM=xterm gr00t_dt bash -lc \
    "ln -sfn /isaac-sim /root/project/IsaacLab/_isaac_sim && \
     cd /root/project/IsaacLab && ./isaaclab.sh -i"
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install pyzmq

## step 5.1: 설치 확인 (isaaclab version / +cu12x torch / True)
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -c \
    "import isaaclab, torch; print(isaaclab.__version__, torch.__version__, torch.cuda.is_available())"

## step 5.2: 작동하는 container 를 image 로 동결 (재구축 반복 방지)
docker commit gr00t_dt gr00t_dt_img:stable
