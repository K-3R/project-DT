# 모든 작업은 project-DT에서 실행

# stage 1: frame 추출 (영상은 2d-gaussian-splatting/data/raw/ 에 둠 -- 파일명만 전달)
conda activate surfel_splatting && \
GPU=2 STAGE=1 UNTIL=1 nohup bash 2d-gaussian-splatting/scan/run_scan.sh <영상파일명>.mp4 > <로그경로>.log 2>&1 &

# stage 2: frame matching + pose 역산 (COLMAP)
conda activate surfel_splatting && \
GPU=2 STAGE=2 UNTIL=2 nohup bash 2d-gaussian-splatting/scan/run_scan.sh <영상파일명>.mp4 > <로그경로>.log 2>&1 &

# stage 3: 학습
conda activate surfel_splatting && \
GPU=2 STAGE=3 UNTIL=3 nohup bash 2d-gaussian-splatting/scan/run_scan.sh <영상파일명>.mp4 > <로그경로>.log 2>&1 &

# stage 4: mesh 추출 (TSDF, unbounded 기본 -> fuse_unbounded_post.ply)
conda activate surfel_splatting && \
GPU=2 STAGE=4 UNTIL=4 nohup bash 2d-gaussian-splatting/scan/run_scan.sh <영상파일명>.mp4 > <로그경로>.log 2>&1 &

# stage 1-4: 한 번에 실행 (STAGE/UNTIL 기본값 = 1..4)
conda activate surfel_splatting && \
GPU=2 nohup bash 2d-gaussian-splatting/scan/run_scan.sh <영상파일명>.mp4 > <로그경로>.log 2>&1 &

# stage 5: mesh 후처리 -- RANSAC 자동 정렬 첫 시도 (report/preview 확인용;
# CPU 만 사용, 수 분. plane 목록 보고 --plane-idx/--flip 교정 후 재실행)
conda activate surfel_splatting && \
python 2d-gaussian-splatting/scan/postprocess_mesh.py \
    --ply 2d-gaussian-splatting/output/<input_mesh_path> \
    --out 2d-gaussian-splatting/data/<output_mesh_path>

## stage 5.1: 일정 거리 이내 mesh 재추출 (bounded TSDF -- DEPTH_TRUNC [m] 안쪽만)
conda activate surfel_splatting && \
GPU=2 MESH=bounded STAGE=4 DEPTH_TRUNC=15 MESH_RES=1536 SDF_TRUNC=0.1 \
    nohup bash 2d-gaussian-splatting/scan/run_scan.sh <영상파일명>.mp4 > <로그경로>.log 2>&1 &

## stage 5.2: 특정 영역만 잘라낸 mesh (meshlab pick 4점 정렬 + crop -> 자산 PLY)
conda activate surfel_splatting && \
python 2d-gaussian-splatting/scan/postprocess_mesh.py \
    --ply 2d-gaussian-splatting/output/<input_mesh_path> \
    --out 2d-gaussian-splatting/data/<output_mesh_path> \
    --pick-plane \
    --scale <실측/pick거리> \
    --pick "x,y,z;x,y,z;x,y,z;x,y,z" \
    --pick-box 1.2 # 해당 박스 영역의 1.2배로 자르는 세팅

# stage 6: USD 변환 (컨테이너 안 -- pxr 는 kit 앱 필요; sRGB -> linear 구움)
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_isaac bash -lc "umask 000 && \
    CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p \
    /root/project/Isaac-franka/envs/replica/replica_to_usd.py --headless \
    --up z --floor-pct -1 --no-recenter \
    --ply /root/project/2d-gaussian-splatting/data/<자산>.ply \
    --out /root/project/Isaac-franka/envs/office_scan/assets/<자산>.usd"

# stage 7: scene 미리보기 (자산 교체 후 1회 -- 배경/로봇/태스크 배치 확인)
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_isaac bash -lc \
    "umask 000 && cd /root/project/Isaac-franka/envs/office_scan && \
    CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p \
    preview_office_scan.py --headless --robots 1 --holder 1 --items 4 \
    --out /root/project/out/office_scan_preview"

# stage 8: 데이터 생성 (배치당 GPU 1장 병렬; seed 는 office 와 동일)
BATCHES="m1:100:1,1:220" GPU=2 \
    nohup bash Isaac-franka/envs/office_scan/run_gen_office_scan.sh > <로그경로>.log 2>&1 &
BATCHES="m2:200:2,2:230" GPU=3 \
    nohup bash Isaac-franka/envs/office_scan/run_gen_office_scan.sh > <로그경로>.log 2>&1 &

# stage 9: LeRobot 변환 (호스트 CPU -- 입력/산출 모두 마운트된 NAS)
conda activate gr00t_sh && \
IN=/data1/huggingface/sslunder54/datasets/office_scan_markers \
OUT=/data1/huggingface/sslunder54/datasets/office_scan_markers_lerobot \
    nohup bash Isaac-franka/convert/run_convert_office.sh > <로그경로>.log 2>&1 &

## stage 10: 파인튜닝 (full: batch 10 / 32k 스텝, vision/LLM freeze, 24GB GPU VRAM 필요)
conda activate gr00t_sh && \
GPU=3 MODE=full DATASET=<lerobot경로> OUT=<새ckpt경로> \
    nohup bash Isaac-franka/train/run_finetune.sh > <로그경로>.log 2>&1 &

## stage 11.1: 서버 자동 실행/종료 eval
conda activate gr00t_sh && \
SERVER_GPU=3 CLIENT_GPU=2 PORT=5561 EPISODES_PER_N=50 \
CKPT=checkpoints/lab_office_sim \
    nohup bash Isaac-franka/envs/office_scan/run_eval_office_scan.sh > <로그경로>.log 2>&1 &

## stage 11.2: 떠 있는 서버에서 eval 을 진행
EXTERNAL_SERVER=1 PORT=5561 CLIENT_GPU=2 EPISODES_PER_N=10 \
    bash Isaac-franka/envs/office_scan/run_eval_office_scan.sh
