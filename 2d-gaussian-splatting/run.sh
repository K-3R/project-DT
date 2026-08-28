# stage 1: frame 추출
conda activate surfel_splatting && \
GPU=2 STAGE=1 UNTIL=1 nohup bash scan/run_scan.sh <파일경로>.mp4 > <로그경로>.log 2>&1 &

# stage 2: frame matching + pose 역산 (COLMAP)
conda activate surfel_splatting && \
GPU=2 STAGE=2 UNTIL=2 nohup bash scan/run_scan.sh <파일경로>.mp4 > <로그경로>.log 2>&1 &

# stage 3: 학습
conda activate surfel_splatting && \
GPU=2 STAGE=3 UNTIL=3 nohup bash scan/run_scan.sh <파일경로>.mp4 > <로그경로>.log 2>&1 &

# stage 4: mesh 추출 (TSDF, unbounded 기본 -> fuse_unbounded_post.ply)
conda activate surfel_splatting && \
GPU=2 STAGE=4 UNTIL=4 nohup bash scan/run_scan.sh <파일경로>.mp4 > <로그경로>.log 2>&1 &

# stage 1-4: 한 번에 실행 (STAGE/UNTIL 기본값 = 1..4)
conda activate surfel_splatting && \
GPU=2 nohup bash scan/run_scan.sh <파일경로>.mp4 > <로그경로>.log 2>&1 &

# stage 5: mesh 후처리 -- RANSAC 자동 정렬 첫 시도 (report/preview 확인용;
# CPU 만 사용, 수 분. plane 목록 보고 --plane-idx/--flip 교정 후 재실행)
conda activate surfel_splatting && \
python scan/postprocess_mesh.py \
    --ply output/<input_mesh_path> \
    --out data/<output_mesh_path>

## stage 5.1: 일정 거리 이내 mesh 재추출 (bounded TSDF -- DEPTH_TRUNC [m] 안쪽만)
conda activate surfel_splatting && \
GPU=2 MESH=bounded STAGE=4 DEPTH_TRUNC=15 MESH_RES=1536 SDF_TRUNC=0.1 \
    nohup bash scan/run_scan.sh <파일경로>.mp4 > <로그경로>.log 2>&1 &

## stage 5.2: 특정 영역만 잘라낸 mesh (meshlab pick 4점 정렬 + crop -> 자산 PLY)
conda activate surfel_splatting && \
python scan/postprocess_mesh.py \
    --ply output/<input_mesh_path> \
    --out data/<output_mesh_path> \
    --pick-plane \
    --scale <실측/pick거리> \
    --pick "x,y,z;x,y,z;x,y,z;x,y,z" \
    --pick-box 1.2 # 해당 박스 영역의 1.2배로 자르는 세팅