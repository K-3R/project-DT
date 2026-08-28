# scan -- 연구실 스캔 재구성 글루 (flow [1][2] 단계)

폰 영상 -> COLMAP -> 2DGS 학습 -> TSDF 메시 -> 정렬된 자산 PLY 까지의
우리 스크립트 모음. **이 디렉토리가 upstream(2d-gaussian-splatting)과
우리 코드의 경계다** (예외: 상위의 convert.py 는 수정된 upstream --
패치 목록은 그 파일 헤더 [ver] 참조).

노브/함정의 정본은 각 스크립트 헤더다. 전체 flow 에서의 위치는
레포 루트 README 참조.

## 설치 (1회, conda env surfel_splatting)

```bash
bash scan/setup/install_2dgs.sh
```

(CUDA_HOME 11.8 고정 + 서브모듈 CUDA 빌드 + yml 부족분 보충까지 한 번에.
함정과 순서는 스크립트 헤더 참조.)

## quickstart (2d-gaussian-splatting/ 에서, surfel_splatting env)

```bash
# 영상 -> 메시 원샷 (data/raw/ 에 mp4 를 올린 뒤)
GPU=5 nohup bash scan/run_scan.sh data/raw/take7.mp4 > data/take7_run.log 2>&1 &

# 중단은 항상 세트로: 러너 먼저, 자식 따로 (고아 방지; ; 라서 앞이
# 이미 죽어 있어도 뒤가 실행된다)
pkill -f run_scan.sh; pkill -f "colmap.*take7"

# 자산 제작용 bounded 메시 재추출 (기본 unbounded 런의 학습을 재사용;
# 산출 = fuse_post.ply. 기본 unbounded 산출물은 fuse_unbounded_post.ply)
GPU=5 MESH=bounded STAGE=4 DEPTH_TRUNC=15 MESH_RES=1536 SDF_TRUNC=0.1 \
    bash scan/run_scan.sh take7.mp4

# 후처리 (meshlab 픽 4점 -> 정렬/스케일/크롭 -> 자산 PLY)
python scan/postprocess_mesh.py --ply output/take7/train/ours_30000/fuse_post.ply \
    --out ../Isaac-franka/envs/office_scan/assets/take7_desk.ply \
    --pick-plane --scale <실측/픽거리> --pick "..." --pick-box 1.2

```

다음 단계 (Isaac 쪽): `Isaac-franka/envs/replica/replica_to_usd.py` 로 USD
변환 -> `Isaac-franka/envs/office_scan/` 벤치마크에서 소비 (자산은
`envs/office_scan/assets/` 에 동거). 자산 계보는 `Isaac-franka/ASSETS.md`.

새 스캔으로 교체할 때는 scan-scale 재캘리브레이션이 필수다 (COLMAP
좌표계는 재구성마다 임의) -- 픽 절차는 postprocess_mesh.py 헤더의
레시피 2 참조.

## 파일

```
run_scan.sh          통합 러너 (STAGE/UNTIL 재개, 노브는 헤더 참조)
extract_frames.py    영상 -> 프레임 (fps 자동, 블러 리포트, --merge/--gray)
pick_cores.py        공용 서버 예의: 한가한 CPU 코어 자동 선택
postprocess_mesh.py  자산 제작 정본 (RANSAC/픽 정렬, 크롭, 사이드카 json)
setup/               설치 스크립트 (install_2dgs.sh)
```
