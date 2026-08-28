# scan -- 연구실 촬영 재구성 glue (flow [1][2] 단계)

폰 영상 -> COLMAP -> 2DGS 학습 -> TSDF mesh -> 정렬된 자산 PLY 까지의
우리 script 모음. **이 디렉토리 = upstream(2d-gaussian-splatting)과
우리 코드의 경계** (예외: 상위 convert.py 와 utils/render_utils.py 는
수정된 upstream -- 패치 목록은 각 파일 헤더 [ver] 참조).

knob/함정 정본 = 각 script 헤더. 전체 flow 에서의 위치는 repo 루트
README 참조.

## 설치 (1회, conda env surfel_splatting)

```bash
bash scan/setup/install_2dgs.sh
```

(CUDA_HOME 11.8 고정 + submodule CUDA build + yml 부족분 보충까지 한 번에.
함정과 순서는 script 헤더 참조.)

## quickstart (2d-gaussian-splatting/ 에서, surfel_splatting env)

```bash
# 영상 -> mesh 원샷 (data/raw/ 에 mp4 를 올린 뒤)
GPU=5 nohup bash scan/run_scan.sh data/raw/take7.mp4 > data/take7_run.log 2>&1 &

# 중단은 항상 세트로: runner 먼저, 자식(colmap/train) 따로 (고아 방지;
# ; 라서 앞이 이미 죽어 있어도 뒤가 실행됨)
pkill -f run_scan.sh; pkill -f "colmap.*take7"; pkill -f "python.*take7"

# 자산 제작용 bounded mesh 재추출 (기본 unbounded run 의 학습을 재사용;
# 산출 = fuse_post.ply. 기본 unbounded 산출물은 fuse_unbounded_post.ply)
GPU=5 MESH=bounded STAGE=4 DEPTH_TRUNC=15 MESH_RES=1536 SDF_TRUNC=0.1 \
    bash scan/run_scan.sh take7.mp4

# 후처리 (meshlab pick 4점 -> 정렬/scale/crop -> 자산 PLY)
python scan/postprocess_mesh.py --ply output/take7/train/ours_30000/fuse_post.ply \
    --out ../Isaac-franka/envs/office_scan/assets/take7_desk.ply \
    --pick-plane --scale <실측/pick거리> --pick "..." --pick-box 1.2

```

다음 단계 (Isaac 쪽): `Isaac-franka/envs/replica/replica_to_usd.py` 로 USD
변환 -> `Isaac-franka/envs/office_scan/` 벤치마크에서 소비 (자산은
`envs/office_scan/assets/` 동거). 자산 계보 = `Isaac-franka/ASSETS.md`.

새 scan 교체 시 scan-scale 재보정 필수 (COLMAP 좌표계는 재구성마다 임의)
-- pick 절차는 postprocess_mesh.py 헤더 recipe 2 참조.

## 파일

```
run_scan.sh          통합 runner (STAGE/UNTIL 재개, knob 은 헤더 참조)
extract_frames.py    영상 -> frame (fps 자동, blur 경고, --merge/--gray)
pick_cores.py        공용 서버 배려: 한가한 CPU core 자동 선택
postprocess_mesh.py  자산 제작 정본 (RANSAC/pick 정렬, crop, sidecar json)
setup/               설치 script (install_2dgs.sh)
```
