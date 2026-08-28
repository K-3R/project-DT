# gsrecon -- 연구실 스캔 재구성 글루 (team-sr)

폰 영상 -> COLMAP -> 2DGS 학습 -> TSDF 메시 -> 정렬된 자산 PLY 까지의
우리 스크립트 모음. **이 디렉토리가 upstream(2d-gaussian-splatting)과
우리 코드의 경계다** (예외: 루트의 convert.py 는 수정된 upstream --
패치 목록은 그 파일 헤더 [ver] 참조).

정본 문서: `gr00t_Isaacsim/docs/gsrecon_pipeline.md`
(촬영 프로토콜, 노브 전체, 함정 13건, 후처리 픽 절차, Isaac 접속까지)

## quickstart (레포 루트에서, surfel_splatting env)

```bash
# 영상 -> 메시 원샷 (data/raw/ 에 mp4 를 올린 뒤)
GPU=5 nohup bash gsrecon/run_recon.sh data/raw/take7.mp4 > data/take7_run.log 2>&1 &

# 중단은 항상 세트로: 러너 먼저, 자식 따로 (고아 방지)
pkill -f run_recon.sh && pkill -f "colmap.*take7"

# 후처리 (meshlab 픽 4점 -> 정렬/스케일/크롭; 상세는 정본 5절)
python gsrecon/postprocess_mesh.py --ply output/take7/train/ours_30000/fuse_post.ply \
    --out ~/project/gr00t_Isaacsim/datasets/take7_desk.ply --pick-plane --scale <실측/픽거리> \
    --pick "..." --pick-box 1.2

# 육안 판정 보조 (궤적 depth 비디오 재생성)
python gsrecon/make_depth_video.py --dir output/take7/traj/ours_30000 --range 0.2 2.5 --unit-scale <스케일>
```

다음 단계 (Isaac 쪽): `isaac_franka/envs/replica/replica_to_usd.py` 로 USD
변환 -> `isaac_franka/envs/office_scan/` 벤치마크에서 소비. 자산 계보는
`isaac_franka/ASSETS.md`.

## 파일

```
run_recon.sh         통합 러너 (STAGE/UNTIL 재개, 노브는 헤더 참조)
extract_frames.py    영상 -> 프레임 (fps 자동, 블러 리포트, --merge/--gray)
pick_cores.py        공용 서버 예의: 한가한 CPU 코어 자동 선택
postprocess_mesh.py  자산 제작 정본 (RANSAC/픽 정렬, 크롭, 사이드카 json)
make_depth_video.py  traj depth 비디오 재생성 (정규화 3모드)
setup/               설치 기록 (install_2dgs.sh) + git 부트스트랩
```
