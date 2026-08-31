# project-DT — 실사 스캔 배경 로봇 벤치마크

연구실 책상을 **폰으로 스캔**해 Isaac Sim 학습 환경으로 만들고,
"**배경만 실사로 바뀌면 VLA 정책 성능이 어떻게 되는가**"를 측정하는
전체 플로우를 담은 레포. 프로시저럴 배경(env2)과 실사 스캔 배경(env4)이
**동일 태스크·동일 시드·동일 expert 궤적**으로 짝을 이루므로, 두 환경의
성공률 차이가 곧 배경 도메인 갭이다.

## 전체 Flow

```
폰 영상 (책상 중심 돔 스캔, 1080p)
  │ [1] 3D 재구성        2d-gaussian-splatting/scan (COLMAP → 2DGS → TSDF 메시)
  ▼
메시 (fuse_post.ply)
  │ [2] 자산화           postprocess_mesh.py (픽 정렬/스케일/크롭) → replica_to_usd.py
  ▼
USD 자산 (Isaac-franka/envs/office_scan/assets/take6_desk_hq.usd)
  │ [3] 씬 스왑          envs/office_scan (env2 의 책상 시각만 교체, 태스크/로봇 동일)
  ▼
env4 벤치마크 씬 (프로토콜 office-scan-v1, 코드 상수가 정본)
  │ [4] 데이터 생성      상태기계 expert, 시드 고정 → 성공 데모 447개 (수율 99%)
  ▼
LeRobot 데이터셋
  │ [5] 파인튜닝         GR00T N1.5-3B, vision/LLM 동결, 32k 스텝
  ▼
체크포인트 (checkpoints/lab_office_sim)
  │ [6] 평가             서버 자동 기동/종료 원샷 러너, N별 50회
  ▼
3행 비교표 (아래 결과)
```

## 결과

| N (마커 수) | env2 기준선 | 제로샷 (env2 ckpt → 스캔 배경) | 인도메인 (스캔 데이터 재학습) |
| --- | --- | --- | --- |
| 1 | 24/50 (48%) | 2/10 (20%) | **37/50 (74%)** |
| 2 | 16/50 (32%) | 0/10 (0%) | 5/10 (50%) |

- 배경만 바꿔도 제로샷 성능이 붕괴한다 (N=2 는 전멸) = **배경 도메인 갭 실재**
- 같은 궤적을 스캔 배경으로 재학습하면 회복을 넘어 **기준선을 초과**한다
  (가설: 실사 배경의 텍스처가 위치 추정의 시각 앵커로 작용)

## 레포 구조

```
2d-gaussian-splatting/   [1][2] 스캔→자산 (호스트, conda surfel_splatting)
                         우리 글루 = scan/, 나머지는 2DGS upstream + 패치 convert.py
Isaac-GR00T/             [5][6] GR00T 1.1.0 + 로컬 패치 + our_configs.py (호스트, conda gr00t_sh)
IsaacLab/                [3][4][6] Isaac Lab 2.2.1 사본 (컨테이너 안에서 사용)
Isaac-franka/            환경 4종(envs/) + 생성/변환/학습/평가 러너 -- 상세는 Isaac-franka/README.md
checkpoints/             (git 제외) 최종 ckpt: lab_office_sim (서버 사본 참조)
datasets/ out/ output/   (git 제외) 실행 시 자동 생성
setting.sh               env/컨테이너 준비 (정비 예정)
```

환경 매트릭스와 실행 함정(TERM/umask/os._exit 등)은
[Isaac-franka/README.md](Isaac-franka/README.md) 참조.

## 시작하기 (요약)

전제: pim-gpu05, conda env 3종(gr00t_sh / surfel_splatting) + 자기 소유
컨테이너(이미지 `nvcr.io/nvidia/isaac-sim:4.5.0`, **자기 클론을
`/root/project` 로 마운트**). 상세 절차는 setting.sh 정비 후 여기로 링크.

```bash
# 스모크 (서버/ckpt 불필요 -- 씬 조립과 자산 경로까지 검증된다)
DRY_RUN=1 CLIENT_GPU=<n> CONTAINER=<자기 컨테이너> \
  bash Isaac-franka/envs/office_scan/run_eval_office_scan.sh
# "[office-scan] effective: {...office-scan-v1...}" 이 뜨면 통과
```

## 단계별 실행 (flow 순서 그대로)

각 러너의 노브/함정은 해당 스크립트 헤더가 정본이다. 모든 GPU 실행은
공용 서버 규약상 GPU 를 명시해야 하며, 미지정 시 에러로 죽는다.

```bash
# [1] 재구성: 영상 → 메시 (2d-gaussian-splatting/ 에서, surfel_splatting env)
GPU=<n> bash scan/run_scan.sh data/raw/<video>.mp4

# [2] 자산화: 정렬/스케일/크롭 → USD (픽 절차는 scan/README.md)
python scan/postprocess_mesh.py --ply <fuse_post.ply> --out <desk.ply> \
    --pick-plane --scale <실측/픽거리> --pick="..." --crop-box ...
# 이후 Isaac-franka/envs/replica/replica_to_usd.py 로 USD 변환 (sRGB→linear)

# [4] 데이터 생성: 시드 고정 배치 (배치당 GPU 1장, 병렬 가능)
BATCHES="m1:100:1,1:220" GPU=<n> bash Isaac-franka/envs/office_scan/run_gen_office_scan.sh
BATCHES="m2:200:2,2:230" GPU=<n> bash Isaac-franka/envs/office_scan/run_gen_office_scan.sh
# → 변환: IN/OUT 지정해 Isaac-franka/convert/run_convert_office.sh

# [5] 파인튜닝: 베이스 n1.5-3b, vision 동결, 32k 스텝
GPU=<n> MODE=full DATASET=<lerobot 데이터셋> OUT=<새 ckpt 경로> \
  bash Isaac-franka/train/run_finetune.sh

# [6] 평가: 서버 자동 기동/종료 원샷, N별 50회 (EPISODES_PER_N=50 = 본평가)
SERVER_GPU=<n> CLIENT_GPU=<n> EPISODES_PER_N=50 CKPT=<클론>/checkpoints/lab_office_sim \
  bash Isaac-franka/envs/office_scan/run_eval_office_scan.sh
```

## 재현 앵커

- 프로토콜 정본 = `Isaac-franka/envs/office_scan/office_scan_scene.py` 상단
  상수 블록 (`office-scan-v1`). 값 변경 = 프로토콜 버전업.
- 데이터 시드 = 배치 정의에 박제 (m1:100 / m2:200) → 동일 데이터 재생성 가능
- 자산 계보 = `Isaac-franka/ASSETS.md` (take → 메시 → 후처리 → USD → 스케일)
- 스택 버전 = isaac-sim 4.5.0 컨테이너 / IsaacLab 2.2.1 / GR00T 1.1.0(+패치)
  / driver 545.23.06 (VERSIONS.md 정비 예정)
