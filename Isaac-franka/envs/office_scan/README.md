# office-scan -- 실사 스캔 책상 벤치마크 (4번째 환경)

office(env2) 마커 태스크를 그대로 두고 **배경만 실제 연구실 책상 스캔**으로
바꾼 환경이다. 정책이 프로시저럴 배경(office)으로 학습됐다면, 여기서의
성공률 하락 = **배경 도메인 갭**의 직접 측정치가 된다.

```
학습(office 배경) ──▶ 평가(office 배경)   : env2 기준선
학습(office 배경) ──▶ 평가(스캔 배경)      : ★이 환경 (Track A, 제로샷 전이)
학습(스캔 배경)   ──▶ 평가(스캔 배경)      : Track B (데이터 생성 후 -- 예정)
```

정본 파이프라인 문서: `gr00t_isaacsim/docs/gsrecon_pipeline.md`
(폰 촬영 -> 2DGS -> 메시 -> 후처리 -> USD 자산 제작 과정은 그쪽 참조.
자산 제작 글루는 2dgs 레포의 `postprocess_mesh.py` 등)

## 파일 구성

```
office_scan_scene.py   scene 모듈: env4 세계 정의 전부 (office 엔진에
                          scan 델타를 인라인한 단일 정의).
                          protocol 확정 상수의 정본 (상단 블록)
preview_office_scan.py        미리보기 (preview_office 사본, scene 만 교체)
eval_office_scan.py           폐루프 평가 (eval_office 사본, scene 만 교체)
gen_office_scan.py            데모 생성 (gen_office 사본, scene 만 교체)
run_eval_office_scan.sh       원샷 평가 runner (server 자동 기동/종료.
                          DRY_RUN / EXTERNAL_SERVER mode 는 헤더 참조)
run_gen_office_scan.sh        생성 runner (Track B: office 와 동일 seed/batch,
                          산출은 datasets/office_scan_markers 로 분리)
assets/                       scan 자산 (take6_desk_hq.usd + .ply + json)
```

동작 원리: env 격리 -- scene/entry 전부 자기 파일로 실행됨.
scene 은 office 엔진에 scan 델타를 인라인한 한 벌의 정의이고, env2 대비
델타 목록은 scene 헤더 docstring 에 있음. task/로봇/카메라/성공 판정은
env2 와 완전히 동일함.
(구판 계보: sys.modules 오버레이 -> 08-31 사본 격리 -> 08-31 인라인)

## 프로토콜 v1 확정값

정본 = `office_scan_scene.py` 상단 상수 블록. 러너는 이 값들을
전달하지 않는다. **값을 바꾸면 `PROTOCOL` 문자열을 올릴 것** (v1 -> v2).

| 항목 | 값 | 근거 |
|---|---|---|
| 자산 | `take6_desk_hq.usd` | take6 책상 중심 돔 스캔, bounded TSDF |
| scan-scale | 1.73 | 자산 스케일(칸막이 장축=1.4m 가정) -> 실물 벤치 2.4m |
| 배치 | 자동 (칸막이 밑선 -> 책상 뒤 가장자리) | yaw 부호 추종 |
| desk-color | 0.878,0.878,0.871 (sRGB) | 스캔 상판 클린패치 중앙값 |
| 조명 | dome 800 / key 2000 | 스캔 색에 촬영 조명이 구워져 있어 감쇠 |
| holder-xy | 0.16,-0.36 | 구운 소품 간섭 회피 (+6cm 사람 쪽) |
| region | 0.12,0.26,-0.12,0.38 | 구운 키보드 회피 + 홀더 금지박스 정합 |

색 규약: 스캔 버텍스컬러는 sRGB -> linear 변환을 거쳐 USD 에 들어간다
(replica_to_usd `--color-gamma srgb`). 프로시저럴 면을 같은 톤으로
맞추기 위해 --desk-color 도 같은 변환을 태운다.

씬 로그의 `[office-scan] effective: {...}` JSON 한 줄이 그 런의
프로토콜/자산/배치 기록이다 (산출물 추적용).

## 실행

미리보기 (배경만 / 로봇·태스크 배치 확인):

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_isaac bash -lc \
  "umask 000 && cd /root/project/Isaac-franka/envs/office_scan && \
   CUDA_VISIBLE_DEVICES=<GPU> /root/project/IsaacLab/isaaclab.sh -p \
   preview_office_scan.py --headless --robots 1 --holder 1 --items 4 \
   --out /root/project/out/office_scan_preview"
```

평가 (호스트에서. 원샷 -- server 자동 기동/종료):

```bash
DRY_RUN=1 CLIENT_GPU=<GPU> bash run_eval_office_scan.sh    # 관측 규약 사전점검
SERVER_GPU=<GPU> CLIENT_GPU=<GPU> EPISODES_PER_N=50 \
  CKPT=<클론루트>/checkpoints/lab_office_sim bash run_eval_office_scan.sh
```

산출물: `<클론루트>/out/eval_office_scan/<KST스탬프>/`
(summary.json + 에피소드 비디오). env2 의 `out/eval_office/` 결과와
나란히 놓는 것이 보고 포맷이다.

## 자산 계보

```
폰 영상 take6 -> 2d-gaussian-splatting: scan/run_scan.sh (COLMAP+학습+bounded 메시)
  -> postprocess_mesh.py --pick-plane ... (정렬/스케일/크롭; 사이드카 json)
  -> replica_to_usd.py --up z --floor-pct -1 --no-recenter (sRGB->linear)
  -> envs/office_scan/assets/take6_desk_hq.usd  (env 자산 동거; + .json
     사이드카가 있으면 씬이 로그로 출력)
```

새 스캔(take7...)으로 교체할 때는 **scan-scale 재캘리브레이션이 필수**다
(COLMAP 좌표계는 재구성마다 임의) -- 픽 절차는 정본 문서 5절 참조.

## 알려진 제약

- scene/entry 는 office 와 독립이라 office 쪽 수정이 자동 반영되지
  않는다 -- 의도적으로 가져올 때만 반영 (각 파일 [ver] 의 계보 참조)
- Track B 절차: `run_gen_office_scan.sh` (수율 검증은 DEMOS_OVERRIDE=3 먼저)
  -> convert 의 office 변환기를 NAS office_scan_markers 로 돌리고
  -> train/run_finetune.sh 파인튜닝 -> 새 ckpt 로 `run_eval_office_scan.sh` 재평가
