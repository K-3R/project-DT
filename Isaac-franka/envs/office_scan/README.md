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
office_scan_scene.py   scene 모듈: [A] office_scene 엔진 사본(무수정) +
                          [B] scan 오버라이드(책상 시각만 scan USD 교체).
                          프로토콜 확정 상수의 정본 ([B] 블록)
preview_office_scan.py        미리보기 (preview_office 사본, scene 만 교체)
eval_office_scan.py           폐루프 평가 (eval_office 사본, scene 만 교체)
gen_office_scan.py            데모 생성 (gen_office 사본, scene 만 교체)
run_evalauto_office_scan.sh   원샷 평가 runner (server 자동 기동/종료)
run_eval_office_scan.sh       평가 runner (host 에서 실행)
run_gen_office_scan.sh        생성 runner (Track B: office 와 동일 seed/batch,
                          산출은 datasets/office_scan_markers 로 분리)
assets/                       scan 자산 (take6_desk_hq.usd + .ply + json)
```

동작 원리: env 격리 -- office 의 scene/entry 를 물리 사본으로 들여와
scan 델타만 오버라이드함 (구판은 sys.modules 오버레이였으나 08-31 격리).
task/로봇/카메라/성공 판정은 env2 와 완전히 동일함. office 원본이
갱신되면 사본([A] 절 + entry 3종) 재복사가 필요함.

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

평가 (호스트에서. 서버 먼저, office 체크포인트 그대로):

```bash
SERVER_GPU=<GPU> PORT=5561 CKPT=/data1/huggingface/sslunder54/checkpoints/office_3view \
  bash ~/project/gr00t_Isaacsim/isaac_franka/train/run_server_finetuned.sh
```

```bash
DRY_RUN=1 bash run_eval_office_scan.sh                      # 관측 규약 사전점검
CLIENT_GPU=<GPU> EPISODES_PER_N=10 VIDEO=1 PORT=5561 bash run_eval_office_scan.sh
```

산출물: `~/project/gr00t_Isaacsim/out/eval_office_scan/<KST스탬프>/`
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

- office 의 cfg 부품 이름 규약(`desk_top_*`, `stand_*_main` 등)에
  의존한다. 이름이 갈리면 build 가 경고를 찍는다
- entry/scene 은 office 의 격리 사본이라 office 쪽 수정이 자동 반영되지
  않는다 -- 원본 갱신 시 사본 재복사 (각 파일 [ver] 의 계보 참조)
- Track B 절차: `run_gen_office_scan.sh` (수율 검증은 DEMOS_OVERRIDE=3 먼저)
  -> convert 의 office 변환기를 NAS office_scan_markers 로 돌리고
  -> train/run_finetune.sh 파인튜닝 -> 새 ckpt 로 `run_eval_office_scan.sh` 재평가
