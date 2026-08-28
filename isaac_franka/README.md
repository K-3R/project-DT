# isaac_franka -- Isaac Lab Franka x GR00T 벤치마크 모음

Isaac Lab 위에 양팔 Franka 환경 4개를 만들고 GR00T 로 데이터 생성/학습/평가
하는 트랙의 인덱스. 각 환경은 하위 디렉토리로 격리되어 있고, 상세는 각자의
README / 정본 문서를 따른다.

## 환경 매트릭스

| env | 디렉토리 | 씬 | 태스크 | 정본 문서 |
| --- | --- | --- | --- | --- |
| 1 | `envs/bimanual/` | 테이블 + 양팔 리그 | 큐브 탑쌓기 | `docs/bimanual_training_pipeline.md` |
| 2 | `envs/office/` | 프로시저럴 사무실 책상 (실측 치수) | 마커 -> 연필꽂이 | `docs/office_env_pipeline.md` |
| 3 | `envs/replica/` | Replica 스캔 방 (배경만) | (미리보기 단계) | `docs/replica_background.md` |
| 4 | `envs/office_scan/` | ★실사 스캔 책상 (env2 씬 스왑) | env2 와 동일 | `docs/gsrecon_pipeline.md` + `envs/office_scan/README.md` |

env4 는 env2 의 태스크/로봇/카메라를 그대로 쓰고 배경만 실사 스캔이라,
env2 대비 성공률 차이 = **배경 도메인 갭** 측정이 된다.

격리 원칙: `envs/office/` 는 무수정 엔진, `envs/office_scan/` 이
sys.modules 바꿔치기로 얹힌다. `envs/replica/` 는 office 를 import 하지
않는 완전 격리.

러너 명명 규칙: `run_{동사}_{환경}.sh` (gen/eval/convert x bimanual/office/
office_scan). "무표기 러너 = 특정 env" 관례는 폐지했다.

## 디렉토리 구성

```
envs/               환경 4개 + groot_client.py (env 스크립트들이 상위
                    디렉토리에서 import 하는 공유 ZMQ 클라이언트)
train/              파인튜닝 + 추론 서버. run_server_finetuned.sh 는
                    env1/2/4 공용, CKPT 로 환경 선택 (ex run_server_bimanual)
convert/            HDF5 -> LeRobot 변환 (run_convert_bimanual.sh /
                    run_convert_office.sh -- office_scan 데이터도 office
                    변환기를 그대로 쓴다)
legacy_libero/      최초의 단일팔 LIBERO 하네스 (동결 -- 아래 legacy 절)
kill.sh             실행 중단 (컨테이너 안 프로세스는 안에서 죽여야 한다)
ASSETS.md           스캔 자산 대장 (take -> ply -> usd -> 스케일 계보)
```

### 컨테이너 경로 (컨테이너 `gr00t_isaac`)

`/root/project` 는 호스트 `~/project/gr00t_Isaacsim` 의 **바인드 마운트**다.
호스트에서 파일을 고치면 즉시 반영된다 (`docker cp` 불필요).

| | 호스트 | 컨테이너 |
| --- | --- | --- |
| 프로젝트 루트 | `~/project/gr00t_Isaacsim` | `/root/project` |
| 이 폴더 | `.../isaac_franka` | `/root/project/isaac_franka` |
| Isaac Lab | `.../IsaacLab` | `/root/project/IsaacLab` |
| 자산 | `.../datasets` | `/root/project/datasets` |

> `python.sh -c "import isaaclab"` 이 `ModuleNotFoundError: omni.physics` 로
> 실패하는 것은 정상이다 -- SimulationApp(AppLauncher) 이후에만 로드된다.

> 셸 스크립트를 Windows 에서 편집했다면 개행이 CRLF 일 수 있다.
> 프로젝트 루트의 `crlf2lf.sh` 로 변환.

### 실행 함정 (전 환경 공통, 전부 실측)

| 증상 | 원인 / 처방 |
| --- | --- |
| `'ansi+tabs': unknown terminal type` 로 즉시 종료 | `docker exec` 에 TTY 가 없어 로그인 셸 초기화가 죽는다 -> `-e TERM=xterm` |
| 우리 `print` 가 하나도 안 보임 | stdout 블록 버퍼링 -> `-e PYTHONUNBUFFERED=1` (Kit 로그는 stderr 라 계속 흐름) |
| 산출물을 호스트에서 못 지움 | `-u 0` 이라 root 소유 -> 명령 앞에 `umask 000 &&`. 기존 파일은 `docker exec -u 0 ... chmod -R 777` |
| `[done]` 후 프로세스가 안 끝남 | `simulation_app.close()` 가 헤드리스에서 반환 안 함 -> 스크립트가 `os._exit(0)` 로 강제 종료 |
| `OgnSdOnNewFrame: frames discarded` | 카메라 캡처가 렌더보다 빠름. 영상 프레임만 누락, 수치엔 영향 없음 |
| USD `FileNotFoundError` | `ISAAC_NUCLEUS_DIR` = `/isaac-assets/Isaac` 로컬 미러. 테이블은 `Props/Mounts/` 아래 |
| 서버 망에서 S3 다운로드가 TLS 에러 | `omniverse-content-production.s3...` 가 막혀 있다 -> 로컬에서 받아 복사 |
| 컨테이너 안 프로세스가 호스트 kill 로 안 죽음 | docker exec 프로세스는 안에서 `docker exec -u 0 gr00t_isaac pkill -f <이름>`. 컨테이너 자체는 건드리지 말 것 |

---

## legacy: LIBERO 단일팔 폐루프 하네스 (legacy_libero/)

최초 작업물. Isaac Lab 의 Franka 단일팔 태스크를 공개 LIBERO 체크포인트로
돌리는 배관 검증용 (`isaac_franka_gr00t.py`, `run_server.sh`, `run_eval.sh`).
현재 트랙(양팔 env1~4)과 별개이며 동결 상태로 유지만 한다.

```bash
cd legacy_libero
DRY_RUN=1 CLIENT_GPU=3 ./run_eval.sh          # 관측 규약 확인 (서버 불필요)
SERVER_GPU=4 CKPT=libero-spatial ./run_server.sh
CLIENT_GPU=3 EPISODES=5 ./run_eval.sh
```

체크포인트: `youliangtan/gr00t-n1.5-libero-{spatial,goal,object,90,long}-posttrain`
(전부 단일팔, `embodiment_tag=new_embodiment`).

규약 요점: 액션 = eef delta 6 + gripper 1 (IK-Rel scale 0.5 별도),
쿼터니언 (w,x,y,z) -> axis-angle 변환 + 이중피복 제거, 카메라 256 강제.
캘리브레이션 노브: `GAIN`(안 움직이면 올림/튀면 내림), `GRIP_SIGN`(-1 로
뒤집기), `--action-chunk`(기본 8), `--warmup`(안정화 무동작).

기대치: 배관 검증이지 성능이 아니다 -- 다른 심 + 다른 태스크 zero-shot 이라
성공률 0 근처가 정상 (ROBOGATE 보고: LIBERO 97.65% -> Isaac 0%).
팔이 물체 쪽으로 그럴듯하게 움직이면 배관 성공.
