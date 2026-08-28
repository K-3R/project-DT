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
                    env1/2/4 공용, CKPT 로 환경 선택
convert/            HDF5 -> LeRobot 변환 (run_convert_bimanual.sh /
                    run_convert_office.sh -- office_scan 데이터도 office
                    변환기를 그대로 쓴다)
kill.sh             실행 중단 (컨테이너 안 프로세스는 안에서 죽여야 한다)
ASSETS.md           스캔 자산 대장 (take -> ply -> usd -> 스케일 계보)
```

### 컨테이너 경로

`/root/project` 는 호스트의 **이 레포 클론 루트**의 바인드 마운트다
(각자 자기 컨테이너를 만들어 자기 클론을 마운트한다 -- setting.sh 참조).
호스트에서 파일을 고치면 즉시 반영된다 (`docker cp` 불필요).

| | 호스트 | 컨테이너 |
| --- | --- | --- |
| 프로젝트 루트 | `~/project/project-DT` (클론 위치) | `/root/project` |
| 이 폴더 | `.../isaac_franka` | `/root/project/isaac_franka` |
| Isaac Lab | `.../IsaacLab` | `/root/project/IsaacLab` |
| 생성 데이터 | `.../datasets` (실행 시 자동 생성) | `/root/project/datasets` |

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
| 컨테이너 안 프로세스가 호스트 kill 로 안 죽음 | docker exec 프로세스는 안에서 `docker exec -u 0 <컨테이너> pkill -f <이름>`. 컨테이너 자체는 건드리지 말 것 |
