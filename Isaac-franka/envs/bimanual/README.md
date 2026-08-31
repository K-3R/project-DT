# bimanual - 양팔 Franka 벤치마크

Isaac Lab 위에 Franka 2대로 만든 양팔 워크셀과, 그 위에서 도는 태스크의
정답(씨앗) 궤적 생성기.

```
bimanual_scene.py         씬 정의       - 로봇 2대 ,  테이블 ,  큐브
gen_bimanual.py   궤적 생성기   - 태스크 상태기계 + 기록
PRESETS.md                확정 프리셋   - 배치 수치와 그 근거
archive/                  역할이 끝난 것 - 배치 결정에 쓴 검증 스크립트
```

두 파일이면 된다. 씬은 태스크와 독립이라, 다른 태스크를 만들 때
`bimanual_scene.py` 는 그대로 두고 생성기만 새로 쓰면 된다.

---

## 실행

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_isaac bash -lc \
"umask 000 && cd /root/project/Isaac-franka/envs/bimanual && \
CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p gen_bimanual.py \
  --headless --demos 20 --table dual \
  --out /root/project/datasets/franka_bimanual/seed.hdf5 \
  --video-dir /root/project/out/sm"
```

산출물은 실행 시각으로 폴더가 갈려 덮이지 않는다.

```
datasets/franka_bimanual/0805_143012/seed.hdf5      성공한 궤적만
                                     run_meta.json  그때 쓴 노브 전부
out/sm/0805_143012/ep000_n4_success.mp4
```

빠르게 확인만 할 때는 `--demos 1 --randomize 0` 에 영상을 끄면 1~2분이면 끝난다.

---

## 태스크

```
cube_1        탑의 바닥. 그대로 둔다
나머지 N-1개   베이스가 가까운 팔이 집어 순서대로 그 위에 쌓는다
N             --num-cubes 범위에서 에피소드마다 무작위 (기본 2~6)
```

집기는 두 팔이 **동시에** 한다 - 담당 큐브가 서로 반대편이라 동선이 안 겹친다.
쌓는 자리는 하나뿐이라 놓기만 한 번에 한 팔씩, `Stack` 이 lock 으로 조율한다.

성공 판정: 모든 큐브의 xy 가 3.5cm 이내로 모이고 z 가 2cm 이상씩 층을 이룰 것.
성공한 에피소드만 HDF5 에 저장된다.

---

## 상태 흐름 (한 큐브당)

```
REST -> ABOVE_PICK -> AT_PICK -> CLOSE -> LIFT -> HOLD
     -> ABOVE_PLACE -> AT_PLACE -> OPEN -> UP -> RETREAT -> (다음 큐브 | FINISHED)
```

| 상태 | 하는 일 | 왜 필요한가 |
| --- | --- | --- |
| `HOLD` | 쥔 채 자기 쪽 바깥에서 대기 | 두 팔이 가운데서 손목이 부딪힌다 |
| `UP` | 놓은 직후 **수직으로만** 이탈 | 바로 옆으로 가면 방금 쌓은 탑을 스친다 |
| `RETREAT` | 자기 쪽 바깥으로 물러나며 lock 해제 | 탑 바로 위에서 멈추면 다음 팔의 진입로를 막는다 |

---

## 설계에서 중요한 네 가지

**(1) 툴 오프셋** - IK 가 제어하는 것은 `panda_hand` 원점이고 실제 파지점(TCP)은
거기서 손 +z 로 `--tool-offset`(0.1034m) 나간 지점이다. 목표를 그만큼 뒤로
물리지 않으면 손이 테이블에 파고든다.

**(2) 파지 지점 고정** - 파지 목표는 `REST` 에서 한 번만 계산한다. 매 스텝
큐브를 다시 읽으면, 쥔 뒤에는 큐브가 손을 따라오므로 목표가 손과 함께 도망가
`LIFT` 에서 영원히 도달하지 못한다.

**(3) 실측 기반 높이** - 놓는 높이를 "바닥 + 층수x4cm" 로 가정하지 않는다.
파지 순간 `TCP - 큐브중심` 간격을 재고(`hold_off`), 놓을 때는 `Stack.top_z()` 로
현재 탑의 실제 높이를 읽는다. 층이 쌓이며 생기는 오차가 누적되지 않는다.

**(4) 명령 없는 구간 금지** - `DifferentialIKController` 는 `set_command` 없이
`compute` 하면 목표가 0 으로 잡혀 팔이 루트 원점으로 끌려간다. 대기 구간에서는
반드시 `hold_here()` 로 현재 자세를 목표로 잡아야 한다.

---

## 주요 노브

| 노브 | 기본 | 증상 -> 조치 |
| --- | --- | --- |
| `--num-cubes` | `2,6` | 개수 범위 |
| `--region` | `0.36,0.60,-0.26,0.26` | 큐브를 뿌릴 구역. 샘플링 실패가 잦으면 넓힌다 |
| `--min-sep` / `--base-clear` | `0.11` / `0.13` | 큐브 간 / 탑과의 최소 거리. 옆 큐브를 건드리면 키운다 |
| `--yaw-range` | `45` | 큐브 회전 폭. 파지 실패가 잦으면 줄인다 |
| `--place-clear` | `0.005` | 탑 꼭대기에서 띄우는 여유. 누르면 키운다 |
| `--pos-tol` / `--dwell` | `0.006` / `25` | 도달 판정. 전이가 안 되면 완화, 불안정하면 강화 |
| `--state-timeout` | `250` | 접촉으로 막혔을 때의 강제 진행. `[!]` 로그로 보인다 |
| `--max-steps` | `1400` | **큐브 1개당** 상한 (실제 상한 = 이 값 x 쌓을 개수) |

씬 쪽 노브는 `PRESETS.md` 참조. 확정값은 `--table dual --mirror-side right
--base-sep 0.9` 이다.

---

## 기록 형식

```
data/demo_i/
  actions   (T,16)  [Lpos3 Lquat4 Lgrip1 | Rpos3 Rquat4 Rgrip1]
                    월드 프레임 절대 TCP 목표 (IK-Abs 계열)
  obs/      joint_pos(T,18)  eef_pos_l/r  eef_quat_l/r  grip_l/r
            cube_pos(T,3N)   cube_quat(T,4N)
  subtask/  place_done (T,)  한 큐브를 쌓아 올릴 때마다 1 증가하는 누적값
```

`subtask/place_done` 은 나중에 Isaac Lab Mimic 의 어노테이션 경계로 쓰기 위한
것이다. 상태기계는 그 시점을 정확히 알고 있으므로 사람이 다시 찍을 필요가 없다.

액션을 **절대 자세(IK-Abs)** 로 기록한 이유는, Isaac Lab 의 공식 상태기계 예제와
GR1T2 Mimic 환경이 모두 Abs 이고, IK-Rel 로 만든 궤적이 Mimic 에서 막힌 보고
([Discussion #4006](https://github.com/isaac-sim/IsaacLab/discussions/4006))가
있기 때문이다.

---

## 실행 함정

상위 폴더 `../../README.md` 의 표 참조. 요약하면:

```
-e TERM=xterm            없으면 로그인 셸 초기화가 죽는다
-e PYTHONUNBUFFERED=1    없으면 우리 print 가 끝날 때까지 안 보인다
umask 000 &&             없으면 산출물이 root 소유라 호스트에서 못 지운다
```
