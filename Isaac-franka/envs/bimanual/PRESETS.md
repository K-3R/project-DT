# 양팔 Franka 환경 - 확정 프리셋 (2026-08-05)

Isaac Lab 2.2.1 에는 양팔 로봇 에셋이 없다 (양팔은 GR1T2 휴머노이드뿐).
**Franka 2대를 한 씬에 놓아 양팔 워크셀을 만든다** - robosuite 의 TwoArm 환경이
Panda 2대로 하는 것과 같은 구성.

두 프리셋 A / C 를 환경 기준선으로 확정. 이 위에서 씨앗 궤적을 만들고 태스크를 수행한다.

---

## 프리셋 A - Isaac 테이블 2장

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_isaac bash -lc \
"umask 000 && cd /root/project/isaac_franka/envs/bimanual && \
CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p archive/dual_franka_scene.py \
  --headless --steps 60 \
  --table dual --table-usd SeattleLabTable \
  --mirror-side left --table-mirror 2 \
  --layout parallel --base-sep 0.9 --table-dx 0.55 \
  --shot /root/project/out/A/view.png --diagram /root/project/out/A/layout.png"
```

실측 결과

```
TableL  x[-0.315,+1.104] 1.419m   y[-0.750,+0.910] 1.660m
TableR  x[-0.315,+1.104] 1.419m   y[-0.910,+0.750] 1.660m
합친 면 y[-0.910,+0.910]  중심 0  (대칭, 겹침)
```

## 프리셋 C - 절차적 테이블

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_isaac bash -lc \
"umask 000 && cd /root/project/isaac_franka/envs/bimanual && \
CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p archive/dual_franka_scene.py \
  --headless --steps 60 \
  --table proc --table-size 1.4,1.8,0.05 --table-height 1.05 --table-dx 0.55 \
  --layout parallel --base-sep 0.9 \
  --shot /root/project/out/C/view.png --diagram /root/project/out/C/layout.png"
```

원점이 곧 상판 중심이라 반전, 마운트판 문제가 구조적으로 없다. 크기를 태스크에
맞춰 자유롭게 정할 수 있어 **새 벤치마크의 기본형으로 적합**.

---

## 두 프리셋 공통 (로봇, 물체, 작업공간)

```
로봇        FRANKA_PANDA_HIGH_PD_CFG x2   dof=9, bodies=11, fixed_base=True
            RobotL (0, +0.45, 0)   RobotR (0, -0.45, 0)   둘 다 +x 향함
작업면      env 프레임 z=0,  지면 z=-1.05
큐브        Props/Blocks/{blue,red,green}_block.usd, 4cm
            cube_1 (0.45,  0.00, 0.0203)   중앙  <- 양팔 공용
            cube_2 (0.45, +0.18, 0.0203)   좌
            cube_3 (0.45, -0.18, 0.0203)   우

Franka 리치 0.855 m   베이스 간격 0.900 m   공유 작업공간 폭 0.810 m
  cube_1  L 0.637(O)  R 0.637(O)     <- handover 성립
  cube_2  L 0.525(O)  R 0.774(O)
  cube_3  L 0.774(O)  R 0.525(O)
```

`base_sep` 상한은 2x0.855 = 1.71 m. 그 이상이면 리치 구가 만나지 않아 협응 태스크
자체가 불가능하다. 0.9 는 겹침 0.81 m 로 여유 있다.

---

## 이 에셋의 함정 (A 프리셋에만 해당)

`SeattleLabTable` 의 **원점이 판 중심이 아니라 마운트 쪽에 치우쳐 있다.**
실측: 원점 -> 판 몸통 중심 오프셋 `(-0.156, +0.370)` (rot Rz(90deg) 적용 후 월드 기준).

그래서 두 장을 그냥 나란히 놓으면 판이 한쪽으로 쏠려 **먼 쪽 다리가 홀로 튀어나온다**.

- **z 회전으로는 고칠 수 없다.** 어떤 각도로도 "x 는 유지하고 y 만 반전"이 안 된다
  (Rz(90deg)->(-b,a), Rz(-90deg)->(b,-a), Rz(180deg)->(-a,-b) 어느 것도 원하는 (+0.394,-0.370) 이 아님).
- 유일한 방법이 **거울 반사**: `scale=(1,-1,1)` + `rot` 을 켤레(Rz(theta)->Rz(-theta))로.
  `--table-mirror 2` 가 이것.
- **왼쪽을 반전**해야 두 판이 안쪽으로 모여 겹친다 (`--mirror-side left`).
  오른쪽을 반전하면 바깥으로 벌어져 합친 폭이 1.82 -> 3.30 m 로 커지고 가운데 이음매가 생긴다.
- 음수 스케일은 법선을 뒤집으므로 렌더가 어두워질 수 있다. C 에는 없는 문제.

---

## 다음 단계

씨앗 궤적을 상태기계로 만든다 (텔레옵은 GUI 가 필요해 헤드리스에서 불가).

```
Franka 는 액션이 이미 eef 자세 delta + 그리퍼 1축이라
궤적 = "eef waypoint 나열" 로 끝난다. 관절 설계 불필요.

양팔이면 상태기계 2개 + 동기화 지점이 추가된다.
태스크 후보: handover (L 이 중앙으로 -> R 이 받아 자기 쪽에 놓기)
```
