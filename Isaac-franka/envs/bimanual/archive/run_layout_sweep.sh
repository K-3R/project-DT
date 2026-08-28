#!/usr/bin/env bash
# [ver] run_layout_sweep.sh 2026-08-26-r2
# r2: GPU 기본값 제거 (공용 서버 규약 -- 명시 필수)
#
# 양팔 환경 후보 구성을 하나씩 돌려 결과를 따로 남긴다.
# 구성마다 시점 4장 + 배치 도해 1장 + 로그 1개가 자기 폴더에 쌓인다.
#
#   GPU=2 ./run_layout_sweep.sh              # 전체
#   GPU=2 ONLY=A ./run_layout_sweep.sh       # A 만
set -euo pipefail

GPU="${GPU:-}"
if [ -z "$GPU" ]; then
    echo "[sweep] ERROR: set GPU explicitly (shared server), e.g. GPU=2 ..."
    exit 1
fi
STEPS="${STEPS:-60}"
SEP="${SEP:-0.9}"
MIRROR_SIDE="${MIRROR_SIDE:-left}"    # left | right | both
MIRROR_MODE="${MIRROR_MODE:-2}"       # 0=끔 1=rot180 2=거울반사
CONTAINER="${CONTAINER:-gr00t_isaac}"
ISAACLAB="${ISAACLAB:-/root/project/IsaacLab}"
DIR="${DIR:-/root/project/isaac_franka/envs/bimanual/archive}"
OUT="${OUT:-/root/project/out/bimanual_sweep}"          # 컨테이너에서 본 경로
HOST_OUT="${HOST_OUT:-$HOME/project/gr00t_Isaacsim/out/bimanual_sweep}"  # 같은 곳, 호스트 경로
ONLY="${ONLY:-}"

# 이름|설명|추가인자
CONFIGS=(
  "A|SeattleLab 2장 (마운트 판 있음), 나란히|--table dual --layout parallel"
  "B|Thorlabs 1장 (평판, 마운트 판 없음), 나란히|--table single --table-usd ThorlabsTable --layout parallel"
  "C|절차적 테이블 (크기 자유), 나란히|--table proc --table-size 1.4,1.8,0.05 --layout parallel"
  "D|SeattleLab 2장, 마주보게 (handover 용)|--table dual --layout opposed"
)

for c in "${CONFIGS[@]}"; do
  name="${c%%|*}"; rest="${c#*|}"
  desc="${rest%%|*}"; extra="${rest#*|}"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue

  echo ""
  echo "=============================================="
  echo " [$name] $desc"
  echo " 인자: $extra  --base-sep $SEP"
  echo "=============================================="
  mkdir -p "$HOST_OUT/$name"

  # umask 000 : -u 0 으로 돌리면 산출물이 root 소유가 되어 호스트 계정이 못 지운다.
  #             디렉터리 777 / 파일 666 으로 만들어 누구나 덮어쓰고 지울 수 있게 한다.
  docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc "
    umask 000 && mkdir -p $OUT/$name && cd $DIR &&
    CUDA_VISIBLE_DEVICES=$GPU $ISAACLAB/isaaclab.sh -p dual_franka_scene.py \
      --headless --steps $STEPS --base-sep $SEP \
      --mirror-side $MIRROR_SIDE --table-mirror $MIRROR_MODE $extra \
      --video '' \
      --shot $OUT/$name/view.png \
      --diagram $OUT/$name/layout.png
  " 2>&1 | tee "$HOST_OUT/$name/run.log" \
    | grep -E "^\[asset|^\[robot|^\[view|^\[diagram|^=====|^Table|^두 테이블|^베이스|^  cube|^  OK|^  X|^\[done" || true
done

echo ""
echo "결과: 호스트 ~/project/gr00t_Isaacsim/out/bimanual_sweep/{A,B,C,D}/"
echo "  view_iso.png / view_top.png / view_front.png / view_side.png / layout.png"
