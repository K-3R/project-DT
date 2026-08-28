#!/usr/bin/env bash
# [ver] kill.sh 2026-08-26-r2
# r2: docker exec 에 TERM=xterm 부착 (TTY 없는 로그인 셸 초기화 사망 방지),
#     pgrep/pkill 자기매칭 제거 (브래킷 패턴), replica_to_usd.py 패턴 추가
#
# Isaac Sim / Isaac Lab 프로세스 정리.
#
#   ./kill.sh          우리 스크립트만
#   ./kill.sh all      컨테이너 안 Kit/Isaac Sim 전부
#
# 왜 필요한가
#   docker exec 로 띄운 프로세스는 컨테이너 네임스페이스에 있다.
#   호스트에서 Ctrl+C 를 눌러도 docker exec 클라이언트만 죽고
#   컨테이너 안 Kit 은 그대로 남아 GPU 를 물고 있는다 (= 좀비).
#   반드시 컨테이너 안에서 죽여야 한다.
#
# 컨테이너 자체는 절대 stop/rm 하지 말 것 - Isaac Lab 설치가 날아간다.
set -uo pipefail

CONTAINER="${CONTAINER:-gr00t_isaac}"
MODE="${1:-mine}"

if [ "$MODE" = "all" ]; then
  PATTERNS=("isaac-sim" "kit" "python.sh" "isaaclab.sh")
else
  # 현행 진입점 전부 (office_scan 은 오버레이 wrapper 이름이 cmdline 에
  # 뜨므로 자기 이름이 필요하다)
  PATTERNS=("dual_franka_scene.py" "dual_franka_stack_sm.py"
            "dual_franka_gr00t_eval.py" "dual_franka_office_eval.py"
            "dual_franka_office_sm.py" "preview_office.py" "preview_replica.py"
            "eval_office_scan.py" "gen_office_scan.py" "preview_office_scan.py"
            "rerender_wrist_views.py" "replica_to_usd.py")
fi

echo "[kill] container=$CONTAINER mode=$MODE"
for p in "${PATTERNS[@]}"; do
  # 브래킷 패턴: pgrep -f 는 우리 래퍼 bash -lc 의 cmdline (패턴 문자열 포함)
  # 까지 세어 count 가 절대 0 이 되지 않는다. 첫 글자를 [x] 로 감싸면
  # 래퍼의 리터럴 "[x]..." 은 정규식 "x..." 에 안 걸리고 실제 프로세스만 잡힘
  p_re="[${p:0:1}]${p:1}"
  n=$(docker exec -u 0 -e TERM=xterm "$CONTAINER" bash -lc "pgrep -fc '$p_re' 2>/dev/null || true")
  if [ "${n:-0}" != "0" ] && [ -n "${n:-}" ]; then
    echo "  - $p : count=$n"
    docker exec -u 0 -e TERM=xterm "$CONTAINER" bash -lc "pkill -9 -f '$p_re'" || true
  fi
done

sleep 2
echo ""
echo "[remaining processes]"
docker exec -u 0 -e TERM=xterm "$CONTAINER" bash -lc \
  "ps -eo pid,etime,cmd | grep -E 'isaac|kit|python' | grep -v grep | head -10" || true

echo ""
echo "[GPU]"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv 2>/dev/null || true

echo ""
echo "if GPU memory is still held:  $0 all"
