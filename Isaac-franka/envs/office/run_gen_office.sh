#!/usr/bin/env bash
# ======================================
# File: run_gen_office.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_gen_office.sh 2026-08-31
# =============================================================================
# 사무실 마커 task 씨앗 생성 + NAS 이동 (host 에서 실행)
#
# 배치 3개: 마커 1개 / 2개 / 3개 각각 ~200 성공 확보 (성공분만 저장).
# 단일 GPU 순차, 배치가 끝날 때마다 산출 폴더를 NAS 로 옮김.
# env1 의 run_gen_bimanual.sh 와 같은 구조 (점진 HDF5, KST stamp, 좀비 없음).
#
# 실행:
#   검증(소량):  GPU=1 DEMOS_OVERRIDE=3 bash run_gen_office.sh
#   본 실행:     GPU=1 nohup bash run_gen_office.sh > <클론루트>/out/gen_office_master.log 2>&1 &
#
# 배치별 시도 횟수는 수율에 맞춰 BATCHES 의 마지막 필드를 조정할 것
# (--demos 10 수율 run 으로 먼저 측정 권장).
#
# 중단:
#   pkill -f run_gen_office.sh
#   docker exec -u 0 gr00t_isaac bash -lc "pkill -9 -f gen_office.py"
# =============================================================================
set -u

GPU="${GPU:-}"
# 공용 server 규약: GPU 는 반드시 명시 (조용한 기본값 금지)
if [ -z "$GPU" ]; then
    echo "[office] ERROR: set GPU explicitly (shared server), e.g. GPU=1 ..."
    exit 1
fi
CONTAINER="${CONTAINER:-gr00t_isaac}"
# repo 자기상대: clone root = container 의 /root/project mount (OUT_CTN 과 동일 위치)
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GEN_DIR=/root/project/Isaac-franka/envs/office
OUT_CTN=/root/project/datasets/office_markers
OUT_HOST="$PROJ_ROOT/datasets/office_markers"
NAS=/data1/huggingface/sslunder54/datasets/office_markers
LOG_DIR="$PROJ_ROOT/out"

# 배치 정의: tag:seed:개수범위:시도횟수 (seed 는 배치마다 달라야 함)
# 환경변수로 덮어쓰기 가능 -- 예: 부분 완료 후 남은 배치만 재실행
#   BATCHES="m2:200:2,2:230 m3:300:3,3:250" GPU=7 nohup bash run_gen_office.sh ...
BATCHES="${BATCHES:-m1:100:1,1:220 m2:200:2,2:230 m3:300:3,3:250}"

# ---- 시작 전 검증: NAS 쓰기 가능 여부 ----
mkdir -p "$NAS" "$LOG_DIR" || { echo "[office] ERROR: cannot create $NAS"; exit 1; }
if ! touch "$NAS/.write_test" 2>/dev/null; then
    echo "[office] ERROR: no write permission on $NAS"
    exit 1
fi
rm -f "$NAS/.write_test"
if ! df "$NAS" 2>/dev/null | grep -q "192.168.10.101"; then
    echo "[office] ERROR: NFS not mounted (df shows local disk)"
    exit 1
fi

# ---- 시작 전 검증: 같은 tag 잔재 금지 ----
# kill/crash 된 이전 run 의 폴더가 home 이나 NAS 에 남아 있으면, seed 가 tag 에
# 고정돼 있어 재실행분과 동일한 episode 가 중복 유입됨 (변환기는 NAS 의
# 모든 폴더를 조용히 순회). 지우거나 옮긴 뒤 재실행할 것.
for spec in $BATCHES; do
    tag="${spec%%:*}"
    for base in "$OUT_HOST" "$NAS"; do
        for d in "$base"/*_"$tag"; do
            [ -d "$d" ] || continue
            echo "[office] ERROR: stale folder $d (same tag+seed = duplicate episodes)"
            echo "[office] delete or move it first, then relaunch"
            exit 1
        done
    done
done
echo "[office] start: gpu=$GPU nas=$NAS"

for spec in $BATCHES; do
    tag="${spec%%:*}"
    rest="${spec#*:}"
    seed="${rest%%:*}"
    rest="${rest#*:}"
    range="${rest%%:*}"
    demos="${rest#*:}"
    demos="${DEMOS_OVERRIDE:-$demos}"

    echo "[office] ==== batch $tag (seed=$seed markers=$range demos=$demos) start $(date '+%m%d %H:%M:%S') ===="
    docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc \
        "umask 000 && cd $GEN_DIR && CUDA_VISIBLE_DEVICES=$GPU /root/project/IsaacLab/isaaclab.sh -p gen_office.py --headless --demos $demos --num-markers $range --seed $seed --tag $tag --out $OUT_CTN/seed.hdf5" \
        > "$LOG_DIR/gen_office_$tag.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[office] ERROR: batch $tag exited rc=$rc, stopping (see $LOG_DIR/gen_office_$tag.log)"
        exit $rc
    fi

    moved=0
    for d in "$OUT_HOST"/*_"$tag"; do
        [ -d "$d" ] || continue
        echo "[office] move $(basename "$d") ($(du -sh "$d" | cut -f1)) -> $NAS"
        mv "$d" "$NAS/" || { echo "[office] ERROR: mv failed for $d"; exit 1; }
        moved=1
    done
    if [ $moved -eq 0 ]; then
        echo "[office] ERROR: no output folder matching *_$tag under $OUT_HOST"
        exit 1
    fi
    echo "[office] ==== batch $tag done $(date '+%m%d %H:%M:%S') ===="
done

echo "[office] all batches done. NAS contents:"
ls -la "$NAS"
du -sh "$NAS"/* 2>/dev/null
echo "[office] finished $(date '+%m%d %H:%M:%S')"
