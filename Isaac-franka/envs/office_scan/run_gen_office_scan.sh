#!/usr/bin/env bash
# ======================================
# File: run_gen_office_scan.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_gen_office_scan.sh 2026-08-31
# =============================================================================
# scan 책상판 office marker task 씨앗 생성 + NAS 이동 (host 에서 실행)
#
# run_gen_office.sh 의 scan 배경판 (Track B: in-domain 데이터).
# batch/seed/수량을 office 와 동일하게 유지해 초기상태 분포를 맞춤 --
# 두 dataset 의 차이가 "배경"뿐이어야 비교가 성립함.
# 산출 폴더/NAS 는 office_scan_markers 로 분리 (office 데이터와 섞임 방지).
#
# 실행:
#   검증(소량):  GPU=1 DEMOS_OVERRIDE=3 bash run_gen_office_scan.sh
#   본 실행:     GPU=1 nohup bash run_gen_office_scan.sh \
#                  > ~/project/gr00t_Isaacsim/out/gen_office_scan_master.log 2>&1 &
#
# 중단:
#   pkill -f run_gen_office_scan.sh
#   docker exec -u 0 gr00t_isaac bash -lc "pkill -9 -f gen_office_scan.py"
# =============================================================================
set -u

GPU="${GPU:-}"
# 공용 server 규약: GPU 는 반드시 명시 (조용한 기본값 금지)
if [ -z "$GPU" ]; then
    echo "[office-scan] ERROR: set GPU explicitly (shared server), e.g. GPU=1 ..."
    exit 1
fi
CONTAINER="${CONTAINER:-gr00t_isaac}"
# repo 자기상대: clone root = container 의 /root/project mount (OUT_CTN 과 동일 위치)
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GEN_DIR=/root/project/Isaac-franka/envs/office_scan
OUT_CTN=/root/project/datasets/office_scan_markers
OUT_HOST="$PROJ_ROOT/datasets/office_scan_markers"
NAS=/data1/huggingface/sslunder54/datasets/office_scan_markers
LOG_DIR="$PROJ_ROOT/out"

# batch 정의: tag:seed:개수범위:시도횟수 (office 와 동일 seed = 분포 정합)
BATCHES="${BATCHES:-m1:100:1,1:220 m2:200:2,2:230 m3:300:3,3:250}"

# ---- 시작 전 검증: NAS 쓰기 가능 여부 ----
mkdir -p "$NAS" "$LOG_DIR" || { echo "[office-scan] ERROR: cannot create $NAS"; exit 1; }
if ! touch "$NAS/.write_test" 2>/dev/null; then
    echo "[office-scan] ERROR: no write permission on $NAS"
    exit 1
fi
rm -f "$NAS/.write_test"
if ! df "$NAS" 2>/dev/null | grep -q "192.168.10.101"; then
    echo "[office-scan] ERROR: NFS not mounted (df shows local disk)"
    exit 1
fi

# ---- 시작 전 검증: 같은 tag 잔재 금지 ----
# kill/crash 된 이전 run 의 폴더가 남아 있으면 seed 가 tag 에 고정이라
# 재실행분과 동일한 episode 가 중복 유입됨 (변환기는 NAS 의 모든 폴더를
# 조용히 순회). 지우거나 옮긴 뒤 재실행할 것.
for spec in $BATCHES; do
    tag="${spec%%:*}"
    for base in "$OUT_HOST" "$NAS"; do
        for d in "$base"/*_"$tag"; do
            [ -d "$d" ] || continue
            echo "[office-scan] ERROR: stale folder $d (same tag+seed = duplicate episodes)"
            echo "[office-scan] delete or move it first, then relaunch"
            exit 1
        done
    done
done
echo "[office-scan] start: gpu=$GPU nas=$NAS"

for spec in $BATCHES; do
    tag="${spec%%:*}"
    rest="${spec#*:}"
    seed="${rest%%:*}"
    rest="${rest#*:}"
    range="${rest%%:*}"
    demos="${rest#*:}"
    demos="${DEMOS_OVERRIDE:-$demos}"

    echo "[office-scan] ==== batch $tag (seed=$seed markers=$range demos=$demos) start $(date '+%m%d %H:%M:%S') ===="
    docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc \
        "umask 000 && cd $GEN_DIR && CUDA_VISIBLE_DEVICES=$GPU /root/project/IsaacLab/isaaclab.sh -p gen_office_scan.py --headless --demos $demos --num-markers $range --seed $seed --tag $tag --out $OUT_CTN/seed.hdf5" \
        > "$LOG_DIR/gen_office_scan_$tag.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[office-scan] ERROR: batch $tag exited rc=$rc, stopping (see $LOG_DIR/gen_office_scan_$tag.log)"
        exit $rc
    fi

    moved=0
    for d in "$OUT_HOST"/*_"$tag"; do
        [ -d "$d" ] || continue
        echo "[office-scan] move $(basename "$d") ($(du -sh "$d" | cut -f1)) -> $NAS"
        mv "$d" "$NAS/" || { echo "[office-scan] ERROR: mv failed for $d"; exit 1; }
        moved=1
    done
    if [ $moved -eq 0 ]; then
        echo "[office-scan] ERROR: no output folder matching *_$tag under $OUT_HOST"
        exit 1
    fi
    echo "[office-scan] ==== batch $tag done $(date '+%m%d %H:%M:%S') ===="
done

echo "[office-scan] all batches done. NAS contents:"
ls -la "$NAS"
du -sh "$NAS"/* 2>/dev/null
echo "[office-scan] finished $(date '+%m%d %H:%M:%S')"
