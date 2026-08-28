#!/usr/bin/env bash
# =============================================================================
# 스캔 책상판 office 마커 태스크 씨앗 생성 + NAS 이동 (호스트에서 실행)
#
# run_gen_office.sh 의 스캔 배경판 (Track B: 인도메인 데이터).
# 배치/시드/수량을 office 와 동일하게 유지해 초기상태 분포를 맞춘다 --
# 두 데이터셋의 차이가 "배경"뿐이어야 비교가 성립한다.
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
# 공용 서버 규약: GPU 는 반드시 명시 (조용한 기본값 금지)
if [ -z "$GPU" ]; then
    echo "[office-scan] ERROR: set GPU explicitly (shared server), e.g. GPU=1 ..."
    exit 1
fi
CONTAINER=gr00t_isaac
GEN_DIR=/root/project/isaac_franka/envs/office_scan
OUT_CTN=/root/project/datasets/office_scan_markers
OUT_HOST="$HOME/project/gr00t_Isaacsim/datasets/office_scan_markers"
NAS=/data1/huggingface/sslunder54/datasets/office_scan_markers
LOG_DIR="$HOME/project/gr00t_Isaacsim/out"

# 배치 정의: 태그:시드:개수범위:시도횟수 (office 와 동일 시드 = 분포 정합)
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

# ---- 시작 전 검증: 같은 태그 잔재 금지 ----
# 킬/크래시된 이전 런의 폴더가 남아 있으면 시드가 태그에 고정돼 있어
# 재실행분과 동일한 에피소드가 중복 유입된다 (변환기는 NAS 의 모든 폴더를
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
