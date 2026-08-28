#!/usr/bin/env bash
# =============================================================================
# Replica 스캔 데이터셋 다운로드 + office 씬 mesh.ply 추출 (호스트에서 실행)
#
# Replica (facebookresearch/Replica-Dataset) 는 통짜 tar.gz 를 2GB 씩 17개로
# 쪼갠 형태로만 배포된다 (부분 다운로드 불가, 총 37.9GB). 이 스크립트는:
#   1. 분할 파트 17개를 NAS 로 받는다 (wget -c 라 중단돼도 이어받기)
#   2. 파트별 기대 크기를 검증한다 (aa..ap = 2GB 정확히, aq = 1.86GB)
#   3. 스트림 해제하며 office_0..4 의 mesh.ply 만 추출한다 (PTEX 텍스처는
#      ReplicaSDK 없이 못 쓰므로 받지 않는다 -- 디스크 절약)
#
# 실행:
#   nohup bash download_replica.sh > ~/project/gr00t_Isaacsim/out/replica_dl.log 2>&1 &
# 재실행해도 안전하다 (받은 파트는 건너뛰고, 추출만 다시 한다).
#
# 다음 단계 (변환): 컨테이너는 NAS 를 못 보므로 mesh.ply 를 홈으로 복사 후
# replica_to_usd.py 실행 -- docs/replica_background.md 참조.
# =============================================================================
set -u

NAS_BASE=/data1/huggingface/sslunder54/replica
PARTS_DIR="$NAS_BASE/parts"
REL=https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0
SCENES="${SCENES:-office_0 office_1 office_2 office_3 office_4}"
# 마지막 파트(aq)만 크기가 다르다. 나머지는 정확히 2,000,000,000 바이트
SIZE_NORMAL=2000000000
SIZE_LAST=1859047808

# ---- 시작 전 검증: NFS 마운트 + 쓰기 가능 ----
mkdir -p "$PARTS_DIR" || { echo "[replica] ERROR: cannot create $PARTS_DIR"; exit 1; }
if ! df "$NAS_BASE" 2>/dev/null | grep -q "192.168.10.101"; then
    echo "[replica] ERROR: NFS not mounted (df shows local disk)"
    exit 1
fi
if ! touch "$NAS_BASE/.write_test" 2>/dev/null; then
    echo "[replica] ERROR: no write permission on $NAS_BASE"
    exit 1
fi
rm -f "$NAS_BASE/.write_test"

# 압축 해제기: pigz 가 있으면 병렬, 없으면 gzip (느리지만 동작)
if command -v unpigz >/dev/null 2>&1; then
    DECOMP="unpigz -p 8"
else
    DECOMP="gunzip -c"
    echo "[replica] note: pigz not found, using single-thread gunzip"
fi

echo "[replica] start $(date '+%m%d %H:%M:%S') nas=$NAS_BASE scenes=$SCENES"

# ---- 1. 분할 파트 다운로드 (이어받기) ----
SUFFIXES="aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq"
for suf in $SUFFIXES; do
    f="$PARTS_DIR/replica_v1_0.tar.gz.part$suf"
    want=$SIZE_NORMAL
    [ "$suf" = "aq" ] && want=$SIZE_LAST
    have=0
    [ -f "$f" ] && have=$(stat -c %s "$f")
    if [ "$have" = "$want" ]; then
        echo "[replica] part$suf ok (cached)"
        continue
    fi
    echo "[replica] part$suf downloading ($(date '+%H:%M:%S'))"
    wget -c -q -O "$f" "$REL/replica_v1_0.tar.gz.part$suf"
    rc=$?
    have=0
    [ -f "$f" ] && have=$(stat -c %s "$f")
    if [ $rc -ne 0 ] || [ "$have" != "$want" ]; then
        echo "[replica] ERROR: part$suf rc=$rc size=$have (want $want). rerun to resume"
        exit 1
    fi
done
echo "[replica] all 17 parts verified ($(du -sh "$PARTS_DIR" | cut -f1))"

# ---- 2. office 씬의 mesh.ply 만 스트림 추출 ----
# 통짜 스트림을 한 번 훑는다 (수십 분). 패턴 앞의 * 는 tar 내부 경로가
# "office_0/..." 인지 "./office_0/..." 인지 확정 못 해서 (둘 다 매칭).
PATTERNS=""
ALL_DONE=1
for s in $SCENES; do
    if [ -f "$NAS_BASE/$s/mesh.ply" ]; then
        echo "[replica] $s/mesh.ply ok (cached)"
    else
        ALL_DONE=0
    fi
    PATTERNS="$PATTERNS --wildcards *$s/mesh.ply"
done
if [ $ALL_DONE -eq 1 ]; then
    echo "[replica] all scene meshes already extracted"
else
    echo "[replica] extracting (one pass over 37.9GB, be patient)"
    # shellcheck disable=SC2086
    cat "$PARTS_DIR"/replica_v1_0.tar.gz.part?? | $DECOMP | tar -x -C "$NAS_BASE" $PATTERNS
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[replica] ERROR: extract rc=$rc"
        exit 1
    fi
fi

echo "[replica] result:"
for s in $SCENES; do
    if [ -f "$NAS_BASE/$s/mesh.ply" ]; then
        echo "  $s/mesh.ply $(du -h "$NAS_BASE/$s/mesh.ply" | cut -f1)"
    else
        echo "  $s/mesh.ply MISSING (check tar member names: tar -t | head)"
    fi
done
echo "[replica] done $(date '+%m%d %H:%M:%S')"
echo "[replica] parts kept in $PARTS_DIR (37.9GB). delete manually if not needed"
