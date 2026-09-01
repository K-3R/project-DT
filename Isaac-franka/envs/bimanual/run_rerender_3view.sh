#!/usr/bin/env bash
# ======================================
# File: run_rerender_3view.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] run_rerender_3view.sh 2026-08-31
# =============================================================================
# 3view rerender pipeline (host 에서 실행)
#
# batch 마다 순차로:
#   [1] NAS -> home 복사 (container 는 NAS 를 못 보므로)
#   [2] container 에서 rerender: 기록된 관절/cube 상태를 재생하며 손목 2 view 를
#       추가한 seed_3view.hdf5 생성 (ego 와 궤적/action 은 원본 그대로)
#   [3] seed_3view.hdf5 를 NAS batch 폴더로 이동 (새 이름이라 원본과 공존)
#   [4] home 사본 삭제 (home 점유는 항상 batch 1개분, 최대 ~25GB)
#
# 실행:
#   본 실행:  GPU=1 nohup bash run_rerender_3view.sh > ~/project/gr00t_Isaacsim/out/rr3_master.log 2>&1 &
#   test:     GPU=1 DEMOS=2 BATCHES=0806_152305_n2 bash run_rerender_3view.sh
#             (확인 후 NAS 의 해당 seed_3view.hdf5 를 지우고 본 실행할 것 --
#              부분 파일이 남으면 변환기가 그걸 그대로 읽음)
#
# 이후 단계:
#   재변환:  bash ../../convert/run_convert_bimanual.sh   (seed_3view -> ..._3view)
#   재학습:  GPU=<빈번호> MODE=full BATCH=8 STEPS=12000 SAVE=4000 \
#            BASE=/data1/huggingface/sslunder54/checkpoints/bimanual_full \
#            OUT=/data1/huggingface/sslunder54/checkpoints/bimanual_3view \
#            DATASET=/data1/huggingface/sslunder54/datasets/franka_bimanual_lerobot_3view \
#            bash ../../train/run_finetune.sh
#
# timeout / 재시도 없음. batch 는 종료까지 대기함.
# =============================================================================
set -u

GPU="${GPU:-}"
# 공용 server 규약: GPU 는 반드시 명시 (조용한 기본값 금지)
if [ -z "$GPU" ]; then
    echo "[rr3] ERROR: set GPU explicitly (shared server), e.g. GPU=1 ..."
    exit 1
fi
DEMOS="${DEMOS:-0}"          # 0 = 전체, test 는 2 등
WRIST_FOV="${WRIST_FOV:-75}"
NAS="${NAS:-/data1/huggingface/sslunder54/datasets/franka_bimanual}"
# repo 자기상대: HOME_DIR 은 CTN_DIR(/root/project/datasets/...)의 host 측
# 실체여야 함 = clone root/datasets (container mount 와 동일 위치)
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOME_DIR="${HOME_DIR:-$PROJ_ROOT/datasets/franka_bimanual}"
CTN_DIR=/root/project/datasets/franka_bimanual
LOG_DIR="$PROJ_ROOT/out"
CONTAINER="${CONTAINER:-gr00t_dt}"
BATCHES="${BATCHES:-0806_152305_n2 0806_152305_n3 0806_213259_n4 0806_182704_n5}"

# NAS mount 확인 (reboot 후 유실 사고 재발 방지)
# NFS_HOST 를 비우면 이 검사를 건너뜀 (외부 환경에서 로컬 디스크로 쓸 때)
NFS_HOST="${NFS_HOST:-192.168.10.101}"
if [ -n "$NFS_HOST" ] && ! df "$NAS" 2>/dev/null | grep -q "$NFS_HOST"; then
    echo "[rr3] ERROR: NFS not mounted (df shows local disk for $NAS)"
    exit 1
fi
mkdir -p "$HOME_DIR" "$LOG_DIR"

for b in $BATCHES; do
    echo "[rr3] ==== $b start $(date '+%m%d %H:%M:%S') ===="
    if [ ! -f "$NAS/$b/seed.hdf5" ]; then
        echo "[rr3] ERROR: missing $NAS/$b/seed.hdf5"
        exit 1
    fi
    mkdir -p "$HOME_DIR/$b"
    if [ ! -f "$HOME_DIR/$b/seed.hdf5" ]; then
        echo "[rr3] copy $b/seed.hdf5 -> home"
        cp "$NAS/$b/seed.hdf5" "$HOME_DIR/$b/"
    fi

    docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc \
        "umask 000 && cd /root/project/Isaac-franka/envs/bimanual && CUDA_VISIBLE_DEVICES=$GPU /root/project/IsaacLab/isaaclab.sh -p rerender_wrist_views.py --headless --input $CTN_DIR/$b/seed.hdf5 --out $CTN_DIR/$b/seed_3view.hdf5 --wrist-fov $WRIST_FOV --demos $DEMOS" \
        > "$LOG_DIR/rr_$b.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[rr3] ERROR: rerender $b rc=$rc (see $LOG_DIR/rr_$b.log)"
        exit $rc
    fi
    if [ ! -f "$HOME_DIR/$b/seed_3view.hdf5" ]; then
        echo "[rr3] ERROR: no seed_3view.hdf5 produced for $b"
        exit 1
    fi

    echo "[rr3] move $b/seed_3view.hdf5 ($(du -sh "$HOME_DIR/$b/seed_3view.hdf5" | cut -f1)) -> NAS"
    mv "$HOME_DIR/$b/seed_3view.hdf5" "$NAS/$b/"
    rm -f "$HOME_DIR/$b/seed.hdf5"
    rmdir "$HOME_DIR/$b" 2>/dev/null || true
    echo "[rr3] ==== $b done $(date '+%m%d %H:%M:%S') ===="
done

echo "[rr3] all batches done. NAS contents:"
ls -la "$NAS"/*/seed_3view.hdf5
