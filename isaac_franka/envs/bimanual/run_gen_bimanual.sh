#!/usr/bin/env bash
# =============================================================================
# 양팔 Franka 씨앗 800개 생성 + NAS 이동 파이프라인 (호스트에서 실행)
#
# 구조: 2레인 병렬 -- 레인 A(GPU_A) = n2 -> n5, 레인 B(GPU_B) = n3 -> n4.
#       레인 안에서는 순차, 레인끼리는 병렬. 페어링은 부하 균형 목적
#       (에피소드 평균 길이 기준 n2+n5 와 n3+n4 가 비슷 -> 동시 종료).
#       각 배치가 끝나는 즉시 산출 폴더를 NAS 로 옮긴다. 홈 디스크에는
#       레인당 배치 1개 분량만 머문다.
#
# 실행 (GPU_A/GPU_B 는 반드시 명시, DEMOS 기본 210):
#   검증(소량):  GPU_A=0 GPU_B=1 DEMOS=2 bash run_gen_bimanual.sh
#   본 실행:     GPU_A=0 GPU_B=1 nohup bash run_gen_bimanual.sh > ~/project/gr00t_Isaacsim/out/gen_800_master.log 2>&1 &
#
# 진행 확인:
#   tail -f ~/project/gr00t_Isaacsim/out/gen_800_master.log
#   tail -f ~/project/gr00t_Isaacsim/out/gen_n2.log   (배치별 상세)
#
# 중단 (레인이 백그라운드 자식이라 스크립트만 죽이면 안 됨):
#   pkill -f run_gen_bimanual.sh
#   docker exec -u 0 gr00t_isaac bash -lc "pkill -f dual_franka_stack_sm.py"
#
# 타임아웃 / 재시도 / 워치독 없음 -- 각 배치는 끝날 때까지 그냥 기다린다.
# =============================================================================
set -u

GPU_A="${GPU_A:-}"
GPU_B="${GPU_B:-}"
# 공용 서버 규약: GPU 는 반드시 명시 (조용한 기본값 금지)
if [ -z "$GPU_A" ] || [ -z "$GPU_B" ]; then
    echo "[gen800] ERROR: set GPU_A and GPU_B explicitly (shared server), e.g. GPU_A=0 GPU_B=1 ..."
    exit 1
fi
DEMOS="${DEMOS:-210}"
CONTAINER=gr00t_isaac
GEN_DIR=/root/project/isaac_franka/envs/bimanual
OUT_CTN=/root/project/datasets/franka_bimanual
OUT_HOST="$HOME/project/gr00t_Isaacsim/datasets/franka_bimanual"
NAS=/data1/huggingface/sslunder54/datasets/franka_bimanual
LOG_DIR="$HOME/project/gr00t_Isaacsim/out"

# 레인 정의: 태그:시드:큐브개수 (시드는 배치마다 달라야 함)
LANE_A="n2:100:2,2 n5:400:5,5"
LANE_B="n3:200:3,3 n4:300:4,4"

# 레인 하나 = 배치들을 순차로 [생성 -> rc 검사 -> NAS 이동]
run_lane() {
    local lane="$1" gpu="$2" specs="$3"
    local spec tag rest seed cubes rc moved d
    for spec in $specs; do
        tag="${spec%%:*}"
        rest="${spec#*:}"
        seed="${rest%%:*}"
        cubes="${rest#*:}"

        echo "[gen800:$lane] ==== batch $tag (gpu=$gpu seed=$seed cubes=$cubes) start $(date '+%m%d %H:%M:%S') ===="
        docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 "$CONTAINER" bash -lc \
            "umask 000 && cd $GEN_DIR && CUDA_VISIBLE_DEVICES=$gpu /root/project/IsaacLab/isaaclab.sh -p dual_franka_stack_sm.py --headless --demos $DEMOS --num-cubes $cubes --seed $seed --tag $tag --out $OUT_CTN/seed.hdf5" \
            > "$LOG_DIR/gen_$tag.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "[gen800:$lane] ERROR: batch $tag exited rc=$rc, lane stopped (see $LOG_DIR/gen_$tag.log)"
            return $rc
        fi

        # 생성 직후 이동: 이 배치의 타임스탬프 폴더(*_태그)를 NAS 로 옮긴다
        moved=0
        for d in "$OUT_HOST"/*_"$tag"; do
            [ -d "$d" ] || continue
            echo "[gen800:$lane] move $(basename "$d") ($(du -sh "$d" | cut -f1)) -> $NAS"
            mv "$d" "$NAS/" || { echo "[gen800:$lane] ERROR: mv failed for $d"; return 1; }
            moved=1
        done
        if [ $moved -eq 0 ]; then
            echo "[gen800:$lane] ERROR: no output folder matching *_$tag under $OUT_HOST"
            return 1
        fi
        echo "[gen800:$lane] ==== batch $tag done $(date '+%m%d %H:%M:%S') ===="
    done
    return 0
}

# ---- 시작 전 검증: NAS 쓰기 가능 여부 (하루 돌린 뒤 이동 실패 방지) ----
mkdir -p "$NAS" "$LOG_DIR" || { echo "[gen800] ERROR: cannot create $NAS"; exit 1; }
if ! touch "$NAS/.write_test" 2>/dev/null; then
    echo "[gen800] ERROR: no write permission on $NAS"
    exit 1
fi
rm -f "$NAS/.write_test"
# NFS 마운트 확인 (재부팅 후 로컬 디스크에 조용히 쌓이는 사고 방지 --
# office/office_scan 러너와 동일 게이트)
if ! df "$NAS" 2>/dev/null | grep -q "192.168.10.101"; then
    echo "[gen800] ERROR: NFS not mounted (df shows local disk)"
    exit 1
fi

# ---- 시작 전 검증: 같은 태그 잔재 금지 (office 러너와 동일 게이트) ----
# 킬/크래시된 이전 런의 폴더가 홈/NAS 에 남아 있으면 시드가 태그에 고정돼
# 있어 재실행분과 동일한 에피소드가 중복 유입된다 (변환기는 NAS 의 모든
# 폴더를 조용히 순회). 지우거나 옮긴 뒤 재실행할 것.
for spec in $LANE_A $LANE_B; do
    tag="${spec%%:*}"
    for base in "$OUT_HOST" "$NAS"; do
        for d in "$base"/*_"$tag"; do
            [ -d "$d" ] || continue
            echo "[gen800] ERROR: stale folder $d (same tag+seed = duplicate episodes)"
            echo "[gen800] delete or move it first, then relaunch"
            exit 1
        done
    done
done
echo "[gen800] start: laneA gpu=$GPU_A [$LANE_A]  laneB gpu=$GPU_B [$LANE_B]  demos=$DEMOS per batch"

run_lane A "$GPU_A" "$LANE_A" &
pid_a=$!
run_lane B "$GPU_B" "$LANE_B" &
pid_b=$!
wait "$pid_a"
rc_a=$?
wait "$pid_b"
rc_b=$?

if [ $rc_a -ne 0 ] || [ $rc_b -ne 0 ]; then
    echo "[gen800] ERROR: lane A rc=$rc_a, lane B rc=$rc_b"
    exit 1
fi

echo "[gen800] all batches done. NAS contents:"
ls -la "$NAS"
du -sh "$NAS"/* 2>/dev/null
echo "[gen800] finished $(date '+%m%d %H:%M:%S')"
