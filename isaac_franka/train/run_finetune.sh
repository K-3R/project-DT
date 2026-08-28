#!/usr/bin/env bash
# =============================================================================
# GR00T N1.5 파인튜닝 러너 (호스트 gr00t 환경에서 실행)
#
# 모드 (MODE):
#   pipe = LoRA 50스텝 배관검증 -- 데이터 로딩/저장/VRAM 실측이 목적
#   full = 풀 파인튜닝 본학습 -- batch 10 / 32k 스텝 (STEPS/SAVE 로 조절)
#
# 사용:
#   GPU=3 MODE=pipe bash run_finetune.sh
#   GPU=3 MODE=full nohup bash run_finetune.sh > ~/project/gr00t_Isaacsim/out/finetune_full.log 2>&1 &
#
# 사전 1회 패치 (Isaac-GR00T/scripts/gr00t_finetune.py 2곳 -- 러너는 검사만 함):
#   (a) 체크포인트 보존 개수:
#     sed -i -E 's/save_total_limit=[0-9]+,/save_total_limit=4,/' scripts/gr00t_finetune.py
#     (4 = 8k/16k/24k/32k 를 남겨 에폭별 성능 비교 가능. 17GB x 4 = 68GB.
#      1 이면 마지막 것만 남아 최적 에폭을 사후에 찾을 수 없다)
#   (b) GPU 강제고정 해제 (num_gpus=1 이면 무조건 0번을 쓰는 코드):
#     sed -i 's|os.environ\["CUDA_VISIBLE_DEVICES"\] = "0"|os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")|' scripts/gr00t_finetune.py
#
# 사전 1회 배치: our_configs.py 를 Isaac-GR00T 레포 루트로 복사
#
# VRAM 한도 20GB (공용서버): 시작 후 nvidia-smi 로 피크 확인.
# 초과 시 레버: batch 10 -> 8, 그래도면 full 에서 --tune-visual 제거.
# =============================================================================
set -u
export PYTHONUNBUFFERED=1

GPU="${GPU:-}"
# 공용 서버 규약: GPU 는 반드시 명시 (기본 0번이 남의 학습을 무는 사고 방지)
if [ -z "$GPU" ]; then
    echo "[ft] ERROR: set GPU explicitly (shared server), e.g. GPU=3 MODE=full ..."
    exit 1
fi
MODE="${MODE:-pipe}"
REPO="${REPO:-$HOME/project/gr00t_Isaacsim/Isaac-GR00T}"
DATASET="${DATASET:-/data1/huggingface/sslunder54/datasets/franka_bimanual_lerobot}"
BASE="${BASE:-/data1/huggingface/sslunder54/checkpoints/n1.5-3b}"
OUT="${OUT:-/data1/huggingface/sslunder54/checkpoints/bimanual_$MODE}"
BATCH="${BATCH:-10}"
STEPS="${STEPS:-32000}"
SAVE="${SAVE:-8000}"

# 웜스타트(BASE = 기존 파인튜닝 체크포인트) 시 OUT 이 BASE 를 덮으면 안 된다
if [ "$BASE" = "$OUT" ]; then
    echo "[ft] ERROR: OUT must differ from BASE (would overwrite the warm-start checkpoint)"
    exit 1
fi

# ---- 사전 검증: 잘못된 상태로 몇 시간 돌기 전에 전부 여기서 걸러낸다 ----
fail() { echo "[ft] ERROR: $1"; exit 1; }

[ -d "$REPO" ] || fail "repo not found: $REPO"
[ -f "$REPO/our_configs.py" ] || fail "our_configs.py missing in repo root -- copy it first"
[ -f "$DATASET/meta/info.json" ] || fail "dataset not found: $DATASET"
[ -f "$BASE/config.json" ] || fail "base model not found: $BASE"

# NFS 마운트 확인 (재부팅 후 마운트 유실 사고 재발 방지)
if ! df "$DATASET" 2>/dev/null | grep -q "192.168.10.101"; then
    fail "NFS not mounted -- df shows local disk for $DATASET"
fi

# gr00t_finetune.py 패치 2곳 적용 여부 (헤더의 sed 명령 참고)
keep=$(grep -oE "save_total_limit=[0-9]+," "$REPO/scripts/gr00t_finetune.py" | head -1)
[ -n "$keep" ] || fail "cannot read save_total_limit in gr00t_finetune.py"
if [ "$keep" = "save_total_limit=5," ]; then
    fail "patch (a) not applied: still the stock value 5 -- run sed (a) in header"
fi
echo "[ft] checkpoint retention: $keep"
grep -q 'setdefault("CUDA_VISIBLE_DEVICES"' "$REPO/scripts/gr00t_finetune.py" \
    || fail "patch (b) not applied: CUDA_VISIBLE_DEVICES forced to 0 -- run sed (b) in header"

case "$MODE" in
    lora)
        EXTRA="--lora-rank 64 --lora-alpha 128 --max-steps $STEPS --save-steps $SAVE"
        ;;
    pipe)
        EXTRA="--lora-rank 64 --lora-alpha 128 --max-steps 50 --save-steps 50"
        ;;
    full)
        EXTRA="--max-steps $STEPS --save-steps $SAVE"
        ;;
    *)
        fail "unknown MODE=$MODE (use lora, pipe or full)"
        ;;
esac

echo "[ft] mode=$MODE gpu=$GPU batch=$BATCH out=$OUT"
echo "[ft] dataset=$DATASET"
echo "[ft] base=$BASE"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
python scripts/gr00t_finetune.py \
    --dataset-path "$DATASET" \
    --data-config our_configs:BimanualFrankaConfig \
    --embodiment-tag new_embodiment \
    --base-model-path "$BASE" \
    --output-dir "$OUT" \
    --num-gpus 1 \
    --batch-size "$BATCH" \
    --report-to tensorboard \
    --video-backend decord \
    $EXTRA
rc=$?
echo "[ft] finished mode=$MODE rc=$rc out=$OUT"
exit $rc
