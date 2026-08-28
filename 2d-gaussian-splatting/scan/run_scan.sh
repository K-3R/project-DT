#!/usr/bin/env bash
# ======================================
# File: run_scan.sh
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-28
# ======================================
# [ver] run_scan.sh 2026-08-28  (구명: gsrecon_run.sh -> run_recon.sh -> run_scan.sh)
# =============================================================================
# scan -> mesh pipeline (영상 1개 -> vertex color PLY)
#
# 단계: [1] 영상->frame  [2] COLMAP pose  [3] 2DGS 학습  [4] mesh 추출
#       STAGE=시작 단계, UNTIL=끝 단계 (앞 단계 산출물은 재사용)
#
# 실행 (repo root 에서, surfel_splatting env):
#   GPU=5 bash scan/run_scan.sh data/raw/desk.mp4       # 파일명만도 가능
#   GPU=5 UNTIL=2 bash scan/run_scan.sh <video>         # COLMAP 까지만
#   GPU=5 STAGE=4 MESH=bounded bash scan/run_scan.sh <video>    # mesh 만 다시
#
# 산출물: data/<scene>/{input,sparse,images}/, output/<scene>/
#         최종 mesh = output/<scene>/train/ours_30000/fuse_unbounded_post.ply
# =============================================================================
set -u

VIDEO="${1:-}"
GPU="${GPU:-}"               # 필수 (기본값 없음 -- GPU 0 오점유 방지)
STAGE="${STAGE:-1}"          # 시작 단계 (1=frame 2=colmap 3=학습 4=mesh)
UNTIL="${UNTIL:-4}"          # 끝 단계
MESH="${MESH:-unbounded}"    # unbounded | bounded(책상 주변만)
TARGET="${TARGET:-250}"      # 목표 frame 수
FORCE="${FORCE:-0}"          # 1 = 기존 input/ 덮어쓰기
GRAY="${GRAY:-0}"            # 1 = frame 흑백 추출 (mesh 색도 흑백이 됨)
ITER="${ITER:-30000}"        # 학습 iteration
# lambda = 논문 값. 코드 기본 lambda_dist=0 은 3DGS 잔재 (표면 뭉개짐).
# 실내 방 = unbounded -> 100, 물체 하나만 잘라 찍으면 1000
LAMBDA_DIST="${LAMBDA_DIST:-100}"
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.05}"
DEPTH_RATIO="${DEPTH_RATIO:-0}"     # 0 = mean depth (무경계 scene 권장)
MESH_RES="${MESH_RES:-1024}"
DEPTH_TRUNC="${DEPTH_TRUNC:-3.0}"   # bounded 전용 [m]
SDF_TRUNC="${SDF_TRUNC:--1}"        # TSDF 융합 대역. -1 = 자동, 구멍나면 키울 것
NUM_CLUSTER="${NUM_CLUSTER:-50}"    # 남길 연결 cluster 수 (부유물 정리)
COLMAP_BIN="${COLMAP_BIN:-}"        # colmap 실행파일 직접 지정 (선택)
COLMAP_GPU="${COLMAP_GPU:-auto}"    # auto | 1 | 0
SIFT_MAXF="${SIFT_MAXF:-2048}"      # SIFT 특징점 상한. 0 = colmap 기본(8192)
CORES_EXPLICIT="${CORES+1}"         # CORES 직접 지정 여부
CORES="${CORES:-8}"                 # CPU core 제한. 0 = 해제
CORE_LIST="${CORE_LIST:-}"          # 예: "32-39" (비우면 한가한 core 자동 선택)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMES="$REPO/scan/extract_frames.py"

if [ -z "$VIDEO" ]; then
    echo "[scan] usage: GPU=5 bash scan/run_scan.sh <video.mp4>"
    exit 1
fi
if [ -z "$GPU" ]; then
    echo "[scan] ERROR: set GPU explicitly (shared server), e.g. GPU=5 ..."
    exit 1
fi
case "$MESH" in bounded|unbounded) ;; *)
    echo "[scan] ERROR: MESH must be bounded|unbounded (got '$MESH')"
    exit 1 ;;
esac
if ! [[ "$STAGE" =~ ^[1-4]$ ]] || ! [[ "$UNTIL" =~ ^[1-4]$ ]]; then
    echo "[scan] ERROR: STAGE/UNTIL must be 1..4 (got $STAGE..$UNTIL)"
    exit 1
fi
if [ "$STAGE" -gt "$UNTIL" ]; then
    echo "[scan] WARNING: STAGE > UNTIL -- nothing will run"
fi
# 파일명만 주면 data/raw/ 에서 찾음
if [ ! -f "$VIDEO" ] && [ -f "$REPO/data/raw/$VIDEO" ]; then
    VIDEO="$REPO/data/raw/$VIDEO"
fi
if [ "$STAGE" -le 1 ] && [ ! -f "$VIDEO" ]; then
    echo "[scan] ERROR: video not found: $VIDEO"
    exit 1
fi

NAME=$(basename "${VIDEO%.*}")
SCENE="$REPO/data/$NAME"
MODEL="$REPO/output/$NAME"

# ---- COLMAP CUDA 판별 (CUDA 없는 build 는 headless 에서 죽음 -> CPU mode) ----
COLMAP_EXE="$COLMAP_BIN"
[ -z "$COLMAP_EXE" ] && COLMAP_EXE="$(command -v colmap 2>/dev/null)"
if [ "$COLMAP_GPU" = "auto" ]; then
    if [ -n "$COLMAP_EXE" ] && ldd "$COLMAP_EXE" 2>/dev/null | grep -qi "libcudart"; then
        COLMAP_GPU=1
    else
        COLMAP_GPU=0
    fi
fi
if [ "$COLMAP_GPU" = "0" ]; then
    echo "[scan] colmap: CPU mode (no CUDA in $COLMAP_EXE)"
    # CPU matching 은 thread 가 곧 속도 -- 직접 지정 없으면 늘림
    [ -z "$CORES_EXPLICIT" ] && CORES=32
else
    echo "[scan] colmap: GPU mode"
fi

# ---- CPU core 제한 (taskset) ----
RUN=""
if [ "$CORES" != "0" ]; then
    if command -v taskset >/dev/null 2>&1; then
        if [ -z "$CORE_LIST" ]; then
            CORE_LIST=$(python "$REPO/scan/pick_cores.py" --n "$CORES")
        fi
        [ -z "$CORE_LIST" ] && CORE_LIST="0-$((CORES - 1))"
        RUN="taskset -c $CORE_LIST"
        echo "[scan] cpu cores: $CORE_LIST ($CORES)"
    else
        echo "[scan] WARNING: taskset not found -- no cpu limit"
    fi
fi

# STAGE..UNTIL 구간만 실행
do_stage() { [ "$STAGE" -le "$1" ] && [ "$UNTIL" -ge "$1" ]; }

echo "[scan] scene=$NAME gpu=$GPU stage=$STAGE..$UNTIL mesh=$MESH"
echo "[scan] scene dir: $SCENE"
echo "[scan] knobs: TARGET=$TARGET GRAY=$GRAY ITER=$ITER" \
     "LAMBDA_DIST=$LAMBDA_DIST LAMBDA_NORMAL=$LAMBDA_NORMAL" \
     "DEPTH_RATIO=$DEPTH_RATIO SIFT_MAXF=$SIFT_MAXF MESH_RES=$MESH_RES" \
     "DEPTH_TRUNC=$DEPTH_TRUNC SDF_TRUNC=$SDF_TRUNC NUM_CLUSTER=$NUM_CLUSTER"
export CUDA_VISIBLE_DEVICES="$GPU"

# ---- [1] 영상 -> frame ----
# 여러 영상 합치기: extract_frames.py --merge 로 직접 돌린 뒤 STAGE=2
if do_stage 1; then
    echo "[scan] ==== 1/4 frames ==== $(date '+%H:%M:%S')"
    EX_FLAGS=""
    [ "$GRAY" = "1" ] && EX_FLAGS="$EX_FLAGS --gray"
    [ "$FORCE" = "1" ] && EX_FLAGS="$EX_FLAGS --force"
    $RUN python "$FRAMES" --src "$VIDEO" --out "$(dirname "$SCENE")" \
        --target "$TARGET" $EX_FLAGS || exit 1
fi

# ---- [2] COLMAP pose (convert.py -> distorted/, sparse/, images/) ----
if do_stage 2; then
    echo "[scan] ==== 2/4 colmap ==== $(date '+%H:%M:%S')"
    cd "$REPO" || exit 1
    # 옛 산출물 제거. DB 있는 distorted/ 는 남겨 재사용
    rm -rf "$SCENE/images" "$SCENE/sparse"
    # DB 재사용 gate: 특징점이 DB 안에 살아서 knob 을 바꿔도 조용히 옛
    # 특징점으로 돎 -> knob(stamp)이 같을 때만 재사용 허용
    N_IN0=$(ls -1 "$SCENE/input"/*.jpg 2>/dev/null | wc -l)
    DB_PARAMS="sift_maxf=$SIFT_MAXF gray=$GRAY frames=$N_IN0"
    DB_STAMP="$SCENE/distorted/.params"
    if [ -f "$SCENE/distorted/database.db" ] && [ ! -f "$DB_STAMP" ]; then
        echo "[scan] ERROR: distorted/ DB exists but has no .params stamp"
        echo "[scan] (predates the reuse gate -- its knobs are unknown)"
        echo "[scan]   rm -rf $SCENE/distorted   and rerun STAGE=2"
        exit 1
    fi
    if [ -f "$DB_STAMP" ] && [ "$(cat "$DB_STAMP")" != "$DB_PARAMS" ]; then
        echo "[scan] ERROR: distorted/ DB was built with different knobs"
        echo "[scan]   old: $(cat "$DB_STAMP")"
        echo "[scan]   new: $DB_PARAMS"
        echo "[scan] features live inside the DB -- run:"
        echo "[scan]   rm -rf $SCENE/distorted   and rerun STAGE=2"
        exit 1
    fi
    mkdir -p "$SCENE/distorted"
    echo "$DB_PARAMS" > "$DB_STAMP"
    CONV_FLAGS=""
    [ "$COLMAP_GPU" = "0" ] && CONV_FLAGS="$CONV_FLAGS --no_gpu"
    [ -n "$COLMAP_BIN" ] && CONV_FLAGS="$CONV_FLAGS --colmap_executable $COLMAP_BIN"
    [ "$SIFT_MAXF" != "0" ] && CONV_FLAGS="$CONV_FLAGS --max_num_features $SIFT_MAXF"
    $RUN python convert.py -s "$SCENE" $CONV_FLAGS || {
        echo "[scan] ERROR: colmap failed."
        echo "[scan] common causes: motion blur / reflective screens / panning in place"
        exit 1
    }
    # convert.py 는 실패해도 rc 0 인 경우가 있어 산출물로 재확인
    if [ ! -s "$SCENE/sparse/0/images.bin" ]; then
        echo "[scan] ERROR: colmap produced no sparse model ($SCENE/sparse/0)"
        exit 1
    fi
    N_IN=$(ls -1 "$SCENE/input"/*.jpg 2>/dev/null | wc -l)
    N_REG=$(ls -1 "$SCENE/images"/* 2>/dev/null | wc -l)
    echo "[scan] registered $N_REG / $N_IN images"
    if [ "$N_IN" -gt 0 ] && [ "$N_REG" -lt $((N_IN * 9 / 10)) ]; then
        echo "[scan] WARNING: registration below 90% -- reconstruction may be poor"
    fi
fi

# ---- [3] 2DGS 학습 ----
if do_stage 3; then
    echo "[scan] ==== 3/4 train (${ITER} iters) ==== $(date '+%H:%M:%S')"
    if [ ! -s "$SCENE/sparse/0/images.bin" ]; then
        echo "[scan] ERROR: no colmap model at $SCENE/sparse/0 -- run STAGE=2 first"
        exit 1
    fi
    cd "$REPO" || exit 1
    echo "[scan] lambda_dist=$LAMBDA_DIST lambda_normal=$LAMBDA_NORMAL depth_ratio=$DEPTH_RATIO"
    # save_iterations 명시: train.py 기본 저장은 7000/30000 뿐
    $RUN python train.py -s "$SCENE" -m "$MODEL" --iterations "$ITER" \
        --save_iterations 7000 "$ITER" \
        --depth_ratio "$DEPTH_RATIO" \
        --lambda_dist "$LAMBDA_DIST" --lambda_normal "$LAMBDA_NORMAL" \
        || { echo "[scan] ERROR: train failed"; exit 1; }
fi

# ---- [4] mesh 추출 (TSDF) ----
if do_stage 4; then
    echo "[scan] ==== 4/4 mesh ($MESH) ==== $(date '+%H:%M:%S')"
    if [ ! -d "$MODEL/point_cloud/iteration_$ITER" ]; then
        echo "[scan] ERROR: no checkpoint at $MODEL/point_cloud/iteration_$ITER"
        echo "[scan] (train saves at its own iterations -- check ITER matches)"
        exit 1
    fi
    cd "$REPO" || exit 1
    if [ "$MESH" = "bounded" ]; then
        $RUN python render.py -m "$MODEL" -s "$SCENE" --skip_train --skip_test \
            --iteration "$ITER" --depth_ratio "$DEPTH_RATIO" \
            --depth_trunc "$DEPTH_TRUNC" --mesh_res "$MESH_RES" \
            --sdf_trunc "$SDF_TRUNC" \
            --num_cluster "$NUM_CLUSTER" \
            || { echo "[scan] ERROR: mesh failed"; exit 1; }
        OUT_NAME="fuse_post.ply"
    else
        $RUN python render.py -m "$MODEL" -s "$SCENE" --skip_train --skip_test \
            --iteration "$ITER" --depth_ratio "$DEPTH_RATIO" \
            --unbounded --mesh_res "$MESH_RES" --num_cluster "$NUM_CLUSTER" \
            || { echo "[scan] ERROR: mesh failed"; exit 1; }
        OUT_NAME="fuse_unbounded_post.ply"
    fi
    # ours_$ITER 경로 직접 구성 (glob 은 ours_7000 이 뒤로 정렬되는 함정)
    PLY="$MODEL/train/ours_$ITER/$OUT_NAME"
    if [ ! -s "$PLY" ]; then
        echo "[scan] ERROR: mesh not produced: $PLY"
        exit 1
    fi
    echo "[scan] mesh: $PLY"
    du -h "$PLY"
fi

echo "[scan] done $(date '+%H:%M:%S')"
echo "[scan] next: scan/postprocess_mesh.py (align/scale/crop) -> replica_to_usd"
