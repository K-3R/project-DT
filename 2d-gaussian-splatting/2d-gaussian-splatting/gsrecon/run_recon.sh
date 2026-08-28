#!/usr/bin/env bash
# =============================================================================
# 연구실 스캔 -> 메시 파이프라인 (영상 1개 -> 버텍스컬러 PLY)
#
# 단계: [1] 영상->프레임  [2] COLMAP 포즈  [3] 2DGS 학습  [4] 메시 추출
# 어느 단계부터 다시 할지 STAGE 로 고를 수 있다 (앞 단계 산출물은 재사용).
#
# 실행 (2DGS env 안에서, 레포 루트에서):
#   conda activate surfel_splatting
#   GPU=5 bash gsrecon/run_recon.sh data/raw/desk.mp4      # 또는 파일명만: desk.mp4
#
#   GPU=5 UNTIL=2 bash gsrecon/run_recon.sh <video>   # COLMAP 까지만 (첫 판 점검)
#   GPU=5 STAGE=3 bash gsrecon/run_recon.sh <video>    # 학습부터 다시
#   GPU=5 STAGE=4 MESH=bounded bash gsrecon/run_recon.sh <video>   # 메시만 다시
#
# 산출물
#   data/raw/*.mp4                     원본 영상 (여기에 올린다)
#   data/<scene>/input/                프레임
#   data/<scene>/sparse/, images/      COLMAP (convert.py 결과)
#   output/<scene>/                    2DGS 모델
#   output/<scene>/train/ours_30000/fuse_unbounded_post.ply   <- 최종 메시
#   (data/ output/ 은 .gitignore 되어 있다)
#
# CPU: 기본 8코어로 제한된다. 어느 코어를 쓸지는 실행 시점에 가장 한가한
#      것으로 자동 선택 (pick_cores.py). CORES=16 개수 변경 / CORES=0 해제
#      / CORE_LIST="32-39" 직접 지정.
# COLMAP: CUDA 빌드 여부를 자동 판별한다 (ldd 로 libcudart 확인).
#      CUDA 없는 빌드면 GPU SIFT 가 OpenGL 폴백 -> 헤드리스에서 크래시하므로
#      자동으로 CPU 모드(--no_gpu)로 돌리고 코어를 32 로 올린다.
#      COLMAP_GPU=1|0 강제 / COLMAP_BIN=<경로> 로 CUDA 빌드 지정 가능.
# 주의: 공용 서버 -- GPU 는 nvidia-smi 로 여유 20GB+ 확인 후 지정할 것.
# =============================================================================
set -u

VIDEO="${1:-}"
# 공용 서버 규약: GPU 는 반드시 명시 (기본값이 조용히 0번 카드를 무는
# 사고 방지 -- 학습(3)과 메시(4)가 진짜로 GPU 를 쓴다)
GPU="${GPU:-}"
STAGE="${STAGE:-1}"          # 어느 단계부터 (1=프레임 2=colmap 3=학습 4=메시)
UNTIL="${UNTIL:-4}"          # 어디까지 (UNTIL=2 면 COLMAP 까지만 -- 첫 판 점검용)
MESH="${MESH:-unbounded}"    # unbounded(튜닝 불필요) | bounded(책상 주변만)
TARGET="${TARGET:-250}"      # 목표 프레임 수
FORCE="${FORCE:-0}"          # 1 이면 기존 input/ 프레임 덮어쓰기 (stage 1)
# GRAY=1 이면 프레임을 흑백으로 뽑는다 (메시 색도 흑백이 된다 --
# 우리 학습 카메라는 RGB 라서 배경만 흑백이면 색 분포가 어긋난다)
GRAY="${GRAY:-0}"
ITER="${ITER:-30000}"        # 학습 이터레이션 (2DGS 기본값)
# 정칙화 = 논문 값. 코드 기본 lambda_dist=0 은 3DGS 포팅 잔재라 그대로 쓰면
# 깊이왜곡 정칙화가 꺼진 채 학습된다 (표면이 두껍게 뭉개짐).
# 논문: alpha = 1000(bounded) / 100(unbounded), beta = 0.05
# 우리 씬은 실내 방 = unbounded -> 100. 물체 하나만 잘라 찍은 경우 1000.
LAMBDA_DIST="${LAMBDA_DIST:-100}"
LAMBDA_NORMAL="${LAMBDA_NORMAL:-0.05}"
# depth_ratio 0 = mean depth (README: 무경계/큰 씬은 disk-aliasing 감소)
DEPTH_RATIO="${DEPTH_RATIO:-0}"
MESH_RES="${MESH_RES:-1024}"
DEPTH_TRUNC="${DEPTH_TRUNC:-3.0}"   # bounded 모드에서만 사용 [m]
# bounded TSDF 융합 대역 (씬 유닛). 기본 -1 = 복셀x5 자동. 깊이 노이즈가
# 대역을 벗어나 구멍/찢김이 생기면 키운다 (구멍 메꿈 <-> 디테일 뭉툭 절충)
SDF_TRUNC="${SDF_TRUNC:--1}"
NUM_CLUSTER="${NUM_CLUSTER:-50}"    # 남길 연결 클러스터 수 (부유물 정리)
# 공용 서버 예의: CPU 코어 제한 (COLMAP mapper 는 기본이 "가용 스레드 전부"라
# 128 논리코어를 물 수 있다). CORES=0 이면 제한 없음.
# COLMAP: 시스템 apt 빌드는 CUDA 없이 컴파일된 경우가 많다. 그러면 GPU
# SIFT 가 OpenGL 로 폴백 -> 화면이 필요해서 헤드리스에서 죽는다
# (Qt "could not connect to display" -> SIGABRT). 아래에서 자동 판별한다.
COLMAP_BIN="${COLMAP_BIN:-}"        # 다른 colmap 실행파일 지정 (선택)
COLMAP_GPU="${COLMAP_GPU:-auto}"    # auto | 1 | 0
# 이미지당 SIFT 특징점 상한. CPU 전수매칭 비용이 쌍당 F^2 라, 1080p 가
# colmap 기본 상한(8192)을 치면 take1(~2000) 대비 ~16배로 폭증한다
# (실측: take5 첫 블록 미완주). 2048 = take1 에서 품질 실증된 수준.
# SIFT_MAXF=0 이면 상한 미지정(colmap 기본 8192).
SIFT_MAXF="${SIFT_MAXF:-2048}"
CORES_EXPLICIT="${CORES+1}"         # 사용자가 CORES 를 직접 줬는가
CORES="${CORES:-8}"
CORE_LIST="${CORE_LIST:-}"          # 직접 지정 예: "32-39" (비우면 자동 선택)

# 이 스크립트는 gsrecon/ 아래에 있다 -- 레포 루트는 한 단계 위
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMES="$REPO/gsrecon/extract_frames.py"

if [ -z "$VIDEO" ]; then
    echo "[gsrecon] usage: GPU=5 bash gsrecon/run_recon.sh <video.mp4>"
    exit 1
fi
if [ -z "$GPU" ]; then
    echo "[gsrecon] ERROR: set GPU explicitly (shared server), e.g. GPU=5 ..."
    exit 1
fi
# 노브 오타는 조용히 엉뚱한 산출물이 된다 -- 시작 전에 걸러낸다
case "$MESH" in bounded|unbounded) ;; *)
    echo "[gsrecon] ERROR: MESH must be bounded|unbounded (got '$MESH')"
    exit 1 ;;
esac
if ! [[ "$STAGE" =~ ^[1-4]$ ]] || ! [[ "$UNTIL" =~ ^[1-4]$ ]]; then
    echo "[gsrecon] ERROR: STAGE/UNTIL must be 1..4 (got $STAGE..$UNTIL)"
    exit 1
fi
if [ "$STAGE" -gt "$UNTIL" ]; then
    echo "[gsrecon] WARNING: STAGE > UNTIL -- nothing will run"
fi
# 헤더에 적힌 "파일명만" 호출 지원: data/raw/<이름> 으로 해석해 본다
if [ ! -f "$VIDEO" ] && [ -f "$REPO/data/raw/$VIDEO" ]; then
    VIDEO="$REPO/data/raw/$VIDEO"
fi
if [ "$STAGE" -le 1 ] && [ ! -f "$VIDEO" ]; then
    echo "[gsrecon] ERROR: video not found: $VIDEO"
    exit 1
fi

NAME=$(basename "${VIDEO%.*}")
SCENE="$REPO/data/$NAME"
MODEL="$REPO/output/$NAME"

# ---- COLMAP CUDA 지원 판별 ----
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
    echo "[gsrecon] colmap: CPU mode (no CUDA in $COLMAP_EXE)"
    # CPU SIFT + 전수 매칭은 스레드가 곧 속도다 -- 직접 지정 안 했으면 늘린다
    [ -z "$CORES_EXPLICIT" ] && CORES=32
else
    echo "[gsrecon] colmap: GPU mode"
fi

# CPU 코어 제한 프리픽스
RUN=""
if [ "$CORES" != "0" ]; then
    if command -v taskset >/dev/null 2>&1; then
        # CORE_LIST 를 안 주면 지금 한가한 코어를 골라 온다 (pick_cores.py:
        # /proc/stat 샘플링 + SMT 형제 회피, 실패 시 0..CORES-1 폴백).
        # 고정 대역(0-7)은 남들도 흔히 골라 붐빌 수 있어 기본을 자동으로 둔다.
        if [ -z "$CORE_LIST" ]; then
            CORE_LIST=$(python "$REPO/gsrecon/pick_cores.py" --n "$CORES")
        fi
        [ -z "$CORE_LIST" ] && CORE_LIST="0-$((CORES - 1))"
        RUN="taskset -c $CORE_LIST"
        echo "[gsrecon] cpu cores: $CORE_LIST ($CORES)"
    else
        echo "[gsrecon] WARNING: taskset not found -- no cpu limit"
    fi
fi

# STAGE..UNTIL 구간만 실행한다
do_stage() { [ "$STAGE" -le "$1" ] && [ "$UNTIL" -ge "$1" ]; }

echo "[gsrecon] scene=$NAME gpu=$GPU stage=$STAGE..$UNTIL mesh=$MESH"
echo "[gsrecon] scene dir: $SCENE"
# 재현성 기록: 이 런의 노브를 한 줄로 (로그가 곧 params 기록이 된다)
echo "[gsrecon] knobs: TARGET=$TARGET GRAY=$GRAY ITER=$ITER" \
     "LAMBDA_DIST=$LAMBDA_DIST LAMBDA_NORMAL=$LAMBDA_NORMAL" \
     "DEPTH_RATIO=$DEPTH_RATIO SIFT_MAXF=$SIFT_MAXF MESH_RES=$MESH_RES" \
     "DEPTH_TRUNC=$DEPTH_TRUNC SDF_TRUNC=$SDF_TRUNC NUM_CLUSTER=$NUM_CLUSTER"
export CUDA_VISIBLE_DEVICES="$GPU"

# ---- [1] 영상 -> 프레임 ----
# 영상 여러 개를 한 씬으로 합치려면 extract_frames.py 를 --merge 로 직접
# 돌린 뒤 STAGE=2 로 이 러너를 부르면 된다.
if do_stage 1; then
    echo "[gsrecon] ==== 1/4 frames ==== $(date '+%H:%M:%S')"
    EX_FLAGS=""
    [ "$GRAY" = "1" ] && EX_FLAGS="$EX_FLAGS --gray"
    [ "$FORCE" = "1" ] && EX_FLAGS="$EX_FLAGS --force"
    $RUN python "$FRAMES" --src "$VIDEO" --out "$(dirname "$SCENE")" \
        --target "$TARGET" $EX_FLAGS || exit 1
fi

# ---- [2] COLMAP 포즈 ----
# convert.py 는 <scene>/input 을 읽어 distorted/, sparse/, images/ 를 만든다.
# 서브모델이 여럿 생기면 최대 모델만 남기고 --skip_matching 으로 재실행할 것.
if do_stage 2; then
    echo "[gsrecon] ==== 2/4 colmap ==== $(date '+%H:%M:%S')"
    cd "$REPO" || exit 1
    # 이전 런 산출물 제거 (DB 가 있는 distorted/ 는 남겨 재사용).
    # 안 지우면 이번 런이 실패해도 옛 images/ 때문에 등록수가 거짓이 된다.
    rm -rf "$SCENE/images" "$SCENE/sparse"
    # [!] DB 재사용 게이트: feature_extractor 는 DB 에 이미 있는 이미지의
    # 특징점을 다시 뽑지 않는다 (take5 실측 -- SIFT_MAXF 를 바꿔도 조용히
    # 옛 8192 특징점으로 매칭이 돌았다). 특징점에 영향 주는 노브가 이전
    # 런과 다르면 재사용을 거부하고 삭제를 안내한다.
    N_IN0=$(ls -1 "$SCENE/input"/*.jpg 2>/dev/null | wc -l)
    DB_PARAMS="sift_maxf=$SIFT_MAXF gray=$GRAY frames=$N_IN0"
    DB_STAMP="$SCENE/distorted/.params"
    # 스탬프 없는 기존 DB = 게이트 도입 이전 산출물 (노브 불명). 통과시키면
    # 아래에서 현재 노브로 새 스탬프가 찍혀 옛 특징점이 영구 정당화된다.
    if [ -f "$SCENE/distorted/database.db" ] && [ ! -f "$DB_STAMP" ]; then
        echo "[gsrecon] ERROR: distorted/ DB exists but has no .params stamp"
        echo "[gsrecon] (predates the reuse gate -- its knobs are unknown)"
        echo "[gsrecon]   rm -rf $SCENE/distorted   and rerun STAGE=2"
        exit 1
    fi
    if [ -f "$DB_STAMP" ] && [ "$(cat "$DB_STAMP")" != "$DB_PARAMS" ]; then
        echo "[gsrecon] ERROR: distorted/ DB was built with different knobs"
        echo "[gsrecon]   old: $(cat "$DB_STAMP")"
        echo "[gsrecon]   new: $DB_PARAMS"
        echo "[gsrecon] features live inside the DB -- run:"
        echo "[gsrecon]   rm -rf $SCENE/distorted   and rerun STAGE=2"
        exit 1
    fi
    mkdir -p "$SCENE/distorted"
    echo "$DB_PARAMS" > "$DB_STAMP"
    CONV_FLAGS=""
    [ "$COLMAP_GPU" = "0" ] && CONV_FLAGS="$CONV_FLAGS --no_gpu"
    [ -n "$COLMAP_BIN" ] && CONV_FLAGS="$CONV_FLAGS --colmap_executable $COLMAP_BIN"
    [ "$SIFT_MAXF" != "0" ] && CONV_FLAGS="$CONV_FLAGS --max_num_features $SIFT_MAXF"
    $RUN python convert.py -s "$SCENE" $CONV_FLAGS || {
        echo "[gsrecon] ERROR: colmap failed."
        echo "[gsrecon] common causes: motion blur / reflective screens / panning in place"
        exit 1
    }
    # convert.py 의 종료코드는 신뢰 불가였다 (upstream 이 os.system 의
    # wait status 를 그대로 exit -> 하위 8비트 0 -> 항상 rc 0).
    # 패치했지만 산출물로도 재확인한다.
    if [ ! -s "$SCENE/sparse/0/images.bin" ]; then
        echo "[gsrecon] ERROR: colmap produced no sparse model ($SCENE/sparse/0)"
        exit 1
    fi
    N_IN=$(ls -1 "$SCENE/input"/*.jpg 2>/dev/null | wc -l)
    N_REG=$(ls -1 "$SCENE/images"/* 2>/dev/null | wc -l)
    echo "[gsrecon] registered $N_REG / $N_IN images"
    if [ "$N_IN" -gt 0 ] && [ "$N_REG" -lt $((N_IN * 9 / 10)) ]; then
        echo "[gsrecon] WARNING: registration below 90% -- reconstruction may be poor"
    fi
fi

# ---- [3] 2DGS 학습 ----
if do_stage 3; then
    echo "[gsrecon] ==== 3/4 train (${ITER} iters) ==== $(date '+%H:%M:%S')"
    if [ ! -s "$SCENE/sparse/0/images.bin" ]; then
        echo "[gsrecon] ERROR: no colmap model at $SCENE/sparse/0 -- run STAGE=2 first"
        exit 1
    fi
    cd "$REPO" || exit 1
    echo "[gsrecon] lambda_dist=$LAMBDA_DIST lambda_normal=$LAMBDA_NORMAL depth_ratio=$DEPTH_RATIO"
    # 주의: README 의 --lambda_distortion 은 오타. 실제 플래그는 --lambda_dist
    # --save_iterations: train.py 기본 저장은 7000/30000 뿐이라, ITER 가
    # 다른 값이면 학습 완주 후 stage 4 가 체크포인트를 못 찾는다 -> 명시 저장
    $RUN python train.py -s "$SCENE" -m "$MODEL" --iterations "$ITER" \
        --save_iterations 7000 "$ITER" \
        --depth_ratio "$DEPTH_RATIO" \
        --lambda_dist "$LAMBDA_DIST" --lambda_normal "$LAMBDA_NORMAL" \
        || { echo "[gsrecon] ERROR: train failed"; exit 1; }
fi

# ---- [4] 메시 추출 (TSDF) ----
if do_stage 4; then
    echo "[gsrecon] ==== 4/4 mesh ($MESH) ==== $(date '+%H:%M:%S')"
    if [ ! -d "$MODEL/point_cloud/iteration_$ITER" ]; then
        echo "[gsrecon] ERROR: no checkpoint at $MODEL/point_cloud/iteration_$ITER"
        echo "[gsrecon] (train saves at its own iterations -- check ITER matches)"
        exit 1
    fi
    cd "$REPO" || exit 1
    if [ "$MESH" = "bounded" ]; then
        $RUN python render.py -m "$MODEL" -s "$SCENE" --skip_train --skip_test \
            --iteration "$ITER" --depth_ratio "$DEPTH_RATIO" \
            --depth_trunc "$DEPTH_TRUNC" --mesh_res "$MESH_RES" \
            --sdf_trunc "$SDF_TRUNC" \
            --num_cluster "$NUM_CLUSTER" \
            || { echo "[gsrecon] ERROR: mesh failed"; exit 1; }
        OUT_NAME="fuse_post.ply"
    else
        $RUN python render.py -m "$MODEL" -s "$SCENE" --skip_train --skip_test \
            --iteration "$ITER" --depth_ratio "$DEPTH_RATIO" \
            --unbounded --mesh_res "$MESH_RES" --num_cluster "$NUM_CLUSTER" \
            || { echo "[gsrecon] ERROR: mesh failed"; exit 1; }
        OUT_NAME="fuse_unbounded_post.ply"
    fi
    # 사전순 glob 금지: ours_7000 이 ours_30000 보다 뒤로 정렬된다.
    # 실제로 렌더한 이터 경로를 직접 만든다.
    PLY="$MODEL/train/ours_$ITER/$OUT_NAME"
    if [ ! -s "$PLY" ]; then
        echo "[gsrecon] ERROR: mesh not produced: $PLY"
        exit 1
    fi
    echo "[gsrecon] mesh: $PLY"
    du -h "$PLY"
fi

echo "[gsrecon] done $(date '+%H:%M:%S')"
echo "[gsrecon] next: gsrecon/postprocess_mesh.py (align/scale/crop) -> replica_to_usd"
