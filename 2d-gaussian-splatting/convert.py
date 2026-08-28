#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# ======================================
# File: convert.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-28
# ======================================
# [ver] convert.py upstream + team-sr local patches r4 (2026-08-26)
#   p1: os.system wait status 절단 수정 -- 실패 경로 전부 exit(1) 통일
#       (wait status = code<<8, 그대로 exit() 하면 하위 8bit=0 이라 호출
#        script 가 실패를 놓침. 실측: mapper code 256 -> rc 0)
#   p2: --max_num_features knob (1080p 특징점 폭증 -> CPU matching 16배 방지)
#   p3: mapper 의 ba_global_function_tolerance 버전 조회 후 조건부 부착
#   p4: mapper 산출물 검증 -- 이전 run 의 stale sparse/0 소비 차단
#       (이번 실행이 실제로 새로 썼는지 mtime 확인)
# upstream 갱신 시 패치 소실 -- diff 로 복원할 것
#

import os
import logging
import time
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument(
    "--no_gpu",
    action="store_true",
)
parser.add_argument(
    "--skip_matching",
    action="store_true",
)
parser.add_argument(
    "--source_path",
    "-s",
    required=True,
    type=str,
)
parser.add_argument(
    "--camera",
    default="OPENCV",
    type=str,
)
parser.add_argument(
    "--colmap_executable",
    default="",
    type=str,
)
parser.add_argument(
    "--resize",
    action="store_true",
)
# p2: 이미지당 SIFT 특징점 상한 (0 = colmap 기본값 8192).
# 1080p 는 기본 상한까지 참. CPU 전수 matching 비용은 쌍당 F^2 라
# ~2000(take1) 대비 ~16배 폭증 (실측: take5 첫 block 미완주)
parser.add_argument(
    "--max_num_features",
    default=0,
    type=int,
)
parser.add_argument(
    "--magick_executable",
    default="",
    type=str,
)
args = parser.parse_args()

colmap_command = (
    '"{}"'.format(args.colmap_executable)
    if len(args.colmap_executable) > 0
    else "colmap"
)
magick_command = (
    '"{}"'.format(args.magick_executable)
    if len(args.magick_executable) > 0
    else "magick"
)
use_gpu = 1 if not args.no_gpu else 0

if not args.skip_matching:
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = (
        colmap_command + " feature_extractor "
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    )
    if args.max_num_features > 0:
        feat_extracton_cmd += " --SiftExtraction.max_num_features " + str(
            args.max_num_features
        )
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(1)  # p1: wait status 절단 방지 (파일 헤더 참조)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(1)  # p1: wait status 절단 방지 (파일 헤더 참조)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    # p3: 이 옵션은 COLMAP 버전에 따라 없음 (3.11 에서 옵션 정리).
    # 없는 버전에 넘기면 mapper 전체가 "unrecognised option" 으로 죽어
    # matching 결과가 버려짐 -> 도움말 조회 후 지원할 때만 부착
    ba_opt = ""
    mapper_help = os.popen(colmap_command + " mapper -h 2>&1").read()
    if "ba_global_function_tolerance" in mapper_help:
        ba_opt = " --Mapper.ba_global_function_tolerance=0.000001"
    else:
        logging.info("Mapper: ba_global_function_tolerance unsupported -- skipped")
    mapper_cmd = colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --output_path " + args.source_path + "/distorted/sparse" + ba_opt
    mapper_t0 = time.time()  # p4: 이번 실행의 산출물인지 판별 기준 시각
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(1)  # p1: wait status 절단 방지 (파일 헤더 참조)
    # p4: mapper 는 모델 0개여도 rc=0 가능. 재실행 시 distorted/sparse/0 에
    # 이전 run 의 stale 모델이 남아 있으면 undistorter 가 조용히 소비해
    # 옛 pose 로 pipeline 이 계속 진행됨 -> 새로 쓴 모델인지 검증
    model_images = args.source_path + "/distorted/sparse/0/images.bin"
    if not os.path.exists(model_images):
        logging.error(
            "Mapper produced no model (missing sparse/0/images.bin). Exiting."
        )
        exit(1)
    if os.path.getmtime(model_images) < mapper_t0 - 1.0:
        logging.error(
            "Mapper wrote no new model this run -- distorted/sparse/0 is stale "
            "output from a previous run. Remove distorted/sparse and rerun. Exiting."
        )
        exit(1)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP"
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
    exit(1)  # p1: wait status 절단 방지 (파일 헤더 참조)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == "0":
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

if args.resize:
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 50% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(1)  # p1: wait status 절단 방지 (파일 헤더 참조)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 25% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(1)  # p1: wait status 절단 방지 (파일 헤더 참조)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(
            magick_command + " mogrify -resize 12.5% " + destination_file
        )
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(1)  # p1: wait status 절단 방지 (파일 헤더 참조)

print("Done.")
