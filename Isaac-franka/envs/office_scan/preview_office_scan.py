#!/usr/bin/env python
# ======================================
# File: preview_office_scan.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-31
# ======================================
# [ver] preview_office_scan.py 2026-08-25-r2  (ascii-only console/comments)
"""스캔 책상판 office 씬 미리보기 -- preview_office 를 씬만 바꿔 재사용.

사용 (컨테이너 안):
    CUDA_VISIBLE_DEVICES=2 isaaclab.sh -p preview_office_scan.py --headless \
        --out /root/project/out/office_scan_preview
    # 로봇/태스크 배치까지: --robots 1 --holder 1 --items 4
    # 스캔 파일 교체: --scan-usd /root/project/datasets/<name>.usd
확정값(스캔/배치/색/조명)은 office_scan_scene.py 기본값이 정본임.
"""

from _overlay import run_office_script

run_office_script("preview_office.py")
