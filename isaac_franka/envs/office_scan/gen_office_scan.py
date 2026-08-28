#!/usr/bin/env python
# [ver] gen_office_scan.py 2026-08-26-r1  (ascii-only console/comments)
"""스캔 책상판 office 마커 데모 생성 -- 상태기계 생성기를 씬만 바꿔 재사용.

Track B 용: 스캔 배경에서 씨앗 데모를 생성해 재학습하면 인도메인 행이
된다. 태스크/상태기계/기록 형식은 office 와 동일 (변환기도 그대로 사용).
직접 실행보다 run_gen_office_scan.sh 권장.
"""

from _overlay import run_office_script

run_office_script("dual_franka_office_sm.py")
