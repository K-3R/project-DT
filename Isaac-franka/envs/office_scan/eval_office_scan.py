#!/usr/bin/env python
# [ver] eval_office_scan.py 2026-08-25-r2  (ascii-only console/comments)
"""스캔 책상판 office 마커 태스크 폐루프 평가 -- eval 을 씬만 바꿔 재사용.

태스크/로봇/카메라/판정은 office 와 동일, 배경만 스캔 책상이다
(= 배경 도메인 갭 측정 실험). 직접 실행보다 run_eval_office_scan.sh 권장.
"""

from _overlay import run_office_script

run_office_script("dual_franka_office_eval.py")
