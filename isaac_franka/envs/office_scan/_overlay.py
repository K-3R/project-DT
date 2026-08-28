#!/usr/bin/env python
# [ver] _overlay.py 2026-08-25-r1  (ascii-only console/comments)
"""office 스크립트를 씬 모듈만 바꿔 실행하는 공용 관용구.

계약 (순서가 전부다):
  1. office/ 와 이 디렉토리를 sys.path 에 넣는다
  2. office_scan_scene 을 먼저 import 한다 (이때 진짜 office_scene 이
     로드되어 그 안에 바인딩된다)
  3. sys.modules["office_scene"] 을 우리 모듈로 바꿔치기한다
  4. runpy 로 office 스크립트를 __main__ 으로 실행한다 -- 그 안의
     "import office_scene" 이 우리 모듈을 받는다

주의: office 스크립트들은 말미에 os._exit(0) 를 부르는 경우가 있어
run_office_script() 호출 뒤의 코드는 실행되지 않을 수 있다 -- 이 함수
호출을 항상 러너의 마지막 문장으로 둘 것.
"""

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_OFFICE = os.path.join(_HERE, "..", "office")


def run_office_script(script_name):
    """office/<script_name> 을 office_scan 씬으로 바꿔치기해 실행한다."""
    sys.path.insert(0, _OFFICE)
    sys.path.insert(0, _HERE)
    import office_scan_scene  # noqa: PLC0415 -- 순서 계약 (모듈 docstring)

    sys.modules["office_scene"] = office_scan_scene
    runpy.run_path(os.path.join(_OFFICE, script_name), run_name="__main__")
