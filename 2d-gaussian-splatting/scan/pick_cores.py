#!/usr/bin/env python
# ======================================
# File: pick_cores.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-28
# ======================================
# [ver] pick_cores.py 2026-08-20-r1  (ascii-only console/comments)
"""지금 한가한 CPU core N 개를 골라 taskset 형식으로 출력.

0-7 처럼 대역을 고정해 두면 하필 그 core 가 붐빌 때 피할 방법이 없음.
  1. /proc/stat 두 번 sampling -> core 별 사용률
  2. SMT 형제는 빼고 물리 core 당 하나씩
  3. 한가한 순으로 N 개
어디서든 실패하면 0..N-1 fallback 이라 실행 자체는 항상 됨.

stdout = core 목록만 (예: "12,20,33,41"), 진단 메시지는 stderr.
사용: python scan/pick_cores.py --n 8
"""

import argparse
import os
import sys
import time


# ---- /proc/stat sampling --------------------------------------------------
def read_stat():
    """core 별 (busy, total) 누적 시간"""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            # "cpu " 총합 줄은 빼고 "cpu0", "cpu1" ... 만
            if not line.startswith("cpu") or line.startswith("cpu "):
                continue
            parts = line.split()
            cid = int(parts[0][3:])
            vals = [int(v) for v in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
            total = sum(vals)
            out[cid] = (total - idle, total)
    return out


def utilization(interval):
    """interval 초 동안의 core 별 사용률 (0~1)"""
    a = read_stat()
    time.sleep(interval)
    b = read_stat()
    util = {}
    for cid in a:
        if cid not in b:
            continue
        d_busy = b[cid][0] - a[cid][0]
        d_total = b[cid][1] - a[cid][1]
        util[cid] = (d_busy / d_total) if d_total > 0 else 1.0
    return util


# ---- core 고르기 -------------------------------------------------------------
def sibling_groups(cpus):
    """물리 core 를 공유하는 논리 CPU 묶음 (SMT). 못 읽으면 1개씩"""
    groups = {}
    for cid in cpus:
        path = f"/sys/devices/system/cpu/cpu{cid}/topology/thread_siblings_list"
        key = str(cid)
        try:
            with open(path) as f:
                key = f.read().strip()
        except OSError:
            pass
        groups.setdefault(key, []).append(cid)
    return list(groups.values())


def pick_cores(util, n):
    """한가한 물리 core 부터 n 개. 모자라면 남은 논리 CPU 로 채움"""
    ranked = []
    for g in sibling_groups(util.keys()):
        # 형제가 바쁘면 내 thread 처리량도 깎이므로 묶음 평균으로 줄 세움
        mean = sum(util[c] for c in g) / len(g)
        rep = min(g, key=lambda c: util[c])  # 대표 = 묶음에서 제일 한가한 형제
        ranked.append((mean, rep))
    ranked.sort()
    picked = sorted(rep for _, rep in ranked[:n])
    if len(picked) < n:
        # 물리 core 수보다 많이 요구한 경우
        rest = sorted(set(util) - set(picked), key=lambda c: util[c])
        picked = sorted(picked + rest[: n - len(picked)])
    return picked


# arguments (argparse)
def parse_args():
    p = argparse.ArgumentParser(description="pick the least busy cpu cores")
    p.add_argument(
        "--n",
        type=int,
        default=8,
        help="how many cores to pick",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=0.4,
        help="sampling seconds",
    )
    return p.parse_args()


def main():
    a = parse_args()
    n = max(1, a.n)
    try:
        util = utilization(a.interval)
        if not util:
            raise RuntimeError("no per-cpu stats")
        picked = pick_cores(util, n)
        busy = sum(util.values()) / len(util) * 100.0
        sys.stderr.write(
            f"[cores] machine busy {busy:.0f}% avg, picked {len(picked)} core(s)\n"
        )
        print(",".join(str(c) for c in picked))
    except Exception as e:  # noqa: BLE001 -- 이유 불문 fallback 해야 함
        # fallback 도 실제 core 수를 넘으면 taskset 이 죽음 -> 상한 clamp
        end = min(n, os.cpu_count() or n) - 1
        sys.stderr.write(f"[cores] auto-pick failed ({e}) -- fallback 0-{end}\n")
        print(f"0-{end}")


if __name__ == "__main__":
    main()
