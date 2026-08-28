#!/usr/bin/env python
# ======================================
# File: pick_cores.py
# ======================================
# Sanghyeok Park, SSL undergraduate
# Edit 2026-08-28
# ======================================
# [ver] pick_cores.py 2026-08-20-r1  (ascii-only console/comments)
"""지금 가장 한가한 CPU 코어 N 개를 골라 taskset 형식으로 출력한다.

공용 서버에서 0-7 같은 고정 대역을 쓰면 하필 그 코어가 붐빌 때 스케줄러가
우리를 다른 코어로 옮겨줄 수 없다. 여기서는
  (1) /proc/stat 을 두 번 샘플링해 코어별 사용률을 재고
  (2) SMT 형제(같은 물리코어)를 피해 물리코어 하나당 하나씩만 뽑아
  (3) 한가한 순으로 N 개를 고른다.
무엇이든 실패하면 0..N-1 로 폴백하므로 실행은 항상 성립한다.

stdout = 코어 목록만 (예: "12,20,33,41"), 진단은 stderr.
사용: python scan/pick_cores.py --n 8
"""

import argparse
import os
import sys
import time


def read_stat():
    """cpu 별 (busy, total) 누적 시간."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
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


def sibling_groups(cpus):
    """같은 물리코어를 공유하는 논리 CPU 묶음 (SMT). 실패 시 1개씩."""
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


def main():
    p = argparse.ArgumentParser(description="pick the least busy cpu cores")
    p.add_argument("--n", type=int, default=8, help="how many cores to pick")
    p.add_argument("--interval", type=float, default=0.4, help="sampling seconds")
    a = p.parse_args()

    n = max(1, a.n)
    try:
        util = utilization(a.interval)
        if not util:
            raise RuntimeError("no per-cpu stats")
        groups = sibling_groups(util.keys())
        # 물리코어 = 형제 묶음. 묶음의 평균 사용률이 낮을수록 한가하다
        # (형제가 바쁘면 우리 스레드의 실효 처리량도 깎인다).
        ranked = []
        for g in groups:
            mean = sum(util[c] for c in g) / len(g)
            rep = min(g, key=lambda c: util[c])  # 묶음 대표 = 가장 한가한 형제
            ranked.append((mean, rep))
        ranked.sort()
        picked = sorted(rep for _, rep in ranked[:n])
        if len(picked) < n:
            # 물리코어보다 많이 요구한 경우: 남은 논리 CPU 로 채운다
            rest = sorted(set(util) - set(picked), key=lambda c: util[c])
            picked = sorted(picked + rest[: n - len(picked)])
        busy = sum(util.values()) / len(util) * 100.0
        sys.stderr.write(
            f"[cores] machine busy {busy:.0f}% avg, picked {len(picked)} core(s)\n"
        )
        print(",".join(str(c) for c in picked))
    except Exception as e:  # noqa: BLE001 -- 어떤 이유든 폴백해야 한다
        # 폴백도 머신 코어 수를 넘으면 taskset 이 죽는다 -> 상한 클램프
        end = min(n, os.cpu_count() or n) - 1
        sys.stderr.write(f"[cores] auto-pick failed ({e}) -- fallback 0-{end}\n")
        print(f"0-{end}")


if __name__ == "__main__":
    main()
