# 스캔 자산 대장 (datasets/)

자산의 계보 (take -> 메시 -> 후처리 -> USD -> 소비처)를 기록한다.
새 자산 추가 시 여기에 한 행 + postprocess 사이드카 json (자동 생성)이
정본 근거다. **스케일은 재구성마다 임의**이므로 자산 교체 = 재캘리브레이션.

## 현행 (프로토콜 office-scan-v1)

| 자산 | 원본 | 제작 체인 | 자산 스케일 | 스폰 보정 | 소비처 |
| --- | --- | --- | --- | --- | --- |
| `take6_desk_hq.usd` | take6.mp4 (책상 중심 돔 스캔, 1080p) | bounded TSDF (DEPTH_TRUNC=15, MESH_RES=1536, SDF_TRUNC=0.1) -> postprocess pick-plane (scale 0.1209) -> replica_to_usd (sRGB->linear) | 1 유닛 = 0.1209m 가정 (칸막이 장축 = 1.4m) | x1.73 (실물 벤치 2.4m) | env4 office_scan (기본 자산) |
| `take6_desk_final.usd` | take6.mp4 | 동일 체인, unbounded 메시 기반 | 동일 | x1.73 | env4 예비 (hq 이전 판) |
| `replica/office_0.usd` | Replica office_0 mesh.ply | replica_to_usd (z-up, 바닥 재원점) | 원본 미터 | 없음 | env3 replica |

주의: 08-25 이전에 변환된 .usd 는 감마 무변환(linear passthrough) 판이라
색이 다르다 -- replica_to_usd r2 로 재변환하면 현행 색 규약과 일치.

## 프레임 규약 (take6 계열)

```
원점 = 상판 중앙, z0 = 상판면, 칸막이 = +y (밑선 y=+0.35 자산미터)
사람/로봇 쪽 = -y. 스폰 시 z+90도 회전으로 office 프레임(-x = 뒤)에 정합.
상판 위 칸막이 높이 = 0.374 자산미터 (x1.73 = 실물 0.65m)
```

## 재제작 절차

정본 = `docs/gsrecon_pipeline.md` (촬영 2절, 러너 3절, 후처리 5절).
새 take 는 픽 4점 캘리브레이션(5절)과 스폰 스케일 재산출이 필수다 --
office_scan_scene.py 의 SCAN_* 상수 갱신 = 프로토콜 버전업.
