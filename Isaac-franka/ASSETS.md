# Scan asset ledger

Records asset lineage (take -> mesh -> postprocess -> USD -> consumers).
When adding a new asset, one row here + the postprocess sidecar json
(auto-generated) is the canonical evidence. **Scale is arbitrary per
reconstruction**, so swapping an asset = recalibration.

Location: `take6_desk_hq.usd` (+.ply, sidecar json) = the repo's
`envs/office_scan/assets/` (git-tracked -- this is the source of truth; the
scene default reads this path). `take6_desk_final.usd` (backup edition) and
the `replica/` assets are not in the repo -- they exist only in the server
worktree's `datasets/` (gitignored).

## Current (protocol office-scan-v1)

| asset | source | production chain | asset scale | spawn correction | consumers |
| --- | --- | --- | --- | --- | --- |
| `take6_desk_hq.usd` | take6.mp4 (desk-centered dome scan, 1080p) | bounded TSDF (DEPTH_TRUNC=15, MESH_RES=1536, SDF_TRUNC=0.1) -> postprocess pick-plane (scale 0.1209) -> replica_to_usd (sRGB->linear) | 1 unit = 0.1209m assumed (partition long axis = 1.4m) | x1.73 (real bench 2.4m) | env4 office_scan (default asset) |
| `take6_desk_final.usd` | take6.mp4 | same chain, based on the unbounded mesh | same | x1.73 | env4 backup (pre-hq edition) |
| `replica/office_0.usd` | Replica office_0 mesh.ply | replica_to_usd (z-up, floor re-origin) | native meters | none | env3 replica |

Caution: .usd files converted before 08-25 are the no-gamma (linear
passthrough) edition and the colors differ -- reconverting with
replica_to_usd r2 matches the current color convention.

## Frame convention (take6 family)

```
origin = tabletop center, z0 = tabletop surface, partition = +y (base line y=+0.35 asset-meters)
human/robot side = -y. On spawn, a z+90deg rotation aligns to the office frame (-x = back).
partition height above the tabletop = 0.374 asset-meters (x1.73 = real 0.65m)
```

## Rebuild procedure

Source of truth = `docs/gsrecon_pipeline.md` (capture sec. 2, runners sec. 3,
postprocess sec. 5). A new take requires the 4-point pick calibration (sec. 5)
and re-deriving the spawn scale -- updating the SCAN_* constants in
office_scan_scene.py = a protocol version bump.
