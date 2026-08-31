# scan -- lab-capture reconstruction glue (flow stages [1][2])

Our script collection covering phone video -> COLMAP -> 2DGS training ->
TSDF mesh -> aligned asset PLY. **This directory = the boundary between
upstream (2d-gaussian-splatting) and our code** (exceptions: top-level
convert.py and utils/render_utils.py are modified upstream -- see each
file's header [ver] for the patch list).

Source of truth for knobs/pitfalls = each script's header. See the repo root
README for where this sits in the overall flow.

## Install (once, conda env surfel_splatting)

```bash
bash scan/setup/install_2dgs.sh
```

(Pins CUDA_HOME 11.8 + submodule CUDA build + fills the yml's gaps, all in
one shot. See the script header for pitfalls and ordering.)

## quickstart (from 2d-gaussian-splatting/, surfel_splatting env)

```bash
# video -> mesh one-shot (after placing the mp4 under data/raw/)
GPU=5 nohup bash scan/run_scan.sh data/raw/take7.mp4 > data/take7_run.log 2>&1 &

# always kill as a set: runner first, children (colmap/train) separately (avoids
# orphans; with ; the later commands still run even if the earlier one is already dead)
pkill -f run_scan.sh; pkill -f "colmap.*take7"; pkill -f "python.*take7"

# re-extract bounded mesh for asset building (reuses the training of the default
# unbounded run; output = fuse_post.ply. Default unbounded output is fuse_unbounded_post.ply)
GPU=5 MESH=bounded STAGE=4 DEPTH_TRUNC=15 MESH_RES=1536 SDF_TRUNC=0.1 \
    bash scan/run_scan.sh take7.mp4

# postprocess (meshlab pick 4 points -> align/scale/crop -> asset PLY)
python scan/postprocess_mesh.py --ply output/take7/train/ours_30000/fuse_post.ply \
    --out ../Isaac-franka/envs/office_scan/assets/take7_desk.ply \
    --pick-plane --scale <measured_over_pick_dist> --pick "..." --pick-box 1.2

```

Next step (Isaac side): convert to USD with
`Isaac-franka/envs/replica/replica_to_usd.py` -> consumed by the
`Isaac-franka/envs/office_scan/` benchmark (assets co-located in
`envs/office_scan/assets/`). Asset lineage = `Isaac-franka/ASSETS.md`.

When swapping in a new scan, scan-scale recalibration is mandatory (the
COLMAP coordinate frame is arbitrary per reconstruction)
-- see recipe 2 in the postprocess_mesh.py header for the pick procedure.

## Files

```
run_scan.sh          unified runner (STAGE/UNTIL resume; see header for knobs)
extract_frames.py    video -> frames (auto fps, blur warning, --merge/--gray)
pick_cores.py        shared-server courtesy: auto-picks idle CPU cores
postprocess_mesh.py  asset-build source of truth (RANSAC/pick align, crop, sidecar json)
setup/               install script (install_2dgs.sh)
```
