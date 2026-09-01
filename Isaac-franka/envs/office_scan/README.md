# office-scan -- real-scan desk benchmark (4th environment)

Environment that keeps the office(env2) marker task as-is and **swaps only the
background for a real lab desk scan**. If the policy was trained on the
procedural background (office), the success-rate drop here = a direct
measurement of the **background domain gap**.

```
train(office bg) -> eval(office bg) : env2 baseline
train(office bg) -> eval(scan bg)   : ★ this env (Track A, zero-shot transfer)
train(scan bg)   -> eval(scan bg)   : Track B (after data generation -- planned)
```

Source-of-truth pipeline doc: `gr00t_isaacsim/docs/gsrecon_pipeline.md`
(see there for the phone capture -> 2DGS -> mesh -> postprocess -> USD asset
build process. Asset-build glue is `postprocess_mesh.py` etc. in the 2dgs repo)

## Files

```
office_scan_scene.py   scene module: entire env4 world definition (a single
                          definition with the scan delta inlined into the
                          office engine).
                          source of truth for protocol-final constants (top block)
preview_office_scan.py        preview (copy of preview_office, only scene swapped)
eval_office_scan.py           closed-loop eval (copy of eval_office, only scene swapped)
gen_office_scan.py            demo generation (copy of gen_office, only scene swapped)
run_eval_office_scan.sh       one-shot eval runner (auto server start/stop.
                          see header for DRY_RUN / EXTERNAL_SERVER mode)
run_gen_office_scan.sh        generation runner (Track B: same seed/batch as office,
                          outputs split into datasets/office_scan_markers)
assets/                       scan assets (take6_desk_hq.usd + .ply + json)
```

How it works: env isolation -- scene/entry all run from their own files.
The scene is a single definition with the scan delta inlined into the office
engine; the delta list vs env2 is in the scene header docstring.
Task/robot/camera/success check are fully identical to env2.
(older lineage: sys.modules overlay -> 08-31 copy isolation -> 08-31 inline)

## Protocol v1 final values

Source of truth = the constant block at the top of `office_scan_scene.py`.
The runner does not pass these values. **If you change a value, bump the
`PROTOCOL` string** (v1 -> v2).

| item | value | rationale |
|---|---|---|
| asset | `take6_desk_hq.usd` | take6 desk-centric dome scan, bounded TSDF |
| scan-scale | 1.73 | asset scale (assumes partition long axis=1.4m) -> real bench 2.4m |
| placement | auto (partition bottom line -> desk rear edge) | follows yaw sign |
| desk-color | 0.878,0.878,0.871 (sRGB) | median of clean patch on scanned desktop |
| lighting | dome 800 / key 2000 | capture lighting is baked into scan colors, so attenuated |
| holder-xy | 0.16,-0.36 | avoids baked-in prop interference (+6cm toward person) |
| region | 0.12,0.26,-0.12,0.38 | avoids baked-in keyboard + aligns with holder exclusion box |

Color convention: scan vertex colors go through an sRGB -> linear conversion
before entering the USD (replica_to_usd `--color-gamma srgb`). To match the
procedural faces to the same tone, --desk-color goes through the same
conversion.

The single JSON line `[office-scan] effective: {...}` in the scene log is
that run's protocol/asset/placement record (for output tracking).

## Run

Preview (background only / check robot and task placement):

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_dt bash -lc \
  "umask 000 && cd /root/project/Isaac-franka/envs/office_scan && \
   CUDA_VISIBLE_DEVICES=<GPU> /root/project/IsaacLab/isaaclab.sh -p \
   preview_office_scan.py --headless --robots 1 --holder 1 --items 4 \
   --out /root/project/out/office_scan_preview"
```

Eval (on the host. One-shot -- auto server start/stop):

```bash
DRY_RUN=1 CLIENT_GPU=<GPU> bash run_eval_office_scan.sh    # observation contract precheck
SERVER_GPU=<GPU> CLIENT_GPU=<GPU> EPISODES_PER_N=50 \
  CKPT=<clone_root>/checkpoints/lab_office_sim bash run_eval_office_scan.sh
```

Outputs: `<clone_root>/out/eval_office_scan/<kst_stamp>/`
(summary.json + episode videos). The reporting format is to place these side
by side with env2's `out/eval_office/` results.

## Asset lineage

```
phone video take6 -> 2d-gaussian-splatting: scan/run_scan.sh (COLMAP+training+bounded mesh)
  -> postprocess_mesh.py --pick-plane ... (align/scale/crop; sidecar json)
  -> replica_to_usd.py --up z --floor-pct -1 --no-recenter (sRGB->linear)
  -> envs/office_scan/assets/take6_desk_hq.usd  (co-located with env assets; + if
     a .json sidecar exists the scene prints it to the log)
```

When swapping in a new scan (take7...), **scan-scale recalibration is
mandatory** (the COLMAP coordinate frame is arbitrary per reconstruction)
-- see section 5 of the source-of-truth doc for the pick procedure.

## Known constraints

- scene/entry are independent of office, so office-side changes do not
  propagate automatically -- pull them in only when intended (see the lineage
  in each file's [ver])
- Track B procedure: `run_gen_office_scan.sh` (verify yield with DEMOS_OVERRIDE=3 first)
  -> run convert's office converter against NAS office_scan_markers
  -> finetune with train/run_finetune.sh -> re-eval with `run_eval_office_scan.sh` on the new ckpt
