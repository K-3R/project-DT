# Isaac-franka -- Isaac Lab Franka x GR00T benchmark collection

Index for the track that builds 4 bimanual Franka environments on Isaac Lab
and does data generation/training/evaluation with GR00T. Each environment is
isolated in its own subdirectory; details follow its own README / source-of-truth doc.

## Environment matrix

| env | directory | scene | task | source-of-truth doc |
| --- | --- | --- | --- | --- |
| 1 | `envs/bimanual/` | table + bimanual rig | cube tower stacking | `docs/bimanual_training_pipeline.md` |
| 2 | `envs/office/` | procedural office desk (measured dimensions) | marker -> pencil holder | `docs/office_env_pipeline.md` |
| 3 | `envs/replica/` | Replica scanned room (background only) | (preview stage) | `docs/replica_background.md` |
| 4 | `envs/office_scan/` | ★photoreal scanned desk (env2 scene swap) | same as env2 | `docs/gsrecon_pipeline.md` + `envs/office_scan/README.md` |

env4 reuses env2's task/robot/cameras as-is with only the background swapped
for a real scan, so the success-rate difference vs env2 = a measurement of the
**background domain gap**.

Isolation principle: `envs/office/` is the unmodified engine; `envs/office_scan/`
is layered on top via a sys.modules swap. `envs/replica/` is fully isolated and
does not import office.

Runner naming rule: `run_{verb}_{env}.sh` (gen/eval/convert x bimanual/office/
office_scan). The "unmarked runner = a specific env" convention has been abolished.

## Directory layout

```
envs/               4 environments + groot_client.py (shared ZMQ client that
                    env scripts import from the parent directory)
train/              finetuning + inference server. run_server_finetuned.sh is
                    shared by env1/2/4; select the environment via CKPT
convert/            HDF5 -> LeRobot conversion (run_convert_bimanual.sh /
                    run_convert_office.sh -- office_scan data also uses the
                    office converter as-is)
kill.sh             stop runs (processes inside the container must be killed
                    from inside)
ASSETS.md           scan asset ledger (take -> ply -> usd -> scale lineage)
```

### Container paths

`/root/project` is a bind mount of the **clone root of this repo** on the host
(each person creates their own container and mounts their own clone -- see setting.sh).
Files edited on the host are reflected immediately (no `docker cp` needed).

| | host | container |
| --- | --- | --- |
| project root | `~/project/project-DT` (clone location) | `/root/project` |
| this folder | `.../Isaac-franka` | `/root/project/Isaac-franka` |
| Isaac Lab | `.../IsaacLab` | `/root/project/IsaacLab` |
| generated data | `.../datasets` (auto-created on run) | `/root/project/datasets` |

> `python.sh -c "import isaaclab"` failing with `ModuleNotFoundError: omni.physics`
> is normal -- it only loads after SimulationApp(AppLauncher).

> If a shell script was edited on Windows, its line endings may be CRLF.
> Convert with `crlf2lf.sh` at the project root.

### Runtime pitfalls (common to all environments, all field-verified)

| symptom | cause / fix |
| --- | --- |
| immediate exit with `'ansi+tabs': unknown terminal type` | `docker exec` has no TTY, so login-shell init dies -> `-e TERM=xterm` |
| none of our `print`s show up | stdout block buffering -> `-e PYTHONUNBUFFERED=1` (Kit logs go to stderr, so they keep flowing) |
| cannot delete outputs from the host | root-owned because of `-u 0` -> prefix the command with `umask 000 &&`. For existing files: `docker exec -u 0 ... chmod -R 777` |
| process does not exit after `[done]` | `simulation_app.close()` does not return in headless -> the script force-exits with `os._exit(0)` |
| `OgnSdOnNewFrame: frames discarded` | camera capture is faster than rendering. Only video frames are dropped; no effect on the numbers |
| USD `FileNotFoundError` | `ISAAC_NUCLEUS_DIR` = `/isaac-assets/Isaac` local mirror. Tables are under `Props/Mounts/` |
| S3 download hits a TLS error on the server network | `omniverse-content-production.s3...` is blocked -> download locally and copy over |
| process inside the container does not die from a host kill | docker exec processes must be killed from inside: `docker exec -u 0 <container> pkill -f <name>`. Do not touch the container itself |
