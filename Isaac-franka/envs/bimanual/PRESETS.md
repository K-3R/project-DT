# Bimanual Franka environment - confirmed presets (2026-08-05)

Isaac Lab 2.2.1 has no bimanual robot asset (the only dual-arm option is the
GR1T2 humanoid). **Two Frankas are placed in one scene to form a bimanual
workcell** -- the same setup robosuite's TwoArm envs use with two Pandas.

Presets A / C are confirmed as the environment baselines. Seed trajectories are
built and tasks are run on top of them.

---

## Preset A - two Isaac tables

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_dt bash -lc \
"umask 000 && cd /root/project/Isaac-franka/envs/bimanual && \
CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p archive/dual_franka_scene.py \
  --headless --steps 60 \
  --table dual --table-usd SeattleLabTable \
  --mirror-side left --table-mirror 2 \
  --layout parallel --base-sep 0.9 --table-dx 0.55 \
  --shot /root/project/out/A/view.png --diagram /root/project/out/A/layout.png"
```

Measured results

```
TableL  x[-0.315,+1.104] 1.419m   y[-0.750,+0.910] 1.660m
TableR  x[-0.315,+1.104] 1.419m   y[-0.910,+0.750] 1.660m
combined surface y[-0.910,+0.910]  center 0  (symmetric, overlapping)
```

## Preset C - procedural table

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_dt bash -lc \
"umask 000 && cd /root/project/Isaac-franka/envs/bimanual && \
CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p archive/dual_franka_scene.py \
  --headless --steps 60 \
  --table proc --table-size 1.4,1.8,0.05 --table-height 1.05 --table-dx 0.55 \
  --layout parallel --base-sep 0.9 \
  --shot /root/project/out/C/view.png --diagram /root/project/out/C/layout.png"
```

The origin is the tabletop center itself, so the mirroring and mount-plate
problems structurally do not exist. Size can be chosen freely to fit the task,
making it **the right default for new benchmarks**.

---

## Common to both presets (robots, objects, workspace)

```
robots       FRANKA_PANDA_HIGH_PD_CFG x2   dof=9, bodies=11, fixed_base=True
             RobotL (0, +0.45, 0)   RobotR (0, -0.45, 0)   both facing +x
work surface env frame z=0,  ground z=-1.05
cubes        Props/Blocks/{blue,red,green}_block.usd, 4cm
             cube_1 (0.45,  0.00, 0.0203)   center  <- shared by both arms
             cube_2 (0.45, +0.18, 0.0203)   left
             cube_3 (0.45, -0.18, 0.0203)   right

Franka reach 0.855 m   base separation 0.900 m   shared workspace width 0.810 m
  cube_1  L 0.637(O)  R 0.637(O)     <- handover feasible
  cube_2  L 0.525(O)  R 0.774(O)
  cube_3  L 0.774(O)  R 0.525(O)
```

The `base_sep` upper bound is 2x0.855 = 1.71 m. Beyond that the reach spheres
do not intersect and coordinated tasks become impossible at all. 0.9 gives an
overlap of 0.81 m, with margin.

---

## Pitfalls of this asset (Preset A only)

> Caution: the measurements/conclusions in this section date from
> `archive/dual_franka_scene.py` (the old scene). In the current
> `bimanual_scene.py` the default `--mirror-side right` is the render-verified
> value and the main generation run (704 demos) used this default -- when using
> the dual table with the current scene, the code defaults are the source of truth.

The **origin of `SeattleLabTable` is not the slab center; it is shifted toward
the mount side.** Measured: origin -> slab body center offset `(-0.156, +0.370)`
(world frame after applying rot Rz(90deg)).

So placing two side by side naively shifts the slabs to one side and **the far
leg sticks out on its own**.

- **No z rotation can fix it.** No angle achieves "keep x, flip only y"
  (Rz(90deg)->(-b,a), Rz(-90deg)->(b,-a), Rz(180deg)->(-a,-b) -- none is the desired (+0.394,-0.370)).
- The only way is a **mirror reflection**: `scale=(1,-1,1)` + conjugated `rot` (Rz(theta)->Rz(-theta)).
  That is `--table-mirror 2`.
- **The left one must be mirrored** so the two slabs converge inward and overlap (`--mirror-side left`).
  Mirroring the right one spreads them outward, growing the combined width 1.82 -> 3.30 m and leaving a seam in the middle.
- Negative scale flips normals, so rendering can come out dark. Preset C does not have this problem.

---

## Next step

Build seed trajectories with a state machine (teleop needs a GUI, so it is
impossible headless).

```
Franka actions are already eef pose delta + 1-axis gripper, so a
trajectory is just "a list of eef waypoints". No joint-space design needed.

Bimanual adds a second state machine + synchronization points.
Task candidate: handover (L brings it to the center -> R takes it and places it on its own side)
```
