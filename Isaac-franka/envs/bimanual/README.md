# bimanual - dual-arm Franka benchmark

A bimanual workcell built on Isaac Lab with two Frankas, plus the generator
that produces ground-truth (seed) trajectories for the task running on it.

```
bimanual_scene.py         scene definition     - 2 robots, table, cubes
gen_bimanual.py   trajectory generator - task state machine + recording
PRESETS.md                confirmed presets    - placement numbers and their rationale
archive/                  retired              - validation scripts used for placement decisions
```

Two files are all it takes. The scene is task-independent: to build a
different task, keep `bimanual_scene.py` as is and only write a new generator.

---

## Run

```bash
docker exec -u 0 -e TERM=xterm -e PYTHONUNBUFFERED=1 gr00t_dt bash -lc \
"umask 000 && cd /root/project/Isaac-franka/envs/bimanual && \
CUDA_VISIBLE_DEVICES=2 /root/project/IsaacLab/isaaclab.sh -p gen_bimanual.py \
  --headless --demos 20 --table dual \
  --out /root/project/datasets/franka_bimanual/seed.hdf5 \
  --video-dir /root/project/out/sm"
```

Outputs go into per-run timestamped folders, so nothing gets overwritten.

```
datasets/franka_bimanual/0805_143012/seed.hdf5      successful trajectories only
                                     run_meta.json  every knob used for that run
out/sm/0805_143012/ep000_n4_success.mp4
```

For a quick check, `--demos 1 --randomize 0` with video off finishes in 1-2 minutes.

---

## Task

```
cube_1          base of the tower. left as is
other N-1       the arm whose base is closer picks each one and stacks it on top, in order
N               randomized per episode within the --num-cubes range (default 2~6)
```

Both arms pick **simultaneously** -- their assigned cubes are on opposite sides,
so paths do not overlap. There is only one stacking spot, so placing happens one
arm at a time; `Stack` coordinates this with a lock.

Success criterion: all cube xy positions gather within 3.5cm and z forms layers
of at least 2cm each. Only successful episodes are saved to HDF5.

---

## State flow (per cube)

```
REST -> ABOVE_PICK -> AT_PICK -> CLOSE -> LIFT -> HOLD
     -> ABOVE_PLACE -> AT_PLACE -> OPEN -> UP -> RETREAT -> (next cube | FINISHED)
```

| State | What it does | Why it is needed |
| --- | --- | --- |
| `HOLD` | wait outside on its own side while holding | the two arms collide at the wrists in the middle |
| `UP` | leave **vertically only** right after placing | moving sideways immediately grazes the tower just stacked |
| `RETREAT` | back off to its own side and release the lock | stopping right above the tower blocks the next arm's approach |

---

## Four points that matter in the design

**(1) Tool offset** -- what IK controls is the `panda_hand` origin; the actual
grasp point (TCP) is `--tool-offset` (0.1034m) further along the hand +z. Unless
the target is pulled back by that amount, the hand digs into the table.

**(2) Grasp point frozen** -- the grasp target is computed only once in `REST`.
If the cube is re-read every step, then after grasping the cube follows the
hand, so the target runs away with the hand and `LIFT` never converges.

**(3) Measured heights** -- the place height is not assumed as "floor + layer
x 4cm". Measure the `TCP - cube center` gap at grasp time (`hold_off`), and read
the tower's actual current height via `Stack.top_z()` when placing. Errors that
arise as layers stack up do not accumulate.

**(4) No command-less intervals** -- calling `compute` on
`DifferentialIKController` without `set_command` takes the target as 0 and drags
the arm toward the root origin. During wait intervals, `hold_here()` must set
the current pose as the target.

---

## Key knobs

| Knob | Default | Symptom -> action |
| --- | --- | --- |
| `--num-cubes` | `2,6` | count range |
| `--region` | `0.36,0.60,-0.26,0.26` | region where cubes are scattered. Widen if sampling fails often |
| `--min-sep` / `--base-clear` | `0.11` / `0.13` | min distance between cubes / to the tower. Increase if neighboring cubes get bumped |
| `--yaw-range` | `45` | cube rotation span. Reduce if grasps fail often |
| `--place-clear` | `0.005` | clearance above the tower top. Increase if it presses down |
| `--pos-tol` / `--dwell` | `0.006` / `25` | arrival check. Relax if transitions stall, tighten if unstable |
| `--state-timeout` | `250` | forced advance when blocked by contact. Visible as `[!]` in the log |
| `--max-steps` | `1400` | cap **per cube** (actual cap = this value x number of cubes to stack) |

For scene-side knobs see `PRESETS.md`. Confirmed values: `--table dual
--mirror-side right --base-sep 0.9`.

---

## Recording format

```
data/demo_i/
  actions   (T,16)  [Lpos3 Lquat4 Lgrip1 | Rpos3 Rquat4 Rgrip1]
                    absolute TCP targets in world frame (IK-Abs family)
  obs/      joint_pos(T,18)  eef_pos_l/r  eef_quat_l/r  grip_l/r
            cube_pos(T,3N)   cube_quat(T,4N)
  subtask/  place_done (T,)  cumulative count, +1 each time a cube is stacked
```

`subtask/place_done` exists to later serve as annotation boundaries for
Isaac Lab Mimic. The state machine knows those moments exactly, so no human
re-annotation is needed.

Actions are recorded as **absolute poses (IK-Abs)** because Isaac Lab's official
state-machine examples and the GR1T2 Mimic env are both Abs, and there is a
report of IK-Rel-made trajectories getting stuck in Mimic
([Discussion #4006](https://github.com/isaac-sim/IsaacLab/discussions/4006)).

---

## Runtime pitfalls

See the table in the parent folder's `../../README.md`. In short:

```
-e TERM=xterm            without it, login shell init dies
-e PYTHONUNBUFFERED=1    without it, our prints stay invisible until the end
umask 000 &&             without it, outputs are root-owned and cannot be deleted from the host
```
