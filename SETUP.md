# Setup

Two host conda environments plus one Isaac Sim container. This gets a machine
to the point where the environments build and render (stages 1-7 of the
[README](README.md)); the later stages take storage paths and a checkpoint as
runner arguments.

| Where | Name | Used by |
| --- | --- | --- |
| host | `surfel_splatting` | stage 1-5 (scan -> mesh -> asset PLY) |
| host | `gr00t_sh` | stage 9-11 (LeRobot conversion, fine-tuning, policy server) |
| container | `nvcr.io/nvidia/isaac-sim:4.5.0` | stage 6-8, 11 (everything that opens Isaac Sim) |

Verified on Ubuntu 22.04, NVIDIA driver 545.23.06, CUDA 11.8 (for the 2DGS
submodule build), Docker with the NVIDIA container runtime.

Host tools the pipeline shells out to: **COLMAP** (stage 2; a CUDA build is much
faster, and `run_scan.sh` falls back to CPU without one), **MeshLab** (stage 5.2
-- the four alignment points for `--pick` are read off there), and **netcat**
(`nc`, stage 11 -- the eval runner polls the policy server port with it; without
`nc` the wait silently times out instead of erroring).

## 1. Clone

```bash
# the clone root is what the container mounts at /root/project -- keep it stable
git clone <this-repo> project-DT && cd project-DT
```

## 2. Host env: surfel_splatting

```bash
# creates the conda env and builds the 2DGS submodules (needs CUDA 11.8;
# override the location with CUDA_HOME=...). The upstream copy is vendored,
# so no extra clone is needed
cd 2d-gaussian-splatting && bash scan/setup/install_2dgs.sh && cd ..
```

## 3. Host env: gr00t_sh

System prerequisites (upstream): `ffmpeg`, `libsm6`, `libxext6`. CUDA 12.4 is
the tested version; 11.8 also works.

```bash
conda create -n gr00t_sh python=3.10 -y && conda activate gr00t_sh
pip install --upgrade setuptools

# editable install FROM THIS CLONE -- that is what makes the vendored patches
# and our_configs.py the ones that get imported
pip install -e ./Isaac-GR00T[base]

# on CUDA 11.8 use flash-attn==2.8.2 instead
pip install --no-build-isolation flash-attn==2.7.1.post4

# LeRobot conversion (stage 9) needs these on top of GR00T's own deps
pip install h5py pandas pyarrow imageio imageio-ffmpeg

# must print this clone's Isaac-GR00T path
python -c "import gr00t, os; print(os.path.dirname(os.path.dirname(gr00t.__file__)))"
```

```bash
# only if you move or re-clone the repo: an editable install keeps pointing at
# the old location and silently imports the old code
pip install -e ./Isaac-GR00T --no-deps
```

## 4. Container

```bash
# --network host is required: the eval client runs inside the container and
# reaches the policy server on the host over localhost:<PORT>
docker run -d --name gr00t_dt \
  --network host --gpus all --ipc host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v <clone-root>:/root/project \
  -v <cache-dir>/ov-cache:/root/.cache/ov \
  -v <cache-dir>/ov-local:/root/.local/share/ov \
  -v <cache-dir>/omniverse:/root/.nvidia-omniverse \
  --entrypoint /bin/bash \
  nvcr.io/nvidia/isaac-sim:4.5.0 -c "sleep infinity"
```

Then set up Isaac Lab inside it. Every step installs into the kit interpreter
at `/isaac-sim/kit/python/bin/python3`, which is what `python.sh` uses.

```bash
# (a) git -- the Isaac Lab installer fetches two deps (rl-games, robomimic) over git
docker exec gr00t_dt bash -lc "apt-get update -qq && apt-get install -y -qq git"

# (b) setuptools < 81 and a pre-built flatdict -- see pitfall 2
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install "setuptools<81"
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install \
    --no-build-isolation flatdict==4.0.1

# (c) link Isaac Sim into the Isaac Lab tree, then install the extensions
docker exec -e TERM=xterm gr00t_dt bash -lc \
    "ln -sfn /isaac-sim /root/project/IsaacLab/_isaac_sim && \
     cd /root/project/IsaacLab && ./isaaclab.sh -i"

# (d) pin torch to a build your driver can run -- see pitfall 3
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps

# (e) pyzmq -- the eval client talks to the policy server over ZMQ
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -m pip install pyzmq
```

```bash
# expected: an Isaac Lab version, a +cu12x torch build, and True
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -c \
    "import isaaclab, torch; print(isaaclab.__version__, torch.__version__, torch.cuda.is_available())"

# freezing the working container saves repeating all of the above
docker commit gr00t_dt gr00t_dt_img:stable
```

> Assets stream from NVIDIA's content server by default (upstream Isaac Lab
> behaviour). One exception: the office prop stubs committed here
> (`Isaac-franka/envs/office/props/*.usd`) reference `/isaac-assets/...`
> absolute paths because they were generated against a local asset mirror.
> Either mount a copy of the Isaac Sim 4.5 asset pack there
> (`-v <assets>/Assets/Isaac/4.5:/isaac-assets`), or regenerate the stubs for
> your own asset root with `Isaac-franka/envs/office/extract_props.py`.

## 5. Verify

```bash
# real scene and observation contract, no policy server or checkpoint needed.
# passes when it prints [office-scan] effective: {...office-scan-v1...}
# followed by the payload table
CONTAINER=gr00t_dt DRY_RUN=1 CLIENT_GPU=<n> \
    bash Isaac-franka/envs/office_scan/run_eval_office_scan.sh

# the dry run skips the policy client, so check pyzmq separately
docker exec gr00t_dt /isaac-sim/kit/python/bin/python3 -c "import zmq"
```

## Setup pitfalls

All four were hit while rebuilding this container from scratch. The steps above
already avoid them; this is for when one still fails.

| Symptom | Cause and fix |
| --- | --- |
| `isaaclab.sh -i` stops at `Cannot find command 'git'` | The base image ships no git, and two extension groups pull packages from git URLs. Step (a). |
| `isaaclab.sh -i` reports success, but later `ModuleNotFoundError: No module named 'isaaclab'` | The core extension failed silently while the others succeeded: pip builds `flatdict` in an isolated env that pulls the newest setuptools, and setuptools 81 dropped `pkg_resources`, which `flatdict`'s `setup.py` imports. Step (b), then `pip install -e source/isaaclab` inside `IsaacLab/`. |
| `RuntimeError: The NVIDIA driver on your system is too old` | The installer pulls the newest torch, currently a CUDA 13 build needing a much newer driver. A CUDA 12.x build runs on any 12.x driver. Step (d) -- check `nvidia-smi` and pick a matching build. |
| Eval client dies with `ModuleNotFoundError: No module named 'zmq'` | `isaaclab.sh -i` does not install pyzmq. Step (e). |

```bash
# to find anything else missing relative to a container that already works
docker exec <working> /isaac-sim/kit/python/bin/python3 -m pip list --format=freeze | cut -d= -f1 | sort > /tmp/a
docker exec <new>     /isaac-sim/kit/python/bin/python3 -m pip list --format=freeze | cut -d= -f1 | sort > /tmp/b
comm -23 /tmp/a /tmp/b
```

## Runtime pitfalls

Container invocation flags (`TERM`, `PYTHONUNBUFFERED`, `umask`) and the rest
are in [Isaac-franka/README.md](Isaac-franka/README.md).

One that costs a full run if missed -- the policy server changes directory
before loading the model, so a relative `CKPT` resolves against the wrong root
and the loader then treats the string as a Hugging Face repo id:

```bash
CKPT=$PWD/checkpoints/<name>    # not checkpoints/<name>
```
