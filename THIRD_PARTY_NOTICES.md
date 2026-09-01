# Third-party notices

This repository vendors three upstream projects in full. Each keeps its own
license; nothing here relicenses them. Our own code is MIT (see `LICENSE`).

**Effective terms of the repository as distributed: research and evaluation
use only (non-commercial)**, because the 2d-gaussian-splatting component is
under a non-commercial research license and does not permit sublicensing.

| Directory | License | Full text |
| --- | --- | --- |
| `2d-gaussian-splatting/` | Gaussian-Splatting License (Inria + MPII) -- **non-commercial, research only** | `2d-gaussian-splatting/LICENSE.md` |
| `Isaac-GR00T/` | Apache License 2.0 (NVIDIA CORPORATION & AFFILIATES) | `Isaac-GR00T/LICENSE` |
| `IsaacLab/` | BSD-3-Clause (Isaac Lab Project Developers); the `isaaclab_mimic` extension is Apache-2.0 | `IsaacLab/LICENSE`, `IsaacLab/LICENSE-mimic` |

## 2d-gaussian-splatting

Vendored copy of 2D Gaussian Splatting (ShanghaiTech), which builds on the
Inria/MPII gaussian-splatting work and inherits its license
(`2d-gaussian-splatting/LICENSE.md`; the same license is repeated for the two
submodules). It grants use for research purposes only, "without right to
sublicense", and section 4.1 allows redistribution only under that same
license with the license text and all attribution notices retained.

The tree also carries files under other third-party licenses, kept as-is:

| File | License |
| --- | --- |
| `utils/render_utils.py` | Apache-2.0 (Copyright 2022 Google LLC) |
| `utils/sh_utils.py` | BSD-style (Copyright 2021 The PlenOctree Authors) |
| `submodules/diff-surfel-rasterization/third_party/glm/` | Happy Bunny License or MIT (G-Truc Creation) |
| `submodules/diff-surfel-rasterization/third_party/stbi_image_write.h` | Sean Barrett |

Two upstream files are modified here and carry our edit stamp below the
original header, remaining under their own upstream licenses: `convert.py`
(Inria) and `utils/render_utils.py` (Google, Apache-2.0).

`2d-gaussian-splatting/scan/` is separate code written by this project -- it
drives the pipeline but derives from no upstream source -- and is MIT under
`LICENSE`. Running it still requires the non-commercial component above.

Section 4.4 of the Gaussian-Splatting License strongly encourages citing the
upstream publications when the pipeline is used for published results.

## Isaac-GR00T

Apache-2.0. Change notice per section 4(b): `scripts/gr00t_finetune.py` in this
copy is **modified** relative to upstream 1.1.0 -- the checkpoint retention
limit was lowered and a hard-coded CUDA device override was made overridable.
Both edits are described in `Isaac-franka/train/run_finetune.sh`.

`our_configs.py` at the tree root is a file added by this project (MIT;
canonical copy `Isaac-franka/train/our_configs.py`, from which it is copied).

GR00T model weights are **not** distributed in this repository. NVIDIA's model
license applies to the checkpoints separately.

## IsaacLab

BSD-3-Clause, with `isaaclab_mimic` under Apache-2.0. Per BSD-3 clause 3,
neither the Isaac Lab name nor its contributors' names are used to endorse this
project. Per-asset and per-dependency license files live under
`IsaacLab/docs/licenses/` and are kept intact.

## Isaac Sim assets

The office prop USD files under `Isaac-franka/envs/office/props/` are
reference-only stubs (about 1 KB each): they point at prims in a local Isaac Sim
asset installation (or NVIDIA's content server) and carry no upstream geometry
or textures. No NVIDIA Omniverse
asset files are redistributed; resolving these props requires a local Isaac
Sim asset install. (Rendered imagery under `figures/` shows the stock scene.)

## Scanned desk asset

`Isaac-franka/envs/office_scan/assets/take6_desk_hq.{ply,usd}` is our own
capture of our own lab desk, produced through the 2DGS pipeline above. It is
covered by our MIT license, but note that reproducing it requires the
non-commercial 2DGS component. Provenance chain: `Isaac-franka/ASSETS.md`.
