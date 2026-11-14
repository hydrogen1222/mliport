# DPA (DeepMD-kit)

DPA checkpoints are loaded through `deepmd-kit`. Many of them are multi-task
models with several named branches, and mliport will not guess which branch you
mean: you pass it explicitly as the head, or the run fails closed.

## What the model is

- **Architecture.** DPA-3 is a message-passing network on a *line graph series*:
  the first graph holds atoms and pairs, the next holds bond angles, and so on,
  so angular information is built in explicitly
  ([Zhang et al., 2025](https://arxiv.org/abs/2506.01686)).
- **Training data of the example checkpoint.** `DPA-3.1-3M` (about 3.3 million
  parameters) is pretrained on OpenLAM-v1, a collection of 31 datasets
  from different domains, computed with *different* DFT settings. The model keeps them apart with a dataset encoding,
  which is what you select as the **head**: `Omat24` reproduces the OMat24
  reference level (PBE / PBE+U, bulk inorganic, see the [MACE page](mace.md)).
- **Forces** are the gradient of the summed atomic energies (conservative).
- **What this means for you.** Pick the head whose DFT settings match the
  calculations you will compare with, and never compare absolute energies from
  two different heads: each dataset has its own energy reference and possibly its
  own functional. List the heads of a checkpoint with `dp --pt show ...
  model-branch` (below).

## Install

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit @ git+https://github.com/deepmodeling/deepmd-kit.git@0edeaef5617e9878edeb2346bf37bb6a035a4f51` (torch 2.11.0) |

```bash
./scripts/install_mliport.sh --engines dpa --device auto
```
<!-- END GENERATED -->

The installer pins upstream commit `0edeaef5617e9878edeb2346bf37bb6a035a4f51`
(package version `3.2.1.dev63+g0edeaef56`) and PyTorch 2.11.0. The released
3.2.0 wheel loads DPA4-Neo but rejects the `gaussian/fix` basis in these DPA4C
checkpoints. The pinned source contains the upstream implementation of this
basis; mliport does not rewrite checkpoint settings or weights.

This route needs Git and a C++ build toolchain. With CUDA PyTorch, configure
the CUDA Toolkit before installing (set `CUDA_HOME` or `CUDA_PATH` and put
`nvcc` on `PATH`). The installer forwards that explicit toolkit location to
CMake, builds against the selected PyTorch, and disables TensorFlow in this
environment. V100 uses the `2.11.0+cu126` wheel with `sm_70` kernels. Runtime
dependencies include e3nn, vesin/vesin-torch and mpich.

Package mirrors still supply dependencies, but the pinned source requires
GitHub access. Offline installation takes a prebuilt wheel of the exact pinned
version from the local `packages/` directory; it never contacts GitHub.

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `dpa_omat` | DPA-3.1-3M, branch Omat24 | upstream/model-defined | bulk | Omat24 | `86dd3a804d78...` |
<!-- END GENERATED -->

## Run it

```bash
.venv-dpa/bin/mliport sp structure.cif \
  --model /path/to/DPA-3.1-3M.pt --model-type dpa --head Omat24
```

To see which branches a checkpoint offers, ask DeePMD-kit, then pass the one you
want as `--head` (`HEAD` in an INCAR file):

```bash
dp --pt show DPA-3.1-3M.pt model-branch
```

Single-task DPA4-Neo, DPA4C-Nano, DPA4C-Mini and DPA-4.1C-Air checkpoints
use the same entry point without `--head`, for example:

```bash
bin/mliport-dpa sp structure.cif --model-type dpa --task bulk --device cuda:0 \
  --model models/DPA4C-Nano-OMat24.pt
```

DeePMD selects its regular PyTorch or PyTorch-exportable evaluator from the
checkpoint's parameter dialect. mliport records the evaluator, actual GPU
identity and parameter dtype. It checks loaded `.pt` tensors against the
selected checkpoint head and rejects partial loads or changed learned tensors.
The pinned upstream's rebuilding of DPA4C integer gather caches and its
unbridged non-spin descriptor-version migration are recorded separately in
`checkpoint_migrations`.

## Trusted training checkpoint compatibility

Some DPA-SSE training checkpoints include NumPy scalars in both optimizer
metadata and model training information. DeePMD explicitly uses
`torch.load(weights_only=True)`, so these checkpoints cannot be loaded
directly. For a **locally verified, trusted** source file only, convert it
without modifying the original:

```bash
.venv-dpa/bin/python scripts/convert_dpa_checkpoint.py \
  models/SSE-VASP-dpa2.pt models/SSE-VASP-dpa2-inference.pt \
  --i-trust-this-file
```

The converter runs unsafe pickle deserialization **only after explicit opt-in**;
never use it for a checkpoint from an unknown or untrusted source. It drops
optimizer state, normalizes NumPy scalars inside the model metadata and
verifies that every tensor, dtype and value is unchanged when reloaded with
`weights_only=True`. A separate `.conversion.json` preserves both SHA-256
hashes, conversions and the original source path. This verifies *file
compatibility*, not energy/force equivalence: run real DeePMD predictions with
the converted model and compare against the original trusted reference before
scientific use.

The July 2024 `SSE-VASP-dpa2.pt` also has network-layout differences with this
runtime (for example a `(800, 128)` matrix where the current network expects
`(640, 128)`). Converting NumPy metadata does not repair that mismatch.
It remains unsupported until a matching legacy implementation is validated;
mliport does not reshape weights or initialize missing learned parameters.

## Things to know

- **Leaving out the head on a multi-task checkpoint is a fatal configuration
  error.** There is no silent default branch.
- A failed model-metadata check leaves no cached calculator. Retrying the same
  wrapper loads and validates the model again before exposing it.
- **Precision** is defined by the model. mliport records the dtype the backend
  actually reports and never substitutes its own idea of it.
- **Stress** is reported only if the checkpoint and branch you loaded expose it.
- **DPA needs per-process GPU isolation.** The CLI restarts a command with
  `CUDA_VISIBLE_DEVICES` when it is unset; from the Python API, set the variable
  in the calling process before importing the framework.
- DPA has no `inference_mode`. UMA-only options are rejected under strict
  configuration rather than ignored.

mliport is MIT-licensed. The upstream software and model must be cited
separately; see [Citations](../citations.md).
