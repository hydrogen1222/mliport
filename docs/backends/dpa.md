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

## DPA-SSE and the DPA-2 legacy runtime

`SSE-VASP-dpa2.pt` from AIS Square
([model page](https://www.aissquare.com/models/detail?pageType=models&name=SSE-VASP-dpa2))
is a DPA-2 checkpoint trained on sulfide electrolytes: 15 elements, 41 systems,
26 of them chalcogenides, labelled with VASP PAW / **PBEsol**. It is the model
of the DPA-SSE paper
([arXiv 2406.18263](https://arxiv.org/abs/2406.18263),
[npj Comput. Mater. 2025](https://doi.org/10.1038/s41524-025-01764-6)).
The weights are LGPL-3.0; the dataset behind them
([AIS Square entry](https://www.aissquare.com/datasets/detail?pageType=datasets&name=Solid_State_Electrolyte&id=217))
is CC BY 4.0.

The pinned runtime cannot reconstruct that checkpoint. deepmd-kit 3.0.0b4
changed the repformer layout, so 3.2.1 builds `linear1.matrix` as 640x128 and
`proj_g1g2.matrix` as 32x128 where the file stores 800x128 and 128x32. mliport
does not reshape, re-key or re-initialise weights to make a checkpoint fit, so
DPA has a second environment that has to be selected explicitly:

```bash
./scripts/install_mliport.sh --engines dpa-legacy --device auto
DPA_RUNTIME=legacy ./bin/mliport-dpa-legacy sp structure.vasp \
  --model SSE-VASP-dpa2-inference.pt --model-type dpa --task bulk
```

- `.venv-dpa-legacy` pins `deepmd-kit==3.0.0b3` — an upstream pre-release, and
  the newest release whose network layout matches the checkpoint — with
  `torch==2.3.1` (`+cu121` on a GPU host, which still ships sm_70 for Volta) and
  Python 3.10 to 3.12. The cu121 wheel has no Blackwell kernel: `--device cuda`
  is refused there, while `--device auto` installs the CPU runtime instead.
- `DPA_RUNTIME` is `current` by default and never falls back in either
  direction. The legacy environment refuses DPA-3, DPA4 and DPA4C checkpoints
  and names the runtime to use instead; the current environment refuses the
  legacy deepmd-kit.
- Safe loading: the legacy runtime only accepts a checkpoint that loads with
  `torch.load(weights_only=True)`. The distributed file is a *training*
  checkpoint carrying NumPy scalars, so convert the trusted file once:

```bash
.venv-dpa-legacy/bin/python scripts/convert_dpa_checkpoint.py \
  SSE-VASP-dpa2.pt SSE-VASP-dpa2-inference.pt --i-trust-this-file
```

  The conversion changes no tensor; the legacy runtime refuses the original
  instead of unpickling arbitrary objects.
- Provenance records `dpa_runtime`, the deepmd-kit version and the torch
  version, because the two environments are not interchangeable.

!!! warning "A 118-element type_map is not training coverage"
    DPA-2 serialises the full periodic table into `type_map`; DPA-SSE was trained
    on 15 elements of sulfide-electrolyte chemistry (Li, S, P, Ge, Sn, Si, Ga,
    Sb, As, Cl, Br, I and companions). Everything else - including C and H, so an
    organic adsorbate such as LiDS - is extrapolation and must be validated
    against your own reference before any scientific use.

## Things to know

- **Leaving out the head on a multi-task checkpoint is a fatal configuration
  error.** There is no silent default branch.
- A failed model-metadata check leaves no cached calculator. Retrying the same
  wrapper loads and validates the model again before exposing it.
- **Precision** is defined by the model. mliport records the dtype the backend
  actually reports and never substitutes its own idea of it.
- **Lossless dtype widening is recorded, not hidden.** The pinned deepmd-kit
  3.2.1 builds the DPA-3.1 statistics tensors (`out_bias`, `out_std`,
  `bias_atom_e`) as float64 while the distributed checkpoint stores them in
  float32. Widening stores exactly the same values, so the load is allowed and
  the change is listed in the run record's checkpoint migrations. Narrowing,
  shape changes and value changes still stop the load.
- **Stress** is reported only if the checkpoint and branch you loaded expose it.
- **DPA needs per-process GPU isolation.** The CLI restarts a command with
  `CUDA_VISIBLE_DEVICES` when it is unset; from the Python API, set the variable
  in the calling process before importing the framework.
- DPA has no `inference_mode`. UMA-only options are rejected under strict
  configuration rather than ignored.

mliport is MIT-licensed. The upstream software and model must be cited
separately; see [Citations](../citations.md).
