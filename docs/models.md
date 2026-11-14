# Models

mliport is not a model. It loads other people's potentials through adapters,
runs your workflow with them, and writes down exactly which model did the work.
That division of labour is worth keeping in mind when you read a result.

| mliport takes care of | The checkpoint and you are responsible for |
|---|---|
| loading and selecting the model | the weights and the data they were trained on |
| the resolved configuration and the provenance record | the reference state the energies belong to |
| refusing options the backend cannot express | how accurate the potential is |
| checking that the backend runtime can really run | the absolute energy scale and conventions |

## Five backends, no favorite

`mace`, `dpa`, `grace`, `uma` and `nep` are five runtime adapters of equal
standing; `fairchem` is accepted as another name for `uma`. There is no
implicit default. A run that names no backend, directly or through a model
alias or profile, stops with an error.

| Backend | Adapter | Framework | Typical checkpoint |
|---|---|---|---|
| MACE | `mace-torch` | PyTorch | `.model` |
| DPA | `deepmd-kit` | PyTorch | `.pt` / `.pth` |
| GRACE | `tensorpotential` | TensorFlow | SavedModel directory |
| UMA | `fairchem-core` | PyTorch | `.pt` |
| NEP | `calorine` (ASE) / native `gpumd` | Python / GPUMD | `nep.txt` |

Pinned profiles, precision, head semantics and known limits are on the
per-backend pages: [MACE](backends/mace.md), [DPA](backends/dpa.md),
[GRACE](backends/grace.md), [UMA](backends/uma.md), [NEP](backends/nep.md).
NEP is opt-in and needs an explicit `NEP_RUNTIME`; the other rows of this
table need nothing beyond the backend environment.

## Model support status

`verified` in the two device columns means **mliport called the model
correctly**: it loaded, returned finite energies, forces and (where the
checkpoint defines one) stress, a short MD and a small NEB finished, and the
recorded model hash, head, dtype and device matched what was requested. It is
not a statement about accuracy: how well a potential describes your chemistry
is a separate question that this table does not answer.

| Status | Meaning |
|---|---|
| `verified` | Every callable check passed on that device, in the backend's own environment; the evidence is recorded. |
| `untested` | The adapter should load it; no evidence yet. |
| `needs_runtime_upgrade` | A pinned runtime version has to change before the model can be tried. |
| `unsupported` | Known not to load or run on that device, or the adapter cannot express what the model computes; mliport fails closed instead of running it. |

| Label | Engine | Artifact | Head / task | dtype | CPU | GPU | Evidence |
|---|---|---|---|---|---|---|---|
| `uma_omat` | uma | `uma-s-1p2.pt` | `omat` | model-defined | `verified` | `verified` | [re-check record](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6880031494) |
| `mace_omat` | mace | `mace-omat-0-medium.model` | default | `float64` | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `mace_omat_float32` | mace | `mace-omat-0-medium.model` | default | `float32` | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `dpa_omat` | dpa | `DPA-3.1-3M.pt` | `Omat24` | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `grace_omat` | grace | `GRACE-2L-OMAT-medium-base` | default | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `grace_omat_fp64` | grace | `GRACE-2L-OMAT-medium-base-fp64` | default | `float64` | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `grace_3l_omat_large_ft_am` | grace | `GRACE-3L-OMAT-large-ft-AM` | default | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `grace_2l_smax_omat_large` | grace | `GRACE-2L-SMAX-OMAT-large` | default | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `dpa31_sse_pbe` | dpa | `DPA-3.1-3M.pt` | `Domains_SSE_PBE` | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `dpa31_sse_pbesol` | dpa | `DPA-3.1-3M.pt` | `Domains_SSE_PBESol` | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `dpa4_neo_omat` | dpa | `DPA4-Neo-OMat24-v20260805.pt` | single task | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `dpa4_air_omat` | dpa | `DPA4-Air-OMat24-v20260805.pt` | single task | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `dpa4_plus_omat` | dpa | `DPA4-Plus-OMat24-v20260805.pt` | single task | model-defined | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `mace_matpes_r2scan` | mace | `MACE-MATPES-r2SCAN-0` medium | single head (`default`) | `float64` | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `mace_mh1_omat_pbe` | mace | `MACE-MH-1` | `omat_pbe` | `float64` | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `mace_mh1_matpes_r2scan` | mace | `MACE-MH-1` | `matpes_r2scan` | `float64` | `verified` | `verified` | [runs](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6877006841), [CPU–GPU deltas](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878720684) |
| `uma_s_1p2p1_omat` | uma | `uma-s-1p2p1.pt` | `omat` | model-defined | `verified` | `verified` | [UMA record](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878717708) |
| `uma_s_1p2p1_ft_r2scan_omat` | uma | `uma-s-1p2p1-ft-r2scan.pt` | `omat` | model-defined | `verified` | `verified` | SP, forces and MD verified; NEB refused by the gradient preflight — [UMA record](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6878717708) |
| `nep89_gpumd` | nep | NEP89 `nep.txt` | — | model-defined | `unsupported` | `verified` | [native GPU runtime evidence](https://github.com/hydrogen1222/mliport/pull/75#issuecomment-6689717878) |

The checks behind these cells ran on 2026-11-14 on one Tesla
V100-SXM2-16GB (driver 580.173.02), in the environments the installer builds:
mace-torch 0.3.16 on torch 2.8.0, deepmd-kit 3.2.1.dev63 on torch 2.11.0,
tensorpotential 0.6.0 on TensorFlow 2.20.0, fairchem-core 2.25.0 on torch
2.13.0. Every `verified` model loaded, gave finite energy, forces and stress on
the 50-atom `li10gep2s12_primitive.vasp` cell, matched its recorded identity and
completed the bounded probe (shared calculator, central-difference forces, 10
NVE steps, a five-image NEB) on both devices. Across all profiles the largest
CPU–GPU difference on that cell was **6.1e-7 eV/atom** and **7.5e-6 eV/Å**, both
from `mace_omat_float32`; the float64 models agree at the 1e-14 level. The UMA
runs record `inference_preset: batch` with `nitro_enabled: false`, and
`--inference-mode turbo` records `multigpu` with `nitro_enabled: false`. One
exception is stated in the table: the r²SCAN fine-tune is refused by the NEB
gradient preflight, because its energy scale does not resolve the derivative in
float32. Records, hashes, deltas and verbatim failures are in
[pull request #75](https://github.com/hydrogen1222/mliport/pull/75).

One row deserves a note: NEP89 is verified only on the native GPU runtime,
because the Calorine ASE runtime cannot parse that model format, so NEP89 on
CPU is `unsupported` rather than merely untested.

The archived `uma-s-1.pt` is deliberately **not** in this table. It is an
archived upstream release with a known extensivity bug; mliport has not checked
it and does not recommend it, and `unsupported` would be the wrong word because
`unsupported` means mliport refuses the model — nothing refuses this one.

The hex hashes, source URLs and licenses of the pinned profiles live in the
[model manifest](https://github.com/hydrogen1222/mliport/blob/main/validation/science/model_manifest.json);
the callable checks behind each status are described in
[Validation](validation.md).

## Comparing results honestly

Some habits keep a calculation scientifically defensible, and mliport records
what you need to follow them:

- **Energies from different models are not comparable.** A formation energy or
  a barrier must come from one model, one head and one set of reference states.
  Mixing models silently is an error in the science, not a feature.
- **A passing workflow says nothing about accuracy.** That the adapter ran your
  NEB to the end tells you the plumbing works, not that the potential
  describes your chemistry.
- **Benchmark numbers do not travel.** OMat24 results characterise the pinned
  model on that benchmark and do not extend to every material.
- **Task and head are part of the model.** UMA task families (`omat`, `omol`,
  …) and DPA or MACE heads change what the model computes. mliport records the
  task and head actually used and refuses to guess a multi-task branch.
- **Precision is part of the model too.** MACE `float32` and `float64` are
  separate profiles; say so whenever you compare their energies.

## Choosing a backend for a run

The quickest way is a flag:

```bash
mliport sp structure.cif --model model.model --model-type mace
```

For models you use often, define an alias in `settings.ini` (an alias must say
which engine it is for):

```ini
[model:mace_omat]
engine = mace
path = /models/mace-omat-0-medium.model
task = bulk
```

```bash
mliport sp structure.cif --model-alias mace_omat
```

`--model` also accepts an alias name. How aliases, profiles and flags combine
is explained in [Configuration](configuration.md).

## What gets recorded about the model

Every run stores the effective `model_type`, `model_path`, task, head and dtype,
the backend and framework versions, the requested and the actual device, and the
mliport commit. Two results are comparable only if these match. The fields are
described in [Output files](outputs.md).

## Where to get the checkpoints

| Engine | Checkpoint | Source |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release `mace_omat_0`](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

These are the artifacts the validation suite is pinned to. Their SHA-256 values
describe the distributed files (local paths are never written into the
manifest):

| Profile | Artifact | SHA-256 (prefix) |
|---|---|---|
| `mace_omat` | `mace-omat-0-medium.model` | `d4b14be9afa2...` |
| `dpa_omat` | `DPA-3.1-3M.pt` | `86dd3a804d78...` |
| `grace_omat` | `GRACE-2L-OMAT-medium-base` | directory SavedModel |
| `uma_omat` | `uma-s-1p2.pt` | `ba5c0d912efa...` |

The full hashes are in
[`validation/science/model_manifest.json`](https://github.com/hydrogen1222/mliport/blob/main/validation/science/model_manifest.json).

!!! note "What was and was not downloaded during acceptance"
    The MACE release asset was downloaded in an isolated cache during the LGPS
    acceptance and matched the pinned hash. The Hugging Face cold-download path
    was **not** validated over the network on the acceptance host: its
    SOCKS/httpx setup rejects the Hugging Face client's default proxy
    configuration, so the acceptance used hash-verified local copies of all four
    artifacts. If your network needs a proxy, see the recipe below.

Checkpoint licenses belong to the upstream projects. In particular, UMA's is a
non-commercial research license.

## Proxies and offline use

Hugging Face downloads go through `httpx` inside `huggingface_hub`. Behind a
SOCKS proxy (`ALL_PROXY=socks5...`), install the socks extra in the environment
you download with:

```bash
uv pip install 'httpx[socks]' --python .venv/bin/python
```

Some proxy setups also put bracketed entries like `[::1]` into `NO_PROXY`, which
httpx rejects with `Invalid port`. Keep the list to plain host names:

```bash
NO_PROXY=localhost,127.0.0.1 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('deepmodelingcommunity/DPA-3.1-3M', 'DPA-3.1-3M.pt'))"
```

A model on local disk needs no network at all: pass its path to `--model`, and
use `--source offline` when the installer is involved.
