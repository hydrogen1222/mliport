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

## Four backends, no favourite

`mace`, `dpa`, `grace` and `uma` are four runtime adapters of equal standing;
`fairchem` is accepted as another name for `uma`. There is no implicit
default. A run that names no backend, directly or through a model alias or
profile, stops with an error.

| Backend | Adapter | Framework | Typical checkpoint |
|---|---|---|---|
| MACE | `mace-torch` | PyTorch | `.model` |
| DPA | `deepmd-kit` | PyTorch | `.pt` / `.pth` |
| GRACE | `tensorpotential` | TensorFlow | SavedModel directory |
| UMA | `fairchem-core` | PyTorch | `.pt` |

Pinned profiles, precision, head semantics and known limits are on the
per-backend pages: [MACE](backends/mace.md), [DPA](backends/dpa.md),
[GRACE](backends/grace.md), [UMA](backends/uma.md).

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

Checkpoint licences belong to the upstream projects. In particular, UMA's is a
non-commercial research licence.

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
