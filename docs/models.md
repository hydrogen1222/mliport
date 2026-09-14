# Models and backends

mliport is **not a model**. It runs third-party MLIPs through adapters and
records exactly which model ran.

## What mliport does and does not decide

| mliport decides | you / the upstream model decides |
|---|---|
| how a model is loaded and selected | the model weights and training data |
| the resolved workflow configuration and provenance | the scientific reference state |
| whether a requested option is expressible by the backend | the accuracy of the potential |
| whether the backend runtime can actually run | the absolute energy scale and units conventions |

## Backends are equal citizens

`mace`, `dpa`, `grace` and `uma` are four runtime adapters. `fairchem` is an
accepted alias for `uma`. There is no implicit default: a run without an
explicit backend (or a model alias/profile that provides one) fails closed.

| Backend | Adapter | Framework | Typical checkpoint |
|---|---|---|---|
| MACE | `mace-torch` | PyTorch | `.model` |
| DPA | `deepmd-kit` | PyTorch | `.pt`/`.pth` |
| GRACE | `tensorpotential` | TensorFlow | SavedModel directory |
| UMA | `fairchem-core` | PyTorch | `.pt` |

See the per-backend pages for pinned profiles, precision, head semantics and
limitations: [MACE](backends/mace.md), [DPA](backends/dpa.md),
[GRACE](backends/grace.md), [UMA](backends/uma.md).

## Scientific rules that mliport enforces or makes explicit

- **Absolute energy scales differ between models.** A formation energy or
  barrier must be computed with one model/head/reference combination; mixing
  models silently is a scientific error, not a supported feature.
- **Model accuracy is not mliport accuracy.** The adapter passing a workflow
  smoke test says nothing about whether the potential describes your
  chemistry.
- **OMat24 benchmark numbers do not extrapolate to every chemistry.** They
  characterise the pinned model on that benchmark.
- **Preserve task/head semantics.** UMA task families (`omat`, `omol`, ...)
  and DPA/MACE heads are part of the model identity. mliport records the
  effective task/head and refuses to guess a multi-task branch.
- **Compare like with like.** Formation energies need the same model,
  reference states and workflow; barriers need the same model and NEB
  settings.
- **Precision is part of the identity.** MACE float32 and float64 are
  separate profiles; never mix their energies without stating it.

## Selecting a backend

CLI flag (highest priority):

```bash
mliport sp structure.cif --model model.model --model-type mace
```

Model alias in `settings.ini` (the alias must declare `engine`):

```ini
[model:mace_omat]
engine = mace
path = /models/mace-omat-0-medium.model
task = bulk
```

```bash
mliport sp structure.cif --model-alias mace_omat
```

`--model` also accepts an alias name. See
[configuration.md](configuration.md) for profiles and precedence.

## Model identity in outputs

Every run records the effective `model_type`, `model_path`, `task`, `head`,
dtype, backend/framework version, requested vs actual device, and the
software commit. Two results are only comparable when this identity matches.
See [outputs.md](outputs.md).

## Pinned checkpoints

The validation suite pins one artifact per engine. The SHA-256 values are the
distributed files; local paths are never written into the manifest.

| Profile | Artifact | Source | SHA-256 (prefix) |
|---|---|---|---|
| `mace_omat` | `mace-omat-0-medium.model` | [mace-foundations `mace_omat_0`](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) | `d4b14be9afa2...` |
| `dpa_omat` | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) | `86dd3a804d78...` |
| `grace_omat` | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) | directory SavedModel |
| `uma_omat` | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) | `ba5c0d912efa...` |

The MACE GitHub asset was downloaded in an isolated cache during the LGPS
acceptance and matched the pinned hash; local copies of all four artifacts
are used for the acceptance runs. Checkpoint licenses stay with the upstream
projects, and UMA's is a non-commercial research license.

## Proxy and offline access

Hugging Face downloads use `httpx` inside `huggingface_hub`. With a SOCKS
proxy (`ALL_PROXY=socks5...`) install the socks extra in the environment you
use for downloads:

```bash
uv pip install 'httpx[socks]' --python .venv/bin/python
```

Some proxy stacks also put bracket entries such as `[::1]` into `NO_PROXY`;
httpx rejects those with `Invalid port`. Keep the list to plain host names:

```bash
NO_PROXY=localhost,127.0.0.1 HF_ENDPOINT=https://hf-mirror.com \
  .venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('deepmodelingcommunity/DPA-3.1-3M', 'DPA-3.1-3M.pt'))"
```

A model path on local disk needs no network at all: pass it to `--model` and
run with `--source offline` where the installer is involved.
