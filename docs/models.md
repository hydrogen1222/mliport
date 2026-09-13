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
