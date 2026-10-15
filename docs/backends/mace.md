# MACE

MACE checkpoints are loaded through `mace-torch`. A MACE profile is a
checkpoint plus a precision, and mliport treats the two precisions as different
models: `float64` is the primary, accuracy-first profile, while `float32` is a
separate identity and its results are never mixed with float64 ones.

## Install

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines mace --device auto
```
<!-- END GENERATED -->

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `mace_omat` | MACE-OMAT-0 medium, float64 inference | float64 | bulk | None | `d4b14be9afa2...` |
| `mace_omat_float32` | MACE-OMAT-0 medium, float32 inference | float32 | bulk | None | `d4b14be9afa2...` |
<!-- END GENERATED -->

The `mace_omat` profiles use `mace-omat-0-medium.model` from the upstream
MACE releases, referenced by SHA-256 in
`validation/science/model_manifest.json`. See [Models](../models.md) for the
download link.

## Run it

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model /path/to/mace-omat-0-medium.model --model-type mace
```

`--task` is the explicit periodic-boundary semantic (`bulk` or `molecule`).
`--dtype float32` is faster and less precise; use it deliberately and say so
when you compare numbers.

## Things to know

- **Heads.** If a checkpoint offers several heads you must name one with
  `--head`; mliport will not pick for you. Single-head checkpoints need nothing.
- **Precision is part of the identity.** The recorded dtype always comes from
  the backend, never from an assumption. Don't compare `float32` energies with
  `float64` ones without saying so.
- **Stress** is reported only if the loaded checkpoint provides it
  (`implemented_properties`), and that differs between checkpoints, so check the
  `stress_supported` field of the validation record or run `mliport doctor`
  rather than assuming.
- **NEB** works with MACE because its forces are derived from the energy; the
  run-time consistency check is described in [NEB](../workflows/neb.md).

mliport is MIT-licensed. The upstream software and model must be cited
separately; see [Citations](../citations.md).
