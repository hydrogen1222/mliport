# UMA (FAIRChem)

UMA checkpoints from Meta FAIR are loaded through the upstream `fairchem-core`
stack. `uma` and `fairchem` (the upstream package name) are the same runtime, so
either works as `--model-type`.

## Install

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv` | 3.11, 3.12, 3.13 | `fairchem-core==2.21.0` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines uma --device auto
```
<!-- END GENERATED -->

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `uma_omat` | UMA-s 1.2 (OMat task head) | upstream/model-defined | omat | None | `ba5c0d912efa...` |
<!-- END GENERATED -->

## Run it

```bash
.venv/bin/mliport sp structure.cif \
  --model /path/to/uma.pt --model-type uma --task omat
```

Get the checkpoint from the upstream fairchem/UMA release channels and verify
its SHA-256 before use. mliport never redistributes model weights, and UMA's
license is non-commercial research only; see [Citations](../citations.md).

## Things to know

- **`--task` picks the task family** (`omat` for periodic bulk, `omol` for
  molecules, `oc20` for catalysis). UMA has no separate head field. The families
  are not interchangeable: run a molecule with an `omol` task and a periodic
  crystal with an `omat` task.
- **Inference modes.** `--inference-mode` accepts `default` or `turbo`. Turbo
  (TF32 plus `torch.compile`) is opt-in and MD does not use it by default, so
  precision-sensitive runs stay in the default mode. `--activation-checkpointing`
  trades speed for GPU memory.
- **Precision** is defined by the model; mliport reports the framework version
  rather than pretending to control the weights.
- **NEB is not available.** The UMA predictor reports `direct_forces`: its forces
  are predicted directly, not obtained as the gradient of its energy. NEB needs
  the latter, so the consistency pre-flight always refuses such models (see
  [NEB](../workflows/neb.md)).
- **Stress** is reported only if the checkpoint exposes it.

mliport is MIT-licensed. The upstream software and model must be cited
separately.
