# UMA (FAIRChem)

UMA checkpoints from Meta FAIR are loaded through the upstream `fairchem-core`
stack. `uma` and `fairchem` (the upstream package name) are the same runtime, so
either works as `--model-type`.

## What the model is

- **Architecture.** UMA is an eSEN-type equivariant network with a 6 Å cutoff whose
  linear layers are a *mixture of linear experts*: the expert weights are chosen
  once per system from its composition, charge, spin and task, so a large total
  parameter count costs only a small active one at inference
  ([Wood et al., 2025](https://arxiv.org/abs/2506.23971)).
- **Training data.** About 500 million structures from OMat24 (bulk inorganic,
  PBE / PBE+U), OMol25 (molecules, ωB97M-V), OC20 (catalysis surfaces), ODAC25
  (metal-organic frameworks) and OMC25 (molecular crystals). The datasets use
  different DFT settings, which is why you choose a **task**: each task reproduces
  its own dataset's reference level, and energies from different tasks must not be
  compared.
- **Forces.** The UMA paper reports UMA-S and UMA-M as conservative (forces and
  stress by differentiation, after a direct-force pretraining stage). mliport does
  not rely on that statement: NEB and the configuration search read the
  predictor's `direct_forces` flag and measure energy–force consistency at run
  time (see below).
- **What this means for you.** UMA is the one example backend trained on surfaces,
  molecules and porous materials as well as bulk, so for an adsorbate on a slab,
  compare the `omat` and `oc20` tasks against your own DFT before choosing. Its
  license is non-commercial research only.

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
- **NEB and configuration search** need forces that are the gradient of the
  energy. mliport decides this at run time, not from this page: a predictor that
  reports `direct_forces = true` is refused, otherwise a finite-difference
  pre-flight measures the consistency on your structures (see
  [NEB](../workflows/neb.md)).
- **Stress** is reported only if the checkpoint exposes it.

mliport is MIT-licensed. The upstream software and model must be cited
separately.
