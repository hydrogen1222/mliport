# MACE

MACE is an equivariant message-passing neural network for interatomic potentials. In mliport, MACE checkpoints are loaded through the `mace-torch` package. This page explains how the architecture works, how to install and run it, and what to watch out for.

## What the model is

### Architecture

MACE (Multi-Atomic Cluster Expansion) builds higher-order many-body interaction features directly inside each message-passing layer. In a traditional message-passing network, each layer aggregates information from neighboring atoms (two-body interactions), and multi-body effects only emerge after stacking many layers. MACE instead constructs explicit many-body features — three-body, four-body and higher — using the atomic cluster expansion formalism within each layer. This means that just two message-passing layers are typically enough for accurate potentials, which keeps the model compact and fast.

The key innovation is that MACE uses equivariant features: the internal representations transform correctly under rotations and reflections, which means the predicted energy is automatically invariant to rigid-body transformations of the structure. Forces are computed as the exact gradient of the predicted energy, making them conservative — a property required for energy-conserving MD and for gradient-based methods like NEB.

Reference: [Batatia et al., NeurIPS 2022](https://arxiv.org/abs/2206.07697)

### Training data of the example checkpoint

The `mace-omat-0-medium` checkpoint is trained on [OMat24](https://arxiv.org/abs/2410.12771), Meta's dataset of about 110 million VASP calculations on bulk inorganic structures. Many of these structures are deliberately far from equilibrium — rattled atomic positions and high-temperature frames — which improves the model's robustness during MD simulations where atoms can temporarily visit high-energy configurations.

The reference DFT level is **PBE / PBE+U** (VASP 5.4, Materials-Project-style): the +U correction is applied to some transition metals in oxides and fluorides only. This means the model reproduces PBE(+U) results, not experimental measurements.

The older `MACE-MP-0` family is trained on MPtrj (the Materials Project trajectory dataset), also at the PBE / PBE+U level ([Batatia et al., 2024](https://arxiv.org/abs/2401.00096)). The full list of available checkpoints and their licenses is maintained at [mace-foundations](https://github.com/ACEsuit/mace-foundations).

### What this means for you

- Energies and forces approximate PBE(+U) on bulk inorganic matter.
- The model has **no dispersion term** — you can add DFT-D3 with `--d3`, but this is an empirical correction on top of the ML potential.
- Surfaces, molecules and interfaces are **outside the training distribution** of the OMat24-trained checkpoint. Check a few structures against your own DFT before trusting it on those systems.
- Check the checkpoint's license before commercial use — different checkpoints have different licenses.

## Install

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |

```bash
./scripts/install_mliport.sh --engines mace --device auto
```
<!-- END GENERATED -->

After installation, verify with:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `mace_omat` | MACE-OMAT-0 medium, float64 inference | float64 | bulk | None | `d4b14be9afa2...` |
| `mace_omat_float32` | MACE-OMAT-0 medium, float32 inference | float32 | bulk | None | `d4b14be9afa2...` |
<!-- END GENERATED -->

The `mace_omat` profiles use `mace-omat-0-medium.model` from the upstream MACE releases, referenced by SHA-256 in `validation/science/model_manifest.json`. See [Models](../models.md) for download instructions.

## Run it

### Command line

```bash
./bin/mliport-mace sp structure.cif \
  --model /path/to/mace-omat-0-medium.model --model-type mace
```

### INCAR file

```ini
MODEL_TYPE   = MACE
MODEL_PATH   = /path/to/mace-omat-0-medium.model
TASK         = bulk
DTYPE        = float64
```

```bash
./bin/mliport-mace run INCAR.mliport
```

### Python API

```python
from mliport.api import run_single_point

results = run_single_point(
    structure="structure.cif",
    model_path="/path/to/mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
)
print(f"Energy: {results['energy']:.6f} eV")
```

### Precision

`--dtype float64` (the default for the `mace_omat` profile) is the accuracy-first mode. `--dtype float32` is faster but less precise. mliport treats the two precisions as separate model identities — their results are never mixed, and switching precision is recorded in the run provenance. Use float32 deliberately and note it when comparing numbers.

## Things to know

- **Heads.** If a checkpoint offers several heads (e.g. MACE-MH-1), you must name one with `--head`; mliport will not pick for you. Single-head checkpoints need nothing extra. Multi-head energy models load normally.

- **Only energy models are accepted.** A checkpoint without a potential-energy head is refused before it runs. This includes dipole/polarizability-only models (the MACE-MDP family), MACE-POLAR polarizable electrostatics models, and models with charge/spin joint embeddings (the MACE-OMOL family). The error message names the reason and the upstream loader to use instead.

- **Precision is part of the identity.** The recorded dtype always comes from the backend, never from an assumption. Don't compare float32 energies with float64 ones without saying so.

- **Stress** is reported only if the loaded checkpoint provides it (check `implemented_properties`). This differs between checkpoints, so verify with `mliport doctor` or the `stress_supported` field of the validation record rather than assuming.

- **NEB** works with MACE because its forces are derived from the energy (conservative). The run-time consistency check is described in [NEB](../workflows/neb.md).

- **DFT-D3 dispersion** can be added with `--d3` for any MACE calculation. This adds an empirical Grimme D3 correction to the ML-predicted energy and forces.

## Citing MACE

mliport is MIT-licensed. The upstream MACE software and the model checkpoint must be cited separately; see [Citations](../citations.md) for the full list of references.
