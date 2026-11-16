# Backends

mliport supports five machine-learned interatomic potential backends. Each backend wraps a different neural-network architecture and its upstream framework, but all of them present the same interface to mliport: you pass a structure and a checkpoint, and mliport returns energies, forces and (where available) stresses in a consistent format.

## What is a backend?

A "backend" in mliport is a combination of three things:

1. **A neural-network architecture** — the mathematical model that predicts atomic energies and forces from nuclear positions (MACE, DPA, GRACE, UMA or NEP).
2. **An upstream framework** — the Python package that loads and evaluates the trained checkpoint (`mace-torch`, `deepmd-kit`, `tensorpotential`, `fairchem-core` or `calorine`).
3. **A trained checkpoint** — the file containing the learned weights. The checkpoint determines what chemistry the model knows, at what DFT reference level, and for which kinds of systems (bulk, surface, molecule).

mliport runs exactly one backend per calculation. The backend is selected by `--model-type` (or `MODEL_TYPE` in an INCAR file), from a model alias or profile in your `settings.ini`, or the run stops with `No MLIP backend was selected`. There is no default and no fallback.

## Backend comparison

| | MACE | DPA | GRACE | UMA | NEP |
|---|---|---|---|---|---|
| Architecture | Equivariant message passing with many-body features | Message passing on a line-graph series (pairs → angles → …) | Graph atomic cluster expansion (1L local, 2L semilocal) | eSEN with mixture of linear experts | Neuroevolution potential (feedforward, GPU-optimized) |
| Example checkpoint | `mace-omat-0-medium` | `DPA-3.1-3M` | `GRACE-2L-OMAT-medium-base` | `uma-s-1p2` | Your own `nep.txt` |
| Training data | OMat24 (~110M structures) | OpenLAM-v1 (31 datasets) | OMat24 | OMat24 + OMol25 + OC20 + ODAC25 + OMC25 | User-supplied |
| Reference DFT level | PBE / PBE+U (bulk) | Per head (Omat24 = PBE/PBE+U) | PBE / PBE+U (bulk) | Per task (omat = PBE/PBE+U) | Whatever you trained on |
| Key selector | `--dtype` (float32/float64) | `--head` (dataset branch) | Model build (default/fp64) | `--task` (omat/omol/oc20/…) | `--nep-runtime` (ase/gpumd) |
| Framework | PyTorch | PyTorch | TensorFlow | PyTorch | Calorine (ASE) or native GPUMD |
| Environment | `.venv-mace` | `.venv-dpa` | `.venv-grace` | `.venv` | `.venv-nep` |

None of these example checkpoints includes dispersion corrections. Only UMA was trained on surfaces, molecules and porous materials as well as bulk crystals. A foundation model reproduces its reference DFT functional — not experiment — and how well it does that on *your* system is something to verify with a few DFT single points. See [Models](../models.md) for guidance on comparing backends honestly.

## How to choose a backend

The right backend depends on your system and what you need from the calculation:

**I need to study surfaces, molecules or metal-organic frameworks**
→ **UMA** is the only example model trained on non-bulk systems (OC20 for catalysis surfaces, OMol25 for molecules, ODAC25 for MOFs). The other four example models were trained only on bulk inorganic crystals, so applying them to surfaces or molecules is extrapolation.

**I need double-precision (float64) accuracy**
→ **MACE** with `--dtype float64`, or **GRACE** with the fp64 build. Most other backends compute in the precision their checkpoint was exported with.

**I need maximum throughput for long MD runs**
→ **NEP** with the native GPUMD runtime (`--nep-runtime gpumd`) is a single GPU process for the entire run with minimal overhead. **GRACE 1L** models trade some accuracy for speed. UMA's `turbo` inference mode enables `torch.compile` and TF32.

**I have my own trained potential**
→ **NEP** is designed for user-trained `nep.txt` files. The other four backends can also load custom checkpoints, but their example checkpoints are foundation models.

**I need the widest elemental coverage from one model**
→ **DPA** (OpenLAM-v1 covers 31 datasets across many chemistries) or **UMA** (five dataset families covering bulk, surface, molecule, MOF and molecular-crystal chemistry).

**I need to run NEB or configuration search**
→ Any backend whose forces are the gradient of the energy (conservative). MACE, DPA, GRACE and UMA all qualify. NEP works with `--nep-runtime ase` only; the native GPUMD runtime does not support NEB or configuration search.

## Separate environments

Each backend lives in its own Python virtual environment because their upstream frameworks have incompatible dependencies:

| Backend | Environment | Why separate |
|---|---|---|
| MACE | `.venv-mace` | `mace-torch` pins specific PyTorch versions |
| DPA | `.venv-dpa` | `deepmd-kit` needs its own torch + C++ extensions |
| GRACE | `.venv-grace` | `tensorpotential` uses TensorFlow, not PyTorch |
| UMA | `.venv` | `fairchem-core` pins its own torch version |
| NEP | `.venv-nep` | `calorine` is a compiled C extension |

The install script creates the environment you ask for:

```bash
./scripts/install_mliport.sh --engines mace --device auto
```

To verify that a backend can actually run, use `doctor`:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

See [Installation](../installation.md) for the full setup procedure.

## GPU isolation

DPA and GRACE need their GPU chosen before the framework loads its CUDA context. The command line handles this automatically by restarting itself in a fresh process with `CUDA_VISIBLE_DEVICES` set. From the Python API, you must set the environment variable in the calling process yourself before importing the framework:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from mliport.api import run_single_point
```

## Page outline

Every backend page below follows the same structure:

1. **What the model is** — architecture, training data, what the forces represent
2. **Install** — environment and install command
3. **Pinned profiles** — which checkpoints mliport has registered
4. **Run it** — command-line and INCAR examples
5. **Things to know** — gotchas, limitations and backend-specific options
