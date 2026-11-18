# GRACE

GRACE (Graph Atomic Cluster Expansion) models are exported TensorFlow SavedModels, loaded through `tensorpotential`. Unlike the other PyTorch-based backends, GRACE runs on TensorFlow, which means it has its own environment and its own GPU memory management.

## What the model is

### Architecture

GRACE extends the atomic cluster expansion (ACE) formalism from local atomic environments to tree-structured graphs. The key idea behind ACE is to represent the local environment of each atom as a systematic expansion in terms of atomic density correlations — similar in spirit to how a Fourier series represents a function, but adapted to the geometry of atomic neighborhoods.

GRACE comes in two variants:

- **1L (one-layer, local):** Each atom's energy depends only on its immediate neighbors within a cutoff radius. This is the fastest variant and works well when long-range structural correlations are less important.
- **2L (two-layer, semilocal):** One level of equivariant message passing is added on top of the local expansion, allowing each atom to "see" the environments of its neighbors. This captures semilocal interactions — effects that depend not just on what atoms are nearby, but on how those neighbors are themselves arranged.

The distinction matters for your choice: 1L is faster, 2L is more accurate for systems where semilocal effects are significant (most bulk crystals).

Reference: [Bochkarev, Lysogorskiy & Drautz, PRX 2024](https://arxiv.org/abs/2311.16326)

### Training data of the example checkpoint

The `-OMAT-base` models are trained on OMat24 (PBE / PBE+U, bulk inorganic, see the [MACE page](mace.md)) with energies, forces and stresses weighted equally. The `-OMAT` variants are then fine-tuned with a larger energy weight, and the `-OAM` variants are further fine-tuned on MPtrj and a subset of Alexandria ([Lysogorskiy et al., 2025](https://arxiv.org/abs/2508.17936)).

Forces are the gradient of the energy (conservative), and the models are evaluated in the precision of the exported build.

### What this means for you

- Same reference level (PBE / PBE+U) and the same blind spots (no dispersion, bulk-only training) as the other OMat24 models.
- 1L is the choice when speed matters most; 2L when accuracy does.
- Check the model distribution page for its license terms.

## Install

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |

```bash
./scripts/install_mliport.sh --engines grace --device auto
```
<!-- END GENERATED -->

After installation, verify with:

```bash
.venv-grace/bin/mliport doctor --engine grace --device auto
```

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `grace_omat` | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | bulk | None | `d497af4c7d9a...` |
| `grace_omat_fp64` | GRACE-2L-OMAT-medium-base fp64 build | float64 | bulk | None | `53b945efce37...` |
<!-- END GENERATED -->

## Run it

### Command line

```bash
./bin/mliport-grace sp structure.cif \
  --model /path/to/saved_model --model-type grace
```

The `--model` path must point to the SavedModel **directory** (the one containing `saved_model.pb`), not a single file. Official exported SavedModels come from the upstream AMS-ICAMS-RUB release channels.

### INCAR file

```ini
MODEL_TYPE   = GRACE
MODEL_PATH   = /path/to/saved_model
TASK         = bulk
```

```bash
./bin/mliport-grace run INCAR.mliport
```

### Python API

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # must be set before import

from mliport.api import run_single_point

results = run_single_point(
    structure="structure.cif",
    model_path="/path/to/saved_model",
    model_type="grace",
    task="bulk",
)
```

## Things to know

- **No heads.** GRACE models do not expose an mliport-level head. `--task` sets the periodic-boundary semantic (`bulk` or `molecule`).

- **GPU isolation.** TensorFlow's GPU visibility must be fixed per process. The CLI handles this automatically by restarting with `CUDA_VISIBLE_DEVICES`. From the Python API and the queue worker, you need the same isolation — set the environment variable before importing.

- **GPU memory management.** TensorFlow normally reserves nearly all visible GPU memory at startup. mliport enables memory growth by default (`--gpu-memory-growth`), and you can set an explicit limit with `--gpu-memory-limit-mb`. This prevents GRACE from starving other processes on the same GPU.

- **Neighbor list cache.** A Verlet-style neighbor cache is on by default (`--neighbor-cache`, `--neighbor-skin`, default skin 1.5 Å). The neighbor table is rebuilt only when an atom moves more than half the skin distance. This speeds up MD and optimization by avoiding a full neighbor search at every step.

- **Precision** follows the model build. The pinned profiles include both the upstream default build and an fp64 build, so you can choose based on your accuracy needs.

- **Stress** depends on the checkpoint — not all exported builds include stress predictions.

## Licence

**`tensorpotential` is published under the [Academic Software Licence (ASL)](https://github.com/ICAMS/grace-tensorpotential/blob/master/LICENSE.md)**, not an open-source licence: it permits academic, non-commercial use only, and it is reciprocal (GPLv2-like) for derivative works. Installing the `grace` environment means running that software, so commercial or commercial-research use needs a separate licence from the rights holder (ACEworks GmbH), independently of the checkpoint's own licence. mliport does not vendor GRACE, so this restriction does not change mliport's MIT licence — it changes what you may do with this backend.

## Citing GRACE

mliport is MIT-licensed. The upstream software and model must be cited separately; see [Citations](../citations.md).
