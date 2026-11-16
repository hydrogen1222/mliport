# Quickstart

This guide walks you from a fresh clone to a working single-point, relaxation
and short MD run in about ten minutes. We use **MACE** as the first backend
because its checkpoint is openly licensed and small; the other four backends
are at the end.

## Prerequisites

You need a Linux machine (or WSL2 on Windows) with:

- **git** — to clone the repository.
- **[uv](https://docs.astral.sh/uv/)** — the Python environment manager that
  the installer uses. Install it with
  `curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it.
- **NVIDIA GPU + CUDA driver** (optional) — for GPU-accelerated calculations.
  CPU-only works fine for small test structures.

## 1. Clone and install

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto
```

The installer creates a dedicated Python environment (`.venv-mace/`), installs
mliport and the MACE backend into it, writes a launcher script at
`bin/mliport-mace`, and runs `mliport doctor` as a sanity check. The whole
process takes a few minutes.

Use `--device cpu` if the machine has no NVIDIA GPU. For multiple backends at
once, pass `--engines mace dpa grace uma`. Full details — offline installs,
proxy setups, disk space, manual installs — are in [Installation](installation.md).

## 2. Verify the installation

```bash
./bin/mliport-mace doctor --engine mace --device auto
```

A healthy install ends with:

```
Environment checks passed for MACE on CUDA.
```

(or `on CPU` if you installed with `--device cpu`). If a check fails, see
[Troubleshooting](troubleshooting.md) for the common fixes.

!!! tip "JSON output for scripts"
    `mliport doctor --json` prints machine-readable results, useful for CI
    pipelines and automated setup scripts.

## 3. Get a model checkpoint

mliport never downloads or bundles model weights — you fetch them from the
upstream project. For MACE-OMAT-0 (a general-purpose foundation model trained
on OMat24 data):

```bash
# Option A: download from the GitHub release page
wget https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model

# Option B: from Hugging Face (if you have huggingface-cli)
huggingface-cli download ACEsuit/MACE-OMAT-0 mace-omat-0-medium.model
```

Put the file somewhere convenient. In the examples below, replace
`/path/to/mace-omat-0-medium.model` with the actual path.

!!! note "Which checkpoint to use"
    The [Models](models.md) page lists all supported checkpoints, their
    download links, hashes, and license terms. MACE-OMAT-0-medium is a good
    starting point for periodic bulk materials.

## 4. Run a single-point calculation

The repository ships a 50-atom Li₁₀GeP₂S₁₂ (LGPS) cell for testing. Let's
calculate its energy and forces:

```bash
./bin/mliport-mace sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk \
  --output runs/lgps-sp
```

mliport prints a summary and writes results to `runs/lgps-sp/`:

| File | Contents |
|---|---|
| `mliport_results.json` | Energy, forces, stress, timing, provenance |
| `resolved_config.json` | Every configuration value and where it came from |
| `OUTCAR` | VASP-compatible output (energy, forces, stress) |
| `CONTCAR` | Final structure in POSCAR format |
| `run.log` | Full run log |

With the MACE-OMAT-0 checkpoint on the LGPS cell, you should see an energy
around −216 eV and the formula `Ge2Li20P4S24` (50 atoms).

!!! warning "Output directory protection"
    If `runs/lgps-sp/` already contains results, mliport refuses to overwrite
    them. Use `--overwrite` to replace them deliberately, or choose a different
    `--output` path.

## 5. Relax the structure

```bash
./bin/mliport-mace opt examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk \
  --fmax 0.05 --max-steps 200 \
  --output runs/lgps-opt
```

`--fmax` is the force convergence threshold in eV/Å; `--max-steps` caps the
optimization. Whether the relaxation actually converged is recorded in
`mliport_results.json` — a run that did not converge says so honestly. The
relaxed structure is saved as `CONTCAR`.

To relax the cell shape and volume together with the atomic positions, add
`--cell-opt`.

## 6. Run a short MD

```bash
./bin/mliport-mace md examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk \
  --ensemble NVT --thermostat LANGEVIN --temp 600 \
  --timestep 1.0 --steps 1000 --save-interval 10 \
  --output runs/lgps-md
```

This runs 1000 steps of NVT Langevin dynamics at 600 K, saving a frame every
10 steps. The trajectory is written as `XDATCAR` (readable by VESTA, OVITO,
etc.), and thermodynamic quantities are logged in `OSZICAR`.

!!! tip "Choosing an ensemble"
    mliport supports three MD ensembles: `NVE` (microcanonical), `NVT`
    (canonical, with Langevin/Bussi/NHC thermostats), and `NPT` (isothermal-
    isobaric, ASE backends only). See [MD workflow](workflows/md.md) for the
    full options.

## 7. Read the results from Python

Every result lives in `mliport_results.json`. Here is how to load it:

```python
import json
from pathlib import Path

# Read the single-point result
with open("runs/lgps-sp/mliport_results.json") as f:
    record = json.load(f)

calc = record["calculation"]
print(f"Energy:  {calc['results']['energy']:.4f} eV")
print(f"Device:  {record['environment']['actual_device_type']}")
print(f"Model:   {record['model']['path']}")
print(f"SHA-256: {record['model']['sha256'][:16]}...")

# Read the optimization result
with open("runs/lgps-opt/mliport_results.json") as f:
    opt = json.load(f)

print(f"Converged: {opt['calculation']['optimization']['converged']}")
print(f"Steps:     {opt['calculation']['optimization']['steps']}")
```

For calculations inside larger workflows, the Python API is more convenient:

## 7. The task folder way

The commands above work, but you do not want to retype them. The intended daily
use is one folder per task: an `INCAR.mliport` next to the structure file, and
`mliport run` with no arguments at all.

```text
LiDS-on-LPSC/
  INCAR.mliport
  POSCAR
```

```bash
cd LiDS-on-LPSC
mliport run
```

`run` writes into `results/` beside the INCAR, and `--output` or `OUTPUT_DIR`
still choose another directory. A model given as a bare file name is looked up
in the repository's `models/` directory, so `MODEL_PATH = mace-omat-0-medium.model`
works once the file is in `models/`. Every key is documented in the
[INCAR key reference](incar.md).

## 8. Python API

For calculations inside larger workflows:

```python
from mliport.api import calculate_energy, run_single_point

# Just the energy (eV)
energy = calculate_energy(
    "examples/structures/li10gep2s12_primitive.vasp",
    "/path/to/mace-omat-0-medium.model",
    model_type="mace", task="bulk"
)

# Full result dict
result = run_single_point(
    "examples/structures/li10gep2s12_primitive.vasp",
    "/path/to/mace-omat-0-medium.model",
    model_type="mace", task="bulk",
    output_dir="runs/api-sp"
)
```

See [Python API](api.md) for the complete reference.

## Try the other backends

Each backend has its own Python environment and launcher. Install with
`--engines <name>` and run through its launcher:

=== "DPA"

    ```bash
    ./scripts/install_mliport.sh --engines dpa --device auto

    ./bin/mliport-dpa sp examples/structures/li10gep2s12_primitive.vasp \
      --model DPA-3.1-3M.pt --model-type dpa \
      --task bulk --head Omat24 \
      --output runs/lgps-dpa-sp
    ```

    DPA uses multi-task checkpoints — you must specify `--head` to select the
    branch. See [DPA backend](backends/dpa.md).

=== "GRACE"

    ```bash
    ./scripts/install_mliport.sh --engines grace --device auto

    ./bin/mliport-grace sp examples/structures/li10gep2s12_primitive.vasp \
      --model GRACE-2L-OMAT-medium-base --model-type grace \
      --task bulk \
      --output runs/lgps-grace-sp
    ```

    GRACE models are TensorFlow SavedModel directories, not single files. See
    [GRACE backend](backends/grace.md).

=== "UMA"

    ```bash
    ./scripts/install_mliport.sh --engines uma --device auto

    ./bin/mliport-uma sp examples/structures/li10gep2s12_primitive.vasp \
      --model uma-s-1p2.pt --model-type uma \
      --task omat \
      --output runs/lgps-uma-sp
    ```

    UMA uses `--task omat` for bulk materials, `--task omol` for molecules.
    Its checkpoint is under a non-commercial research license. See
    [UMA backend](backends/uma.md).

=== "NEP"

    ```bash
    ./scripts/install_mliport.sh --engines nep --device cpu

    ./bin/mliport-nep sp examples/structures/li10gep2s12_primitive.vasp \
      --model /path/to/nep.txt --model-type nep \
      --nep-runtime ase --task bulk \
      --output runs/lgps-nep-sp
    ```

    NEP requires you to supply your own `nep.txt` trained for your system, and
    you must explicitly choose `--nep-runtime ase` or `--nep-runtime gpumd`.
    See [NEP backend](backends/nep.md).

!!! info "No default backend"
    Leaving out `--model-type` (when no model alias supplies one) is always an
    error. mliport never guesses which model to use — you always know exactly
    which model produced a number.

## What's next

| Topic | Page |
|---|---|
| Understand which model fits your system | [Models](models.md) |
| Configure runs, INCAR files and `settings.ini` | [Configuration](configuration.md) |
| Run MD, NEB, configuration search, batch sweeps | [Workflows](workflows/index.md) |
| Analyze a trajectory (MSD, diffusion, RDF, …) | [Analysis](analysis/index.md) |
| Use mliport from Python | [Python API](api.md) |
| Check what has been validated | [Validation](validation.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) |
