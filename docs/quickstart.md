# Quickstart

In about ten minutes you will install one backend, run a single-point
calculation and a short relaxation, and know where to look for the result.
We use MACE because its checkpoint is open and small; the other backends are at
the end. You do not need all four to get started.

## 1. Install one backend

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto
```

The installer builds a separate Python environment for the backend
(`.venv-mace`), installs mliport into it, writes a launcher in `bin/` and runs
`mliport doctor` as a last step. Use `--device cpu` if the machine has no
NVIDIA GPU. Offline hosts, several backends at once and manual installs are in
[Installation](installation.md).

## 2. Check that it works

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

`doctor` exits non-zero when the environment cannot actually run the backend.
If a check fails, fix that first; [Troubleshooting](troubleshooting.md) lists
the usual causes.

## 3. Get a model and a structure

You need a periodic structure (CIF, POSCAR or XYZ) and a MACE checkpoint.
mliport never downloads or bundles weights: fetch
`mace-omat-0-medium.model` from the upstream release (links in
[Models](models.md)) and note where you saved it.

Don't have a structure at hand? The repository ships a 50-atom Li₁₀GeP₂S₁₂
(LGPS) cell at `examples/structures/li10gep2s12_primitive.vasp`. It is a real
periodic cell and small enough for a CPU.

## 4. Run a single point

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --output runs/mace-sp
```

The run prints an energy and force summary and fills `runs/mace-sp/` with
`OUTCAR`, `OSZICAR`, `CONTCAR`, `mliport_results.json`, `resolved_config.json`
and `run.log`. [Output files](outputs.md) explains each of them.

If `runs/mace-sp` already holds results, mliport refuses to write there; pass
`--overwrite` when you really mean to replace them.

## 5. Relax the structure

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --fmax 0.05 --max-steps 50 \
  --output runs/mace-opt
```

Fifty steps may not be enough, and that is fine: the result says so. Whether
the relaxation converged is recorded in `mliport_results.json`, and a run that
did not converge is reported as such rather than dressed up as a success. The
relaxed structure is in `CONTCAR`.

## 6. Read the result

```python
import json

with open("runs/mace-opt/mliport_results.json") as handle:
    record = json.load(handle)

calculation = record["calculation"]
print(calculation["optimization"]["converged"], calculation["results"]["energy"])
```

Everything needed to reproduce the run is next to it: the model identity,
where each configuration value came from, the requested and the actual device,
and the mliport version.

## The other backends

Install the backend you want with `--engines dpa`, `grace` or `uma`, then use
its own environment:

```bash
# DPA — multi-task checkpoints need the branch you intend to use
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --output runs/dpa-sp

# GRACE — the model is a SavedModel directory
.venv-grace/bin/mliport sp structure.cif --model saved_model \
  --model-type grace --output runs/grace-sp

# UMA — say which task family you want
.venv/bin/mliport sp structure.cif --model uma.pt \
  --model-type uma --task omat --output runs/uma-sp
```

There is no default backend. Leaving out `--model-type` (when no model alias
supplies one) is an error, so you always know which model produced a number.

## A complete run on the LGPS cell

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda --output runs/lgps-mace-sp
```

With the MACE-OMAT-0 checkpoint you should see a finite energy of about
−216 eV, finite forces, the formula `Ge2Li20P4S24` (50 atoms) and, when a GPU
was requested, `actual_device_type: cuda` in `mliport_results.json`. The cell
comes from a 2×2×2 LGPS supercell; `examples/structures/README.md` records its
provenance and hashes.

## What next

- Choose a model with care: [Models](models.md).
- Control runs and strictness: [Configuration](configuration.md).
- MD, NEB and batch runs: [Workflows](workflows/index.md).
- Analyse a trajectory: [Analysis](analysis/index.md).
- See how far the claims are backed by tests: [Validation](validation.md).
