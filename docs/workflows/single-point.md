# Single-point calculation

A single-point calculation evaluates the **energy**, **forces** and
(when the model supports it) **stress tensor** of one structure without
moving any atoms. It is the most fundamental workflow in mliport: every
other workflow — relaxation, MD, NEB — calls a single point at its core.

## What it computes

Given a structure and a model, a single point computes:

- **Total energy** $E$ in eV — the model's prediction for this atomic
  configuration. This is not a DFT total energy; it is defined by whichever
  reference the model was trained against.
- **Forces** $\mathbf{F}_i = -\nabla_i E$ in eV/Å — the negative gradient of
  the energy with respect to each atom's position. These are what relaxation
  and MD integrate.
- **Stress tensor** $\sigma_{\alpha\beta}$ in eV/Å³ (if the model provides it)
  — the derivative of the energy with respect to strain. Needed for cell
  relaxation and NPT dynamics.

A single point is the cheapest way to check that a model, a structure and a
device all work together before committing to a longer calculation.

## Command line

```bash
mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --output runs/sp
```

The structure can be anything ASE reads: CIF, XYZ, POSCAR, extxyz, etc. It
must match what the model expects — a periodic crystal needs a bulk task
(`omat` for UMA, `bulk` elsewhere), an isolated molecule needs a molecular
task (`omol` or `molecule`).

### With different backends

```bash
# DPA with a specific head
mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda:0 \
  --output runs/dpa-sp

# GRACE
mliport sp structure.cif --model saved_model \
  --model-type grace --output runs/grace-sp

# UMA
mliport sp structure.cif --model uma-s-1p2.pt \
  --model-type uma --task omat --output runs/uma-sp

# NEP via Calorine ASE calculator
mliport sp structure.cif --model nep.txt \
  --model-type nep --nep-runtime ase --device cpu \
  --output runs/nep-sp
```

## Python API

```python
from mliport.api import run_single_point, calculate_energy

# Full single-point result
results = run_single_point(
    structure="structure.cif",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    output_dir="runs/sp",
)
print(f"Energy: {results['energy']:.4f} eV")
print(f"Max force: {max(abs(f).max() for f in results['forces']):.4f} eV/Å")

# Just the energy (uses a temp directory, cleaned up automatically)
energy = calculate_energy(
    "structure.cif",
    "mace-omat-0-medium.model",
    model_type="mace",
)
```

`run_single_point` accepts an ASE `Atoms` object or a file path. It returns
a dictionary with `energy`, `forces`, `stress` (if available), and timing
information. `calculate_energy` is a convenience wrapper that returns only
the energy as a float.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model` | *(required)* | Path to the model checkpoint |
| `--model-type` | *(required)* | Backend engine: `mace`, `dpa`, `grace`, `uma` or `nep` |
| `--task` | backend-dependent | Task family: `bulk`, `molecule`, `omat`, `omol`, etc. |
| `--head` | *(auto or required)* | DPA branch or MACE head when the checkpoint has several |
| `--device` | `cpu` | `cpu`, `cuda`, `gpu` or `cuda:N`; both requested and actual device are recorded |
| `--charge`, `--spin` | unset | Electronic-state metadata for molecules (UMA `omol`) |
| `--inference-mode` | backend default | Backend-specific precision and inference settings |
| `--dtype` | backend default | Floating-point precision (`float32` or `float64`) |
| `--write-outcar` | `true` | Write a VASP-style OUTCAR |
| `--write-stress` | `true` | Include stress in the output (if the model provides it) |
| `--overwrite` | `false` | Allow replacing results already in the output directory |

## Output files

In the output directory you will find:

| File | Contents |
|---|---|
| `OUTCAR` | VASP-compatible output with energy, forces and stress |
| `CONTCAR` | A copy of the input structure (nothing moved) |
| `mliport_results.json` | Machine-readable results: energy, forces, stress, timing |
| `resolved_config.json` | Every option that was active and where it came from |
| `run_context.json` | Model identity, device, mliport version |
| `run.log` | Human-readable log of the calculation |

See [Output files](../outputs.md) for the conventions shared by all workflows.

## Error handling

mliport fails fast and loud:

- A model that cannot be loaded stops with a clear error — mliport never
  substitutes a different model or a default backend.
- A multi-task DPA checkpoint without `--head` stops and lists the available
  heads.
- A task that does not fit the backend (e.g. `molecule` on a backend that only
  does bulk) is rejected.
- Energy or forces that are not finite (NaN or infinity) abort the run instead
  of writing corrupt results.

Exit code `1` means the calculation failed; exit code `0` means it completed
and the results are in the output directory.

## Interpreting the result

A single-point energy can only be compared with another single point from
the **same model identity** (same model, head, dtype, task). Absolute
energies are not physical observables across different models — they
differ by an arbitrary reference. See [Models](../models.md) for details
on model selection and what "model identity" means.

If you need a quick energy difference — say, between two polymorphs — run
single points on both with the same model and compare:

```python
from mliport.api import calculate_energy

e_alpha = calculate_energy("alpha.cif", "mace-omat-0-medium.model", model_type="mace")
e_beta  = calculate_energy("beta.cif",  "mace-omat-0-medium.model", model_type="mace")
print(f"ΔE = {(e_beta - e_alpha) * 1000:.1f} meV")
```
