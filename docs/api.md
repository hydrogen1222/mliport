# Python API

mliport exposes six high-level functions for running MLIP calculations from
Python scripts. They are the programmatic equivalent of the CLI subcommands
(`sp`, `opt`, `md`, `neb`, `confsearch`) and share the same configuration
resolver, so built-in defaults, model aliases and `settings.ini` profiles
are honoured exactly as they are on the command line.

```python
from mliport.api import (
    calculate_energy,
    run_single_point,
    run_optimization,
    run_md,
    run_neb,
    run_confsearch,
)
```

## When to use the Python API

The CLI is the right tool for one-off calculations and batch scripts. The
Python API is for:

- **Screening loops** — iterate over a list of structures and collect energies
  or forces in a DataFrame.
- **Multi-stage workflows** — relax a structure, then run MD, then analyze, all
  in one script with programmatic branching on results.
- **Jupyter notebooks** — explore results interactively without shelling out.
- **Integration with other codes** — call mliport from an ASE optimizer, a
  genetic algorithm, or a workflow manager.

## Common parameters

All six functions share these parameters:

| Parameter | Type | Description |
|---|---|---|
| `structure` | `Atoms \| str \| Path` | ASE `Atoms` object, or path to a structure file (CIF, POSCAR, XYZ, etc.). |
| `model_path` | `str` | Path to the model checkpoint file. |
| `model_type` | `str \| None` | Backend engine: `"uma"`, `"mace"`, `"dpa"`, `"grace"`, or `"nep"`. `None` lets the resolver decide (from alias or settings). |
| `task` | `str \| None` | Model head / task name (e.g. `"omat"`, `"omol"`, `"bulk"`). |
| `device` | `str \| None` | `"cpu"` or `"cuda"`. `None` = auto-detect. |
| `output_dir` | `str` | Directory for output files. Default is `"./results"`. |
| `job_name` | `str \| None` | Optional name for organizing results within the output directory. |
| `verbose` | `bool` | Print progress to stdout. Default is `True` (except `calculate_energy` which defaults to `False`). |
| `settings_path` | `str \| None` | Explicit path to a `settings.ini` file. |
| `model_alias` | `str \| None` | Named model alias from `settings.ini` `[model:NAME]` section. |
| `profile` | `str \| None` | Reusable profile from `settings.ini` `[profile:NAME]` section. |
| `strict_config` | `bool \| None` | Override strict-config behaviour. |
| `overwrite` | `bool` | Allow overwriting existing output (default `False`). |
| `default_dtype` | `str \| None` | MACE dtype override (`"float32"` or `"float64"`). |
| `head` | `str \| None` | MACE head or DeepMD/DPA multi-task branch name. |
| `inference_mode` | `str \| None` | Inference-mode override (e.g. UMA `"turbo"` is opt-in). |
| `nep_runtime` | `str \| None` | NEP execution runtime: `"ase"` or `"gpumd"`. Required when `model_type="nep"`. |
| `nep_gpumd_command` | `str \| None` | Path to the `gpumd` executable (for `nep_runtime="gpumd"`). |

All functions also accept `**kwargs` for additional options forwarded to the
configuration resolver.

---

## `run_single_point`

Calculate energy, forces and stress for a structure without moving any atoms.

```python
result = run_single_point(
    structure="POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    output_dir="runs/sp",
)
```

**Returns:** `dict` with keys including `energy` (eV), `forces` (eV/Å array),
`stress` (Voigt 6-vector in eV/Å³), and `time` (seconds).

### Example: compare backends on the same structure

```python
from mliport.api import run_single_point

backends = {
    "mace": {"model_path": "mace-omat-0-medium.model", "task": "bulk"},
    "uma":  {"model_path": "uma-s-1p2.pt", "task": "omat"},
}

for name, opts in backends.items():
    result = run_single_point(
        "structure.cif",
        model_type=name,
        output_dir=f"runs/compare-{name}",
        **opts,
    )
    print(f"{name}: {result['energy']:.4f} eV")
```

---

## `run_optimization`

Optimize atomic positions (and optionally cell parameters) until forces
converge below a threshold.

```python
result = run_optimization(
    structure="POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    fmax=0.02,           # eV/Å
    max_steps=500,
    optimizer="FIRE",    # FIRE, BFGS, or LBFGS
    cell_opt=True,       # also relax the cell
    output_dir="runs/opt",
)
```

**Additional parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fmax` | `float \| None` | resolver default | Force convergence threshold (eV/Å). |
| `max_steps` | `int \| None` | resolver default | Maximum optimization steps. |
| `optimizer` | `str \| None` | resolver default | Algorithm: `"FIRE"`, `"BFGS"`, or `"LBFGS"`. |
| `cell_opt` | `bool \| None` | `False` | Whether to optimize cell parameters. |
| `fix_symmetry` | `bool \| None` | `False` | Preserve symmetry during optimization. |

**Returns:** `dict` with keys including `energy`, `converged` (bool),
`nsteps` (int), `forces`, and `stress`.

### Example: relax a list of structures

```python
from pathlib import Path
from mliport.api import run_optimization

structures = Path("candidates").glob("*.cif")
for cif in sorted(structures):
    result = run_optimization(
        cif,
        model_path="mace-omat-0-medium.model",
        model_type="mace", task="bulk",
        fmax=0.02, max_steps=300,
        output_dir=f"runs/opt-{cif.stem}",
    )
    status = "✓" if result["converged"] else "✗"
    print(f"{cif.name}: {result['energy']:.4f} eV  {status}")
```

---

## `run_md`

Run a molecular dynamics simulation.

```python
result = run_md(
    structure="POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    ensemble="NVT",
    temperature=600,        # K
    timestep=1.0,           # fs
    steps=10000,
    thermostat="LANGEVIN",
    save_interval=10,
    output_dir="runs/md",
)
```

**Additional parameters:**

| Parameter | Type | Description |
|---|---|---|
| `ensemble` | `str \| None` | `"NVT"`, `"NVE"`, or `"NPT"`. |
| `temperature` | `float \| None` | Temperature in Kelvin. |
| `timestep` | `float \| None` | Time step in femtoseconds. |
| `steps` | `int \| None` | Number of MD steps. |
| `thermostat` | `str \| None` | NVT thermostat: `"LANGEVIN"`, `"BUSSI"`, or `"NHC"`. |
| `friction` | `float \| None` | Langevin friction coefficient (1/fs). |
| `bussi_tau` | `float \| None` | Bussi/CSVR coupling time (fs). |
| `nhc_tdamp` | `float \| None` | Nosé-Hoover chain damping time (fs). |
| `nhc_tchain` | `int \| None` | Nosé-Hoover chain length. |
| `nhc_tloop` | `int \| None` | Nosé-Hoover thermostat integration substeps. |
| `save_interval` | `int \| None` | Interval for saving trajectory frames. |
| `pre_relax` | `bool \| None` | Pre-relax positions before MD (reduces initial force spikes). |
| `com_policy` | `str \| None` | Center-of-mass policy: `"auto"`, `"none"`, `"initialize_only"`, or `"constraint"`. |

**Returns:** `dict` with keys including `temperature`, `energy`, `time`.

!!! note "NPT ensemble"
    NPT molecular dynamics uses the ASE `IsotropicMTKNPT` integrator and is
    available only for ASE-based backends (MACE, DPA, GRACE, UMA). The NEP
    backend with `gpumd` runtime does not support NPT. NPT adds these
    parameters: `pressure_gpa`, `npt_tdamp`, `npt_pdamp`, `npt_tchain`,
    `npt_pchain`, `npt_tloop`, `npt_ploop`.

### Example: NVT MD with Bussi thermostat

```python
from mliport.api import run_md

result = run_md(
    "relaxed_CONTCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace", task="bulk",
    ensemble="NVT",
    thermostat="BUSSI",
    bussi_tau=100.0,      # 100 fs coupling time
    temperature=800,
    timestep=2.0,
    steps=50000,
    save_interval=50,
    pre_relax=True,
    output_dir="runs/bussi-md",
)
```

---

## `run_neb`

Run a nudged elastic band (NEB) or climbing-image NEB calculation to find the
minimum energy path between two endpoint structures.

```python
result = run_neb(
    initial="initial_POSCAR",
    final="final_POSCAR",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    n_intermediate_images=7,
    climb=True,
    output_dir="runs/neb",
)
```

**Additional parameters:**

| Parameter | Type | Description |
|---|---|---|
| `initial` | `Atoms \| str \| Path \| None` | Initial endpoint structure. Required for new runs. |
| `final` | `Atoms \| str \| Path \| None` | Final endpoint structure. Required for new runs. |
| `resume` | `str \| Path \| None` | Path to a checkpoint directory for resuming. |
| `n_intermediate_images` | `int \| None` | Number of intermediate images along the band. |
| `climb` | `bool \| None` | Enable climbing-image NEB. |
| `method` | `str \| None` | NEB method. |
| `interpolation` | `str \| None` | Interpolation scheme for initial path. |
| `spring` | `float \| None` | Spring constant for NEB (eV/Å²). |
| `fmax` | `float \| None` | Force convergence threshold (eV/Å). |
| `max_steps` | `int \| None` | Maximum optimizer steps. |
| `pre_fmax` | `float \| None` | Pre-optimization force threshold. |
| `pre_max_steps` | `int \| None` | Pre-optimization step limit. |
| `endpoint_policy` | `str \| None` | How to handle endpoint relaxation. |
| `checkpoint_interval` | `int \| None` | Steps between full-band checkpoints. |

**Returns:** `dict` with the NEB band energies, barrier height, convergence
status, and per-image data.

### Example: resume a NEB run

```python
from mliport.api import run_neb

# Resume from the checkpoint in the original output directory
result = run_neb(
    resume="runs/neb",
    output_dir="runs/neb",
)
print(f"Barrier: {result['barrier']:.3f} eV")
```

!!! warning "NEB is fixed-cell"
    mliport's NEB workflow operates at fixed cell. Variable-cell NEB is not
    supported.

---

## `run_confsearch`

Run an MD-based configuration search for adsorbate placement on a slab
surface. The workflow cycles through high-temperature and low-temperature MD
to explore configuration space, then ranks and exports the most stable
configurations found.

```python
result = run_confsearch(
    slab="slab.vasp",
    molecule="molecule.xyz",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    n_starts=5,
    cycles=3,
    output_dir="runs/confsearch",
)
```

**Key parameters:**

| Parameter | Type | Description |
|---|---|---|
| `starts` | `Any` | Pre-built starting configurations (path or list of paths). |
| `slab` | `str \| Path \| None` | Path to the slab structure. |
| `molecule` | `str \| Path \| None` | Path to the adsorbate molecule. |
| `n_starts` | `int \| None` | Number of random starting placements. |
| `adsorbate_indices` | `str \| None` | Atom indices to treat as adsorbate. |
| `cycles` | `int \| None` | Number of heat/cool cycles. |
| `mode` | `str \| None` | Search mode. |
| `temperature_high` | `float \| None` | High-temperature MD target (K). |
| `temperature_low` | `float \| None` | Low-temperature quench target (K). |
| `top_k` | `int \| None` | Number of top configurations to export. |
| `export_window` | `float \| None` | Energy window for export (eV). |
| `refine` | `bool \| None` | Whether to refine top candidates. |
| `dry_run` | `bool` | Validate inputs without running (default `False`). |
| `resume` | `bool` | Resume a previous search (default `False`). |

**Returns:** `dict` with the content of `search_results.json` — ranked
candidates with energies and structures.

!!! note "Screening, not global optimization"
    The ranking is a screening product: it reports the lowest-energy minimum
    found with saturation evidence, not a guaranteed global minimum. The
    exported candidates are meant for DFT re-ranking.

---

## `calculate_energy`

A convenience wrapper that returns just the energy (in eV) as a float. By
default it uses a temporary directory, so repeated calls never collide.

```python
from mliport.api import calculate_energy

energy = calculate_energy(
    "structure.cif",
    "mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
)
print(f"{energy:.4f} eV")
```

**Additional parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `relax` | `bool` | `False` | Pre-relax the structure before calculating energy. |
| `fmax` | `float \| None` | resolver default | Force threshold for pre-relaxation. |
| `max_steps` | `int \| None` | resolver default | Step limit for pre-relaxation. |

**Returns:** `float` — energy in eV.

### Example: energy screening of candidates

```python
from pathlib import Path
from mliport.api import calculate_energy

candidates = sorted(Path("structures").glob("*.cif"))
energies = []
for cif in candidates:
    e = calculate_energy(
        cif,
        "mace-omat-0-medium.model",
        model_type="mace", task="bulk",
    )
    energies.append((cif.name, e))
    print(f"{cif.name}: {e:.4f} eV")

# Sort by energy
for name, e in sorted(energies, key=lambda x: x[1]):
    print(f"{name}: {e:.4f} eV")
```

---

## Configuration resolution

All API functions route through the same `resolve_config()` pipeline as the
CLI. The precedence order (highest wins) is:

1. Explicit keyword arguments to the function
2. CLI-layer overrides (from `**kwargs`)
3. INCAR file values (when applicable)
4. Profile values (`[profile:NAME]` in `settings.ini`)
5. Model alias values (`[model:NAME]` in `settings.ini`)
6. Settings defaults (`[defaults]` in `settings.ini`)
7. Built-in defaults

This means you can set up a `settings.ini` with your common options and
override only what differs per call:

```ini
; settings.ini
[defaults]
device = cuda
strict_config = true

[model:mace-omat]
model_path = /models/mace-omat-0-medium.model
model_type = mace
task = bulk
```

```python
from mliport.api import run_single_point

# model_path, model_type and task come from the alias
result = run_single_point(
    "structure.cif",
    model_path="",   # unused when alias provides it
    model_alias="mace-omat",
    output_dir="runs/sp",
)
```

See [Configuration](configuration.md) for the full details.

## Error handling

All functions raise standard Python exceptions:

- `ValueError` — invalid inputs (missing structure file, incompatible
  options, resume without output_dir).
- `FileExistsError` — output directory already contains results and
  `overwrite=False`.
- Engine-specific errors — model loading failures, CUDA errors, etc.

The functions do **not** catch and silence errors. If a calculation fails, the
exception propagates with a descriptive message.

## Thread safety

Each function call creates its own `CalculationEngine` instance and runs
synchronously. Multiple calls from different threads are safe as long as they
use different output directories. GPU calculations should generally not be
parallelized within a single process — use separate processes instead.
