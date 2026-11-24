# CLI reference

mliport is a command-line tool. Every calculation starts with
`mliport <command>`, where the command selects the workflow. This page
lists all commands and their most important options. For detailed
physics and worked examples, see each workflow's own page.

## Global options

These appear before the command name:

```bash
mliport [--version] [--verbose] <command> ...
```

| Flag | Meaning |
|---|---|
| `--version` | Print version and exit |
| `--verbose` | Increase log detail |

## Calculation commands

### `mliport run`

Run a calculation from an INCAR-style input file. Everything — model,
structure, calc type, and all options — lives in one file.

```bash
mliport run --incar INCAR.mliport --output runs/my-calc
```

| Option | Required | Description |
|---|---|---|
| `--incar` | yes | Path to the INCAR-style parameter file |
| `--output` | yes | Output directory |
| `--overwrite` | — | Allow overwriting existing output |

### `mliport sp`

Single-point energy, forces, and stress.

```bash
mliport sp structure.vasp --model model.pt --model-type mace --task bulk \
  --output runs/sp-001
```

Key options:

| Option | Default | Description |
|---|---|---|
| `--model` | — | Path to model file (required) |
| `--model-type` | — | Backend: `uma`, `mace`, `dpa`, `grace`, `nep` (required) |
| `--task` | — | Model head: `omat`, `omol`, `oc20`, etc. |
| `--head` | — | DPA multi-task head |
| `--device` | `auto` | `auto`, `cuda`, `cpu`, `cuda:N` |
| `--output` | — | Output directory (required) |
| `--write-outcar` | `true` | Write VASP-style OUTCAR |
| `--write-forces` | `true` | Include forces in output |
| `--write-stress` | `true` | Include stress tensor |

### `mliport opt`

Geometry optimization (atomic positions; optionally cell).

```bash
mliport opt structure.vasp --model model.pt --model-type mace --task bulk \
  --fmax 0.01 --output runs/opt-001
```

| Option | Default | Description |
|---|---|---|
| `--optimizer` | `FIRE` | Algorithm: `FIRE`, `BFGS`, `LBFGS` |
| `--fmax` | 0.05 | Force convergence in eV/Å |
| `--max-steps` | 500 | Maximum optimization steps |
| `--cell-opt` | off | Also optimize cell shape/volume |
| `--fix-symmetry` | off | Preserve space-group symmetry during relaxation |

### `mliport md`

Molecular dynamics.

```bash
mliport md structure.vasp --model model.pt --model-type mace --task bulk \
  --ensemble NVT --temp 600 --steps 50000 --output runs/md-001
```

| Option | Default | Description |
|---|---|---|
| `--ensemble` | `NVT` | `NVT`, `NVE`, or `NPT` |
| `--temp` | 300 | Temperature in K |
| `--timestep` | 1.0 | Timestep in fs |
| `--steps` | 1000 | Number of MD steps |
| `--thermostat` | `LANGEVIN` | `LANGEVIN`, `BUSSI`, `NHC` |
| `--friction` | 0.001 | Langevin friction in 1/fs |
| `--bussi-tau` | 1000 | Bussi/CSVR coupling time in fs |
| `--nhc-tdamp` | 100 | NHC thermostat damping time in fs |
| `--save-interval` | 10 | Save every *n*-th step |
| `--raw-forces-storage` | `json` | `json` or `binary`: how the raw (unprojected) forces are stored in `raw/trajectory.traj` (see [Output files](outputs.md#molecular-dynamics)) |
| `--equilibration-steps` | 0 | Steps before production begins |
| `--pre-relax` | auto | Quick position relaxation before MD |

**NPT-only options** (requires UMA, MACE, GRACE, or DPA):

| Option | Default | Description |
|---|---|---|
| `--pressure-gpa` | 0.0 | Target pressure in GPa |
| `--npt-tdamp` | 100 | MTK thermostat damping in fs |
| `--npt-pdamp` | 1000 | MTK barostat damping in fs |

### `mliport neb`

Nudged elastic band for transition-state search.

```bash
mliport neb initial.vasp final.vasp --model model.pt --model-type mace \
  --task bulk --output runs/neb-001
```

| Option | Default | Description |
|---|---|---|
| `--n-images` | 7 | Intermediate images |
| `--climb` | off | Enable climbing-image NEB |
| `--neb-method` | `improvedtangent` | NEB method |
| `--fmax` | 0.03 | Force convergence in eV/Å |
| `--max-steps` | 1000 | Maximum NEB steps |
| `--endpoint-policy` | `validate` | `validate`, `relax`, or `none` |

### `mliport confsearch`

Configuration search (adsorbate placement + multi-cycle anneal + quench).

```bash
mliport confsearch run slab.vasp adsorbate.xyz \
  --model model.pt --model-type mace --task bulk --output runs/cs-001
```

Subcommands: `run`, `place`, `resume`, `extend`, `merge`, `export`.

### `mliport batch`

Batch single-point or optimization on many structures.

```bash
mliport batch sp structures/*.cif \
  --model model.pt --model-type mace --task bulk --output runs/batch-sp
```

| Option | Default | Description |
|---|---|---|
| `--resume` | off | Skip already-computed structures |
| `--max-workers` | auto | Parallel workers |

## Analysis

```bash
mliport analyze <run-directory-or-trajectory> <task> [options]
```

The source (run directory or trajectory file) comes **before** the task name.

| Task | Description | Key required options |
|---|---|---|
| `validate` | Trajectory metadata and eligibility | — |
| `thermo` | Temperature, energy drift, pressure | — |
| `rdf` | Radial distribution function | `--center`, `--neighbor` |
| `rmsd` | Periodic-displacement RMSD/RMSF | — |
| `msd` | Mean squared displacement | `--mobile` |
| `transport` | Tracer diffusivity and NE conductivity | `--mobile`, `--charge`, `--fit-start-ps` |
| `density` | 3-D number-density map | `--mobile` |
| `arrhenius` | Arrhenius fit from D(T) data | `--temperature` (×N), `--diffusivity` (×N) |
| `electrolyte` | GEMDAT site analysis | `--mobile` |
| `vacf` | Velocity autocorrelation | — |
| `spectrum` | Vibrational spectrum from VACF | — |

See [Analysis](analysis/index.md) for full details on each task.

## Utility commands

### `mliport config`

Manage configuration. Subcommands: `show`, `validate`, `resolve`.

### `mliport template`

Generate an INCAR template for a given calc type.

```bash
mliport template md > INCAR.mliport
```

### `mliport doctor`

Check installation and hardware.

```bash
mliport doctor --engine mace --device auto
```

| Option | Description |
|---|---|
| `--engine` | Probe a specific backend |
| `--device` | Probe a specific device |
| `--json` | Machine-readable JSON output |

### `mliport setup`

Install backends into isolated virtual environments.

```bash
mliport setup --engines mace uma --device auto
```

### `mliport queue`

Background job management. Subcommands: `submit`, `status`, `cancel`, `log`.

### `mliport jobs`

List background jobs.

### `mliport kill`

Kill a running background job by ID.

### `mliport clean`

Remove intermediate files from a completed run.

### `mliport xdatcar`

Convert between trajectory formats (XDATCAR ↔ ASE .traj).

### `mliport state`

Inspect or modify run state. Subcommands: `show`, `reset`.
