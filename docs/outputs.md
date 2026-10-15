# Output files

Each run leaves two kinds of files. The first kind looks like VASP's, so existing
habits and tools carry over. The second kind is JSON that records what actually
happened, and it is the one to trust when the two disagree.

mliport is not VASP and its files are not numerically VASP-equivalent: there is no
SCF, no k-point mesh, no ENCUT or POTCAR, and the absolute energies belong to the
model that produced them, not to a DFT calculation. `OUTCAR` is a report for people
and for tools that read VASP output; the machine-readable truth is in the JSON.

## Single point and relaxation

Written directly into the output directory:

```text
OUTCAR                  VASP-like report of energy, forces and stress
OSZICAR                 per-step energy and force trace (relaxation only)
CONTCAR                 final structure (a copy of the input for a single point)
mliport_results.json    results plus the model's identity
resolved_config.json    the configuration, with the source of every value
run.log                 live log of the run
```

`mliport_results.json` keeps the physics under `calculation.results` (energy,
energy per atom, forces and their statistics, stress and pressure), the structure
under `calculation.system`, and for a relaxation the convergence record under
`calculation.optimization` (`converged`, `nsteps`, `fmax_threshold`).

A run takes a lock on its output directory while it works, and refuses a directory
that already holds results (an earlier run, or VASP's own `OUTCAR` / `CONTCAR`)
unless you pass `--overwrite`.

## Molecular dynamics

MD separates the lossless data from the VASP-style exports:

```text
raw/trajectory.traj     canonical ASE trajectory
raw/md.csv              per-frame scalars: energy, temperature, ...
raw/mliport_results.json
vasp/OUTCAR, vasp/CONTCAR, vasp/XDATCAR
artifacts.json          manifest: files, units, observable definitions, hashes
resolved_config.json
run.log
```

`artifacts.json` states its units and what each observable means, including
`forces_raw_eV_A` against `forces_applied_eV_A` (after constraints) and
configurational against total stress. An ASE trajectory from any run can be turned
into a standard XDATCAR with `mliport convert-xdatcar`.

## NEB

```text
neb_results.json        energies, forces, convergence, barrier numbers
run_context.json        the run's identity, including gradient_preflight
resolved_config.json
artifacts.json
checkpoints/            complete-band checkpoints (band.traj + checkpoint.json)
vasp_path/              one POSCAR per image, for VASP-style NEB tools
```

## Analysis

`mliport analyze` writes under `<run>/analysis/<task>/<analysis-id>/`: JSON, CSV
and, where a figure makes sense, PNG and SVG, together with the provenance of the
analysis itself. See [Analysis](analysis/index.md).

## Units

| Quantity | Unit |
|---|---|
| time | fs |
| length | Å |
| energy | eV |
| force | eV/Å |
| stress | eV/Å³ |
| pressure | GPa |
| temperature | K |

## Provenance

`resolved_config.json` records, for every resolved key, its value and the layer it
came from (built-in default, settings, alias, profile, INCAR or command line). A
run can be reproduced or audited from it without guessing.

The result and context records add:

- **the model's identity**: `model_type`, path, task, head, dtype, backend and
  framework versions, and `model_sha256` when available;
- **the device**, requested and actually used, including the physical GPU UUID
  when it is known;
- **the software commit** of mliport that produced the run;
- **hashes** of the input structure, model and artifacts, which analysis and
  validation evidence rely on.

### Requested versus actual device

A run requested on `cuda:0` may report a different visible index or a UUID as the
actual device. Use the recorded actual device in a methods section. If it says
`unknown`, don't claim a specific GPU for that record.
