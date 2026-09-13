# Output files and provenance

mliport writes VASP-shaped files for familiarity, plus machine-readable
provenance. It is **not** VASP and its files are not numerically
VASP-equivalent.

## Common layout

Single point / relaxation (directly in the run directory):

```text
OUTCAR                  VASP-like energy/force/stress report
OSZICAR                 per-step energy/force trace
CONTCAR                 final structure (input copy for SP)
XDATCAR                 trajectory in VASP layout
mliport_results.json    result payload + model metadata
resolved_config.json    resolved configuration with per-key sources
trajectory.traj         canonical ASE trajectory (when written)
```

MD adds subdirectories and time series:

```text
vasp/OUTCAR, vasp/OSZICAR, vasp/CONTCAR, vasp/XDATCAR
raw/md.csv              per-frame scalars (energy, temperature, ...)
raw/trajectory.traj     canonical trajectory
checkpoints/            restart checkpoints (interval controlled by the run)
artifacts.json          artifact manifest: units, observables, hashes
resolved_config.json
```

NEB writes `neb_results.json`, `run_context.json`, `resolved_config.json`,
band/checkpoint artifacts and the per-image trajectories.

Analysis writes under `<run>/analysis/<analysis_id>/`:
`analysis.json`, arrays/plots, and the provenance of the analysis itself.

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

MD `artifacts.json` declares the unit and observable definitions explicitly,
including `forces_raw_eV_A` vs `forces_applied_eV_A` (constraint-applied) and
configurational vs total stress.

## Provenance

`resolved_config.json` records, for every resolved key, the value and the
layer it came from (built-in/settings/alias/profile/INCAR/CLI), so a run can
be reproduced or audited without guessing.

Result/context records include:

- **model identity**: `model_type`, path, task, head, dtype, backend and
  framework versions, `model_sha256` when available;
- **requested vs actual device**: what the user asked for and what the
  backend actually used (including the physical GPU UUID when available);
- **software commit**: the mliport revision that produced the run;
- **hashes**: input structure/model/artifact hashes used by analysis and
  validation evidence.

## VASP-shaped, not VASP

- No SCF, no k-points, no ENCUT/POTCAR concepts.
- Absolute energies are model-specific; the output does not make them DFT
  numbers.
- `OUTCAR` is a report for humans and tooling; the machine-readable truth is
  in the JSON artifacts.

## Requested vs actual device

A run requested on `cuda:0` may report an actual MIG/visible index or UUID.
Use the recorded actual device when writing methods sections; if it says
`unknown`, do not claim a specific GPU for that record.
