<!-- Generated from the repository root README.md by scripts/generate_docs.py; DO NOT EDIT. -->

# mliport

mliport runs MACE, DPA (DeepMD-kit), GRACE (tensorpotential) and UMA
(FAIRChem) machine-learned interatomic potentials through one
backend-neutral CLI, TUI and Python API. It provides VASP-shaped inputs and
outputs for single points, relaxation, molecular dynamics, NEB and trajectory
analysis — with explicit provenance and honest, fail-closed configuration.

**mliport is not a model, not a trainer and not a DFT code.** It runs
third-party potentials; model accuracy is not mliport accuracy.

## Why mliport

- **Backend-neutral by construction.** No implicit UMA default: the backend
  comes from `--model-type`, a model alias/profile, or the run fails closed.
- **Honest provenance.** Every run records model identity (path, task, head,
  dtype, SHA-256 when available), requested vs actual device, resolved
  configuration sources and the software commit.
- **VASP-shaped, not VASP.** INCAR-style keys and OUTCAR/OSZICAR/CONTCAR/
  XDATCAR files for familiarity, with no SCF, k-points, ENCUT or numerical
  VASP equivalence.
- **Strict science config.** Misspelled or cross-backend options are fatal by
  default (`--lenient-config` is the explicit opt-in).
- **Evidence over adjectives.** Validation claims are tied to recorded
  campaigns, commits and model identities — including `not_run`.

## What it cannot replace

DFT reference calculations, model training/fine-tuning, k-point/ENCUT
convergence studies, transition-state frequency verification, or the
scientific judgement of whether a potential is valid for your chemistry.
See [Scientific scope](#scientific-scope).

## Features

| Area | What you get |
|---|---|
| Backends | MACE, DPA, GRACE, UMA (+ `fairchem` alias for UMA) |
| Calculations | single point (`sp`), relaxation (`opt`), MD (`md`), NEB (`neb`), batch (`batch`), `run` from INCAR-style files |
| Analysis | `msd`, `transport`, `electrolyte` (GEMDAT), `rdf`, `rmsd`, `vacf`, `spectrum`, `arrhenius`, `thermo`, `density`, `validate` |
| Interfaces | CLI, Textual TUI (`tui`), Python API |
| Operations | local `queue`, `jobs`, `kill`, `clean`, `doctor`, `setup`, `config`, `template` |
| Outputs | VASP-shaped files, ASI trajectory, JSON result + artifact manifest |

## Installation

mliport is installed **inside each backend's own Python environment**; it is
not a cross-environment dispatcher. The supported path is the installer:

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

Python support: core CI covers **3.10-3.12**; each backend has its own
constraint (UMA needs >=3.11), and `--python` must satisfy every requested
backend or the installer fails closed. Environment names, Python ranges and
runtime pins are machine-generated in
[docs/installation.md](https://github.com/hydrogen1222/mliport/blob/main/docs/installation.md).

Run the CLI from the environment (or launcher) of the backend you want:

```bash
.venv-mace/bin/mliport sp structure.cif --model model.model --model-type mace
.venv/bin/mliport      sp structure.cif --model uma.pt     --model-type uma
./bin/mliport-mace     ...   # generated launcher, runtime selection only
```

On Windows each environment exposes `Scripts\mliport.exe`. See
[docs/installation.md](https://github.com/hydrogen1222/mliport/blob/main/docs/installation.md) for CPU-only, all-backend,
offline and manual options.

## 5-minute quick start

MACE is used first only because its checkpoints are open and light; the
other backends are equivalent.

```bash
# 1. doctor (fails closed if the runtime cannot really run)
.venv-mace/bin/mliport doctor --engine mace --device auto

# 2. single point
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --output runs/mace-sp

# 3. short relaxation
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/mace-opt
```

Equivalent first runs:

```bash
.venv-dpa/bin/mliport   sp structure.cif --model DPA-3.1-3M.pt --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model GRACE-2L-OMAT-medium-base --model-type grace
.venv/bin/mliport       sp structure.cif --model uma-s-1p2.pt --model-type uma --task omat
```

Then inspect `runs/mace-opt/mliport_results.json` and
`runs/mace-opt/resolved_config.json`; see
[docs/outputs.md](https://github.com/hydrogen1222/mliport/blob/main/docs/outputs.md) for every file, unit and provenance field.

## Choose a backend

| Backend | Environment | Pinned example profile | Precision / head notes |
|---|---|---|---|
| MACE | `.venv-mace` | `mace-omat-0-medium.model` (`mace_omat`) | float64 primary; float32 is a separate identity; `HEAD` optional |
| DPA | `.venv-dpa` | `DPA-3.1-3M.pt` (`dpa_omat`) | multi-task checkpoints need an explicit `--head` (e.g. `Omat24`) |
| GRACE | `.venv-grace` | `GRACE-2L-OMAT-medium-base` (`grace_omat`) | `--model` is the SavedModel directory |
| UMA | `.venv` | `uma-s-1p2.pt` (`uma_omat`) | explicit UMA task family (`omat`/`omol`/...); `fairchem` is an alias |

Per-backend details (profiles, stress/head semantics, citations,
limitations): [docs/backends/](https://github.com/hydrogen1222/mliport/blob/main/docs/backends/mace.md).

## Main workflows

- [Single point](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/single-point.md)
- [Relaxation](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/optimization.md)
- [Molecular dynamics](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/md.md)
- [NEB / CI-NEB](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/neb.md)
- [Batch](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/batch.md)
- [Queue](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/queue.md)
- [Analysis overview](https://github.com/hydrogen1222/mliport/blob/main/docs/analysis/overview.md)

## Scientific scope

- No SCF, no k-points/ENCUT/POTCAR, no DFT numerical equivalence.
- Absolute energies are model-specific; formation energies and barriers need
  one model/head/reference combination.
- OMat24 benchmark results characterise a pinned model on that benchmark and
  do not automatically extrapolate to other chemistry.
- MD-derived transport values depend on sampling; analyses record windows,
  drift policy and uncertainties.
- A converged NEB band gives a saddle *candidate*, not a verified transition
  state.

See [docs/models.md](https://github.com/hydrogen1222/mliport/blob/main/docs/models.md) and [docs/validation.md](https://github.com/hydrogen1222/mliport/blob/main/docs/validation.md).

## Validation status

Release status: **beta candidate**.

Advanced explicit atom mapping/image-shift control is available through
API/direct CLI/TUI only; the INCAR-style key set deliberately does not expose
`neb_atom_map`/`neb_image_shifts`. The machine-readable capability matrix,
including these expressibility states, is
`validation/science/capability_matrix.json`.

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

The full scientific beta campaign targets commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`. The release candidate `6e1a945f91c7` retains that validated scientific implementation and adds productization/release-layer changes. A four-backend release-bridge smoke (`20260913-release-bridge-6e1a945f`) was executed at the release-candidate commit to verify configuration, backend selection, model loading and single-point inference paths. No long scientific trajectories were regenerated.

Validation status per backend, rendered from the beta evidence records (`beta-summary.json`; t1-t8 tiers, 4 backends x OMat24 common subset). `software_validated` means the mliport integration and all recorded checks passed; `model_characterized` means the workflow ran and its behavior was recorded, including honest failures (e.g. float32 arithmetic noise). Full per-test tables: [BETA_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md).

| Workflow | MACE | DPA | GRACE | UMA |
|---|---|---|---|---|
| Install & doctor (CI software tests) | software_validated* | software_validated* | software_validated* | software_validated* |
| Single-point inference (4 structures) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Energy-forces consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) |
| Stress (finite-difference cross-check) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) |
| Stress-energy consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) |
| Coordinate invariance & cache | not_run | not_run | not_run | not_run |
| Fixed-cell relaxation | not_run | not_run | not_run | not_run |
| Cell relaxation | not_run | not_run | not_run | not_run |
| EOS / bulk modulus | not_run | not_run | not_run | not_run |
| Elastic constants | not_run | not_run | not_run | not_run |
| Harmonic phonons | not_run | not_run | not_run | not_run |
| Harmonic thermodynamics | not_run | not_run | not_run | not_run |
| Vacancy formation energy | not_run | not_run | not_run | not_run |
| Surface energy | not_run | not_run | not_run | not_run |
| NEB | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Saddle-point Hessian | not_run | not_run | not_run | not_run |
| Short NVE / NVT MD | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) |
| Transport analysis (demonstration) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) |
| Mechanism analysis (GEMDAT) | model_characterized (characterizedx1) | model_characterized (characterizedx1) | model_characterized (characterizedx1) | model_characterized (characterizedx1) |
| Performance (SP/MD scaling) | software_validated (passx5) | software_validated (passx5) | software_validated (passx10) | software_validated (passx5) |

Held-out OMat24 accuracy: not run on this evidence set.

`*` = CI software test only, no model involved. A cell lists the recorded statuses for that workload; per-workload rows reuse the same evidence tiers, so row counts are not additive. Full per-test tables and limitations: [BETA_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md). Model identities pinned in `validation/science/model_manifest.json`.
<!-- END GENERATED -->

## Python API

```python
from mliport import run_single_point, run_optimization, run_md, run_neb
from mliport import calculate_energy

result = run_single_point(
    structure="structure.cif",
    model="mace-omat-0-medium.model",
    model_type="mace",
)
```

The backend is explicit in the API too; the same resolver and provenance
rules apply as for the CLI.

## Documentation

Full documentation lives in [docs/](https://github.com/hydrogen1222/mliport/blob/main/docs/index.md): installation,
configuration, models/backends, workflows, analysis, outputs, validation,
troubleshooting and the [mlipx migration guide](https://github.com/hydrogen1222/mliport/blob/main/docs/migration-from-mlipx.md).

## Citation

mliport has no DOI of its own yet. Cite mliport (see
[docs/citations.md](https://github.com/hydrogen1222/mliport/blob/main/docs/citations.md)) **and** the upstream model/software
you actually ran; upstream DOIs never belong to mliport. Machine-readable
metadata: [CITATION.cff](https://github.com/hydrogen1222/mliport/blob/main/CITATION.cff).

## License

MIT. See [LICENSE.md](https://github.com/hydrogen1222/mliport/blob/main/LICENSE.md). Upstream models and frameworks keep their
own licenses; check them before redistribution or commercial use.

## Credits

mliport (`hydrogen1222/mliport`) is an independent project and is unrelated
to the other project also named `mliport` on PyPI.
