# mlipx

mlipx runs UMA, MACE, DPA and GRACE machine-learning interatomic
potentials through one VASP-shaped CLI/TUI/Python workflow for single
points, relaxation, molecular dynamics, NEB and trajectory analysis.

Scope and boundary, stated directly:

- mlipx evaluates learned potential-energy surfaces (PES). It does not
  perform DFT electronic-structure calculations.
- "VASP-shaped" refers to the workflow and output conventions (INCAR-style
  configuration, OUTCAR/OSZICAR/CONTCAR/XDATCAR files). It describes
  interface familiarity, not physical equivalence to VASP.
- Energies, forces and stresses come from the selected model. Their
  accuracy is the model's accuracy on your chemistry, not mlipx's.

License: MIT. Status: beta validation completed on one GPU architecture
(V100, sm_70) and on CPU; see
[Validation](#validation) below.

## What it can run

| Task | Command | Notes |
|---|---|---|
| Single point | `mlipx sp` | energy, forces, stress (stress if the model provides it) |
| Ionic relaxation | `mlipx opt` | fixed cell, FIRE/LBFGS/BFGS |
| Cell + ionic relaxation | `mlipx opt` | FrechetCellFilter; requires model stress + 3D PBC |
| NVE MD | `mlipx md` | velocity Verlet |
| NVT MD | `mlipx md` | Langevin, Bussi, Nosé-Hoover chain |
| NEB / CI-NEB | `mlipx neb` | fixed cell, IDPP pre-relaxation, two-stage climbing image |
| Batch | `mlipx batch` | many structures through one model load |
| Trajectory analysis | `mlipx analyze` | validate, thermo, rdf, rmsd, msd, vacf, spectrum, transport, density, arrhenius, GEMDAT mechanisms |
| INCAR-driven runs | `mlipx run -i INCAR.mlipx` | the VASP-shaped entry point |
| Queue | `mlipx queue submit/start`, `mlipx jobs` | background job execution |

Everything ASE can read works as a structure input (POSCAR/CONTCAR, CIF,
EXTXYZ, ...). Output formats are VASP-compatible text files plus JSON.

## What it cannot replace

These require a DFT code or are absent by design:

| Property | Status in mlipx |
|---|---|
| Electronic band structure | not available |
| Density of states | not available |
| Charge density / Bader / ELF | not available |
| Born charges | not available |
| Dielectric response | not available |
| k-point / ENCUT / SCF convergence | not applicable (no SCF) |
| NPT molecular dynamics | not implemented |

## Installation

The tested path is the installer, which builds one isolated environment
per backend (the four stacks have mutually exclusive dependencies):

```bash
git clone https://github.com/hydrogen1222/mlipx
cd mlipx
./scripts/install_mlipx.sh --engines mace dpa grace uma --device auto
```

What the installer does:

- Detects the GPU architecture from the driver and pins compatible
  framework builds per backend (Volta/V100 uses torch 2.8.0+cu126;
  Turing and newer use cu128 builds).
- Creates one virtual environment per engine in the repository root
  (`.venv-mace/`, `.venv-dpa/`, `.venv-grace/`, `.venv-uma/`).
- Targets Python 3.10-3.12 (selected via `uv`; the default is 3.12).
- Downloads model checkpoints on first use; `--source` controls wheel
  sources, and offline/source builds are supported.
- Requires roughly 10-15 GB of disk for all four backends including
  torch and model weights.
- Runs `mlipx doctor` at the end unless `--skip-doctor` is given.

Useful flags: `--dry-run` (print the plan, install nothing),
`--non-interactive`, `--clean` (rebuild environments), `--python 3.10`.

Then install the mlipx CLI itself into your working environment:

```bash
pip install ./mlipx
```

<details>
<summary>Manual installation (per backend)</summary>

Each backend environment needs the mlipx package plus the engine's own
stack. The authoritative version pins live in
`mlipx/mlipx/install/compatibility.py`; the installer is the only path
that keeps them consistent with your GPU architecture. If you install
manually, at minimum verify your torch build against your compute
capability, then run `mlipx doctor` and confirm every check passes
before trusting results.

</details>


## GPU architecture compatibility

The installer and `mlipx setup` choose the correct PyTorch/CUDA wheel
automatically.

| GPU family | Examples | Compute capability | CUDA route |
|---|---|---|---|
| Maxwell | GTX 960, TITAN X | sm_50/52 | cu126 Legacy (experimental) |
| Pascal | Tesla P40, GTX 1080 Ti, P100 | sm_60/61 | cu126 Legacy |
| Volta | V100 | sm_70 | cu126 Legacy |
| Turing | RTX 20xx | sm_75 | cu128+ Modern |
| Ampere | RTX 3080 Ti, 30xx | sm_80/86 | cu128+ Modern |
| Ada | RTX 4090, 40xx | sm_89 | cu128+ Modern |
| Hopper | H100 | sm_90 | cu128+ Modern |
| Blackwell | RTX 50xx | sm_100/120 | cu128+ Modern |
| none | CPU only | - | CPU wheels |

Why two CUDA routes: Maxwell/Pascal/Volta must use the cu126 legacy
channel because PyTorch 2.8+ removed Maxwell/Pascal from cu128 builds
and PyTorch 2.11+ removed Volta from cu128+. Turing and newer use the
modern channel (cu128 for torch 2.8-2.10, cu130 for torch 2.12+).
Maxwell is experimental because official TensorFlow 2.20 wheels start
at sm_60.

**Architecture compatibility** (from `mlipx/install/compatibility.py`; this
describes the install route only - upstream package support, the pinned
backend version, and the CUDA wheel channel - *not* workload
certification). "Needs runtime smoke test" means the installer contract
is consistent for that GPU family but mlipx has not yet verified that
exact engine + framework + GPU combination; "experimental" means
upstream itself does not support or test it. Install-route smoke tests
(engine installed, real model prediction) have been run on real V100 and
RTX 4090 hardware; workload-level evidence exists only for the V100
runtime below. P40 uses the corrected exact `+cu126` wheel pin but still
needs a post-fix model smoke retest.

| Engine | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| MACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| DPA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| GRACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | experimental |

## First calculation

With a UMA environment installed and a structure file present:

```bash
mlipx sp POSCAR --model uma-s-1p2.pt --model-type UMA \
    --task omat --device cuda --output ./results
```

This writes `OUTCAR` (energy, forces, stress, model identity, device
evidence) and a JSON record into `./results`. The same calculation in
INCAR form:

```bash
mlipx template -o INCAR.mlipx sp   # then edit MODEL_PATH/TASK/DEVICE
mlipx run -i INCAR.mlipx
```

Configuration precedence: explicit CLI flags override the INCAR file,
which overrides `settings.ini`, which overrides defaults. `mlipx config
show` prints the fully resolved configuration including where every
value came from.

## Choosing a model

The four beta-validated profiles, all evaluated on a common OMat24
subset in the validation suite:

| Backend | Validated profile | Training domain | Task/head | Precision | Stress | Access |
|---|---|---|---|---|---|---|
| MACE | `mace-omat-0-medium.model` | OMat24 + MPtrj | bulk | float64 (also float32) | yes | meta-predictions download (CC BY 4.0) |
| DPA | `DPA-3.1-3M.pt`, branch `Omat24` | OMat24 + MPtrj | `--head Omat24` | upstream-defined | yes | DeepModeling share |
| GRACE | `GRACE-2L-OMAT-medium-base` | OMat24 | - | upstream-defined (fp32 build) | yes | open download (Intel) |
| UMA | `uma-s-1p2.pt` | OMat + others, multi-head | `--task omat` | upstream-defined | yes | fairchem, meta download |

These are the *common* profiles used across the whole validation suite.
Domain-specific checkpoints (e.g. MACE models fitted to a specific
chemistry, DPA branches other than Omat24) can be loaded the same way,
but their beta validation coverage is not implied by this table, and
their benchmark numbers must not be compared to the OMat24-family
results above.

> Energies from models trained to different reference calculations are
> not on a common thermodynamic energy scale. Do not mix absolute
> energies across models, tasks/heads, or reference levels. Relative
> energies within one model/one task (formation energies against that
> model's own elemental references, barriers from the same model) are
> the safe currency.

## Workflows

### Single point

`mlipx sp` runs one structure through the calculator and writes
energy/forces/stress. NaN/inf energies abort before any output file is
written.

### Relaxation

`mlipx opt` relaxes ionic positions at fixed cell (FIRE, LBFGS or BFGS).
With model stress available and a 3D periodic cell, it relaxes cell and
ions together through ASE's `FrechetCellFilter`. Convergence is reported
as final fmax plus step counts; the CLI prints both initial and final
volumes for cell relaxation.

### Molecular dynamics

`mlipx md` supports NVE (velocity Verlet) and NVT with three
thermostats: Langevin, Bussi stochastic velocity rescaling, and
Nosé-Hoover chain. Trajectories are written as XDATCAR plus JSON with
per-frame energies. The integration timestep and the save interval are
independent; analysis commands read the save interval from trajectory
metadata, so a save interval that is too coarse silently degrades
transport analysis rather than corrupting it.

### NEB / CI-NEB

`mlipx neb` computes minimum-energy paths with a fixed cell:

- Endpoints can be validated as-is or pre-relaxed (`--endpoint-policy
  validate|relax`).
- Initial path: linear interpolation with periodic-image (winding)
  handling, or IDPP pre-relaxation.
- Two-stage climbing image: standard NEB to a force threshold, then
  CI-NEB to convergence.
- Checkpoints are written during the run; a restarted run resumes
  geometry from the checkpoint with endpoint identity preserved.
- Reported barrier: forward and reverse barrier from the sampled band
  (`barrier_forward` / `barrier_reverse` in the JSON record).

> A converged CI image is a saddle-point candidate until a Hessian
> validates it. Use the saddle Hessian diagnostic (one imaginary mode
> expected along the reaction coordinate) before quoting a transition
> state.

### Analysis

`mlipx analyze` operates on mlipx trajectories or external ones with an
explicit coordinate convention (`--positions-convention`). The
hierarchy for diffusion problems:

1. `msd` — windowed, direction-resolved MSD diagnostic.
2. `transport` — quantitative tracer diffusion via kinisi
   (posterior D with credible intervals) and Nernst-Einstein
   conductivity from ionic charges; requires a fixed cell (NVT/NVE).
3. `electrolyte` — GEMDAT site mapping, jump mechanisms and percolation
   as a mechanism-level crosscheck of the transport picture.

Uncertainty semantics: kinisi reports posterior means with 95%
credible intervals; the Nernst-Einstein conductivity is exact only in
the dilute, uncorrelated-hopping limit. The Haven ratio between tracer
and collective transport is not assumed.

## Scientific semantics

- **Energy**: eV, total for the cell as read. Per-atom values are
  labelled per atom in outputs.
- **Forces**: eV/Å.
- **Stress**: ASE Voigt order (`xx, yy, zz, yz, xz, xy`); written in
  eV/Å³ with a GPa conversion line in OUTCAR.
- **Task/head identity**: the model's task or head (e.g. UMA `omat`,
  DPA `Omat24`) selects the reference-energy level inside the model.
  It is recorded in every output and must not be changed between
  energies you intend to subtract.
- **Force-energy consistency**: forces are analytic derivatives of the
  model energy. The beta suite cross-checks them against finite
  differences; per-model results are in the validation report.
- **Stress-energy consistency**: stress is the analytic strain
  derivative where the model provides it, cross-checked by finite
  difference in the beta suite.
- **PBC and winding**: minimum-image conventions are applied with
  explicit winding handling in NEB interpolation and displacement
  analysis; analysis commands reject external trajectories whose
  coordinate convention conflicts with artifact metadata.
- **Constraints**: ASE constraints (FixAtoms, FixSymmetry) are honored
  in relaxation and MD; constraint exactness is regression-tested
  (fixed-atom DOF remain fixed to machine precision).
- **MD timestep**: you choose it. The beta suite runs a timestep sweep
  per system (0.25-2.0 fs on Cu) and gates on NVE energy-drift slopes;
  results in the validation report show which timesteps were stable.
  There is no global "safe" timestep.
- **Save interval vs timestep**: analysis operates on saved frames; the
  saved-frame interval must be short enough to resolve the process you
  are measuring. `mlipx analyze validate` checks this and reports
  insufficient sampling rather than returning a number.
- **Fixed-cell requirement for transport**: kinisi transport analysis
  requires NVT/NVE trajectories. NPT is not implemented, and analysis
  of a barostatted trajectory is therefore not offered.

## Validation

The numbers below are rendered from the beta evidence records; they are
not handwritten. The generated block between the markers is produced by
`validation/science/scripts/generate_beta_report.py` from
`validation/science/reports/beta-summary.json`, and a repository test
fails if the README block drifts from the generator output.

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Validation status per backend, rendered from the beta evidence records (`beta-summary.json`; t1-t8 tiers, 4 backends x OMat24 common subset). `software_validated` means the mlipx integration and all recorded checks passed; `model_characterized` means the workflow ran and its behavior was recorded, including honest failures (e.g. float32 arithmetic noise). Full per-test tables: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md).

| Workflow | MACE | DPA | GRACE | UMA |
|---|---|---|---|---|
| Install & doctor (CI software tests) | software_validated* | software_validated* | software_validated* | software_validated* |
| Single-point inference (4 structures) | software_validated (passx23) | software_validated (passx23) | software_validated (passx23) | software_validated (passx23) |
| Energy-forces consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx8) | model_characterized (characterizedx4) |
| Stress (finite-difference cross-check) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx8, passx3) | model_characterized (characterizedx4, passx3) |
| Stress-energy consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx8) | model_characterized (characterizedx4) |
| Coordinate invariance & cache | model_characterized (characterizedx4, passx20) | model_characterized (characterizedx4, failx18, passx2) | model_characterized (characterizedx8, failx18, passx22) | model_characterized (characterizedx4, failx18, passx2) |
| Fixed-cell relaxation | software_validated (passx12) | software_validated (passx12) | software_validated (passx12) | software_validated (passx12) |
| Cell relaxation | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) |
| EOS / bulk modulus | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Elastic constants | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) |
| Harmonic phonons | software_validated (passx12) | model_characterized (failx1, passx11) | software_validated (passx12) | model_characterized (failx1, passx11) |
| Harmonic thermodynamics | model_characterized (failx6, passx6) | model_characterized (failx3, passx9) | model_characterized (failx6, passx6) | model_characterized (failx6, passx6) |
| Vacancy formation energy | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Surface energy | software_validated (passx6) | software_validated (passx6) | software_validated (passx6) | software_validated (passx6) |
| NEB | model_characterized (characterizedx1, passx7) | model_characterized (characterizedx1, passx7) | model_characterized (characterizedx1, passx7) | model_characterized (characterizedx1, passx7) |
| Saddle-point Hessian | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) |
| Short NVE / NVT MD | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) |
| Transport analysis (demonstration) | model_characterized (characterizedx3, passx3) | model_characterized (characterizedx3, passx3) | model_characterized (characterizedx3, passx3) | model_characterized (characterizedx3, passx3) |
| Mechanism analysis (GEMDAT) | model_characterized (characterizedx3) | model_characterized (characterizedx3) | model_characterized (characterizedx3) | model_characterized (characterizedx3) |
| Performance (SP/MD scaling) | software_validated (passx5) | software_validated (passx5) | software_validated (passx5) | software_validated (passx5) |

Held-out OMat24 accuracy (E/atom MAE over 256 structures, no elemental offsets fitted): MACE (float64) 0.0172 eV; DPA 0.0184 eV; GRACE 0.0143 eV; UMA 0.0108 eV. Stress parity is not computable: the official OMat24 validation split carries no reference stress labels.

`*` = CI software test only, no model involved. A cell lists the recorded statuses for that workload; per-workload rows reuse the same evidence tiers, so row counts are not additive. Full per-test tables and limitations: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md). Model identities pinned in `validation/science/model_manifest.json`.
<!-- END GENERATED -->

What the tiers mean: t1 four-backend inference on the common subset,
t2 invariance and caching, t2fd finite-difference stress, t3 OMat24
label comparison, t4 static workflows (relaxation, EOS, elastic,
phonons, thermo, defects, surfaces), t5 NEB, t6 MD, t7 analysis, t8
performance. Full per-test tables, including every recorded failure:
[validation/science/reports/BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md).

Known limitations recorded by the suite (not hidden): upstream-float32
builds (DPA, UMA, GRACE-cache-off) show 1e-7..1e-6 eV arithmetic noise
on coordinate-invariance checks; small-supercell phonons on the 2x2x2
grid show small imaginary acoustic branches; cohesive and formation
energies are recorded as unsupported because they need elemental
reference energies outside the common comparison.


**Installer-contract runtime smoke (single V100-SXM2-16GB, sm_70, driver
580.173.02)** - short real-model runs with hard timeouts on one GPU, each
against the *exact* runtime the installer produces. Sanitized records
live in [validation/runtime/v100/](validation/runtime/v100/) (the GPU is
identified only by a SHA-256 of its UUID); each status maps to exactly
one record. The MACE records were revalidated on the current installer
contract (`mace-torch 0.3.16` + `torch 2.8.0+cu126`); earlier torch
2.6.0+cu124 evidence remains in git history.

| Backend / model | SP | Short MD | E–F gradient | NEB smoke | GRACE cache | Records |
|---|---|---|---|---|---|---|
| UMA | not run | not run | not run | not run | n/a | [uma.json](validation/runtime/v100/uma.json) (no offline wheelhouse on this machine) |
| MACE float64 | passed | passed | passed | passed | n/a | [mace.json](validation/runtime/v100/mace.json) |
| MACE float32 | passed | passed | passed | passed | n/a | [mace-float32.json](validation/runtime/v100/mace-float32.json) |
| DPA `Domains_Alloy` | passed | passed | passed | passed | n/a | [dpa.json](validation/runtime/v100/dpa.json) |
| GRACE float32 | passed | passed | passed | passed | passed (ON and OFF) | [grace.json](validation/runtime/v100/grace.json), [grace-nocache.json](validation/runtime/v100/grace-nocache.json) |

The NEB smoke is a short 5-image fixed-cell run on a 4-atom Cu cell with
`saddle_validation: not_performed` - see [NEB / CI-NEB](#neb--ci-neb) for
what that implies. This installer smoke evidence predates the beta
suite; the beta suite is the workload-level evidence base for the
models as configured in [Choosing a model](#choosing-a-model).

## DFT-style validation recipes

The validation suite contains EOS, elastic-constant, harmonic phonon,
thermodynamic, defect and surface recipes built on the same calculators
and ASE. They are beta validation recipes with reproducible scripts
(`validation/science/scripts/`), not stable CLI commands; the CLI does
not expose them as first-class subcommands.

## Outputs and reproducibility

A run directory contains:

```
results/
├── OUTCAR        # energy/forces/stress, model identity, device, versions
├── OSZICAR       # per-step log (relaxation, MD)
├── CONTCAR       # final structure (relaxation)
├── XDATCAR       # trajectory (MD)
└── *.json        # machine-readable record incl. provenance
```

Every record carries: model path + SHA-256, task/head, dtype, device
(requested vs actual, GPU UUID hash), mlipx/framework versions, git
commit and scientific suite revision, and the result schema version.
NEB and MD runs write checkpoints; analysis results are keyed by a
request hash so identical requests are served from cache and changed
requests recompute.

## Python API

The public surface (`mlipx.api`):

```python
from mlipx.api import calculate_energy, run_single_point

energy = calculate_energy("POSCAR", model_path="mace-omat-0-medium.model",
                          model_type="MACE", device="cpu")
results = run_single_point("POSCAR", model_path="mace-omat-0-medium.model",
                           model_type="MACE", output_dir="./results")
```

Also exported: `run_optimization`, `run_md`, `run_neb`. Every snippet
above executes in the repository's test suite; snippets that drift from
the real signatures fail CI.

## Configuration precedence

CLI flags > INCAR file > `settings.ini` > built-in defaults. Path
provenance is printed by `mlipx config show`; `mlipx config schema`
lists every recognized key. The TUI (`mlipx tui`) exposes the same
configuration space interactively.

## Troubleshooting

Observed, diagnosable failures:

- **Model download/auth fails** (UMA, MACE meta checkpoints): the
  download requires network access and, for some checkpoints, accepting
  the license. The error from the downloader is passed through.
- **Wrong DPA branch**: DPA-3.1-3M is multi-head. Without `--head`/TASK
  matching the training branch, results are silently from the wrong
  head. The record's `head` field shows what was used; the installer
  profile pins `Omat24`.
- **Model lacks stress**: cell relaxation, elastic recipes and NPT-style
  analysis require stress. mlipx fails closed with an explicit message
  instead of fabricating stress.
- **GPU architecture mismatch** (e.g. Volta with a cu128-only torch
  build): `mlipx doctor` reports the compute capability and the
  installer pins the matching build; a manual install that ignores this
  fails at first kernel launch.
- **GRACE memory**: the GRACE build is the most memory-hungry of the
  four on large cells; batch runs should lower `--batch-size` before
  assuming a bug.
- **Capability evidence mismatch**: the README tables and the curated
  capability registry are cross-checked by tests; a runtime contract
  change without revalidation fails CI rather than shipping stale
  claims.
- **Insufficient transport sampling**: `mlipx analyze transport`
  reports when the trajectory is too short or the save interval too
  coarse for the fit window; it does not return a number in that case.

## Limitations

- Learned PES quality is domain-dependent; surfaces, defects and
  transition states can be out-of-distribution for any model. The beta
  suite records per-model behavior on a fixed recipe set; it does not
  certify your chemistry.
- No electronic structure (see the table above).
- Absolute energies are not interchangeable across models, tasks/heads
  or reference levels.
- Physical transport numbers need much longer trajectories than the
  beta demonstration runs; the suite labels its transport results
  `demonstration_not_converged`.
- No NPT ensemble.
- Model predictive uncertainty is not implemented.
- Beta validation covers one GPU architecture (V100, sm_70) and CPU. It
  does not prove every GPU architecture; use `mlipx doctor` on your
  hardware before trusting a first run.

## Credits

- [FAIR-Chem / UMA](https://github.com/facebookresearch/fairchem)
- [MACE](https://github.com/ACEsuit/mace)
- [DeePMD-kit / DPA](https://github.com/deepmodeling/deepmd-kit)
- [GRACE](https://github.com/intel/grace)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [kinisi](https://github.com/bjmorgan/kinisi)
- [GEMDAT](https://github.com/GEMDAT-repos/GEMDAT)
- [OMat24 / Meta](https://ai.meta.com/blog/open-source-climate-modeling/)

This project (`hydrogen1222/mlipx`) is unrelated to the other project
also named `mlipx` on PyPI.
