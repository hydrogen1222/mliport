# mliport

mliport runs MACE, DPA (DeepMD-kit), GRACE (tensorpotential) and UMA
(FAIRChem) machine-learned interatomic potentials behind one command-line
workflow. It reads ASE-readable structures, accepts INCAR-style options and
writes VASP-like files (`OUTCAR`, `OSZICAR`, `CONTCAR`, `XDATCAR`) for people
who already work with POSCAR and OUTCAR.

Every run records the model file and its SHA-256 when available, the requested
and actual device, the resolved configuration and the mliport version.
Configuration errors fail closed by default: a misspelled key, or an option
that belongs to another backend, stops the run instead of being ignored.
The same calculation can be started from the CLI, an INCAR-style file, the
TUI, the background queue or the Python API.

Workflows: single point, fixed- and variable-cell relaxation, MD (NVE and NVT
with Langevin, Bussi or Nose-Hoover chain thermostats), NEB/CI-NEB, batch
sweeps. Analysis: trajectory validation, thermodynamics, RDF, RMSD/RMSF, MSD,
VACF, velocity spectrum, mobile-ion density, kinisi transport, GEMDAT
electrolyte analysis and Arrhenius fits.

## Install

Linux and WSL2 are supported. The installer needs `git` and
[uv](https://docs.astral.sh/uv/); it creates one Python environment per
backend plus a launcher under `bin/`.

```bash
# 1. uv, if it is not installed yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. the repository
git clone https://github.com/hydrogen1222/mliport
cd mliport
```

NVIDIA GPU (a working CUDA driver is required; this path was verified on a
V100-SXM2-16GB with driver 580.173.02):

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

CPU only:

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

The installer pins the upstream backend versions from
`mliport/install/compatibility.py`, installs the analysis, transport and
electrolyte features, and runs `mliport doctor` for every requested engine.
Python 3.12 is the default for the four-engine install. Per-backend minimums
are MACE and GRACE `>=3.9`, DPA `>=3.10` and UMA `>=3.11`; the package itself
supports Python 3.10-3.12. If you override the interpreter with `--python`,
every requested backend must accept it.

### Older GPUs

The installer picks a framework channel per architecture because upstream
support windows differ:

| GPU family | Installer route | Evidence |
|---|---|---|
| Maxwell | cu126, experimental | binary audit only; install-time warning |
| Pascal (P100/P40) | torch 2.8/2.10 + cu126, TF 2.20 | binary audit; no hardware smoke |
| Volta (V100) | torch 2.8/2.10 + cu126, TF 2.20 | real V100 acceptance |
| Turing / Ampere / Ada | cu128 | binary audit; no hardware smoke |
| Hopper / Blackwell | cu128 for torch backends; GRACE experimental | binary audit; no hardware smoke |

PyTorch stops publishing CUDA 12.6 wheels from 2.15, which drops Maxwell,
Pascal and Volta binary support. Do not upgrade the framework in a legacy
environment; the installer pins the audited versions. Wheel hashes, compiled
architectures and per-backend limits are in
[`validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md`](validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md).

Environment layout after a successful four-engine install:

| Engine | Environment | Launcher |
|---|---|---|
| MACE | `.venv-mace` | `bin/mliport-mace` |
| DPA | `.venv-dpa` | `bin/mliport-dpa` |
| GRACE | `.venv-grace` | `bin/mliport-grace` |
| UMA | `.venv` | `bin/mliport-uma` |

Run mliport from the environment of the backend you want, or through its
launcher:

```bash
.venv-mace/bin/mliport --help
./bin/mliport-mace --help
```

Then check the runtime. The command exits non-zero when a required check
fails, and reports optional features separately:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

A successful result ends with a line such as:

```text
 Environment checks passed for MACE on CUDA.
```

On Windows each environment exposes `Scripts\mliport.exe`, and the installer
writes `bin\mliport-*.cmd` launchers next to the POSIX ones. `mliport doctor
--json` prints the same checks as JSON for scripts and cluster provisioning.
Full details for CPU installs, custom sources, offline hosts, proxies, disk
usage and uninstalling are in [docs/installation.md](docs/installation.md).

### Models

mliport does not download model checkpoints. Fetch the checkpoint with the
upstream tool, then pass its path with `--model`; a local directory works
offline.

| Engine | Checkpoint | Where to get it |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

The exact artifact hashes used by the validation suite are pinned in
[`validation/science/model_manifest.json`](validation/science/model_manifest.json).
Checkpoint licenses belong to their upstream projects; UMA's checkpoint is
distributed under a non-commercial research license. `docs/models.md` covers
Hugging Face mirrors and SOCKS proxy settings.

## Run your first calculation

The repository ships a 50-atom Li10GeP2S12 (LGPS) primitive cell at
`examples/structures/li10gep2s12_primitive.vasp`. It is a real periodic cell
and small enough for CPU smoke tests.

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda \
  --output runs/lgps-mace-sp
```

The run writes `runs/lgps-mace-sp/mliport_results.json` (energy, forces,
stress, timing), `resolved_config.json` (what was actually used),
`OUTCAR`/`OSZICAR`/`CONTCAR`, and `run.log`. Energy and forces must be finite;
the device in `mliport_results.json` is the device the backend reported, not
the one that was requested.

## Choose a backend

The backend comes from `--model-type`, a configured alias/profile, or the run
fails closed: mliport never falls back to another model on its own.

```bash
# MACE
.venv-mace/bin/mliport sp LGPS.vasp --model mace-omat-0-medium.model --model-type mace --task bulk --device cuda

# DPA
.venv-dpa/bin/mliport sp LGPS.vasp --model DPA-3.1-3M.pt --model-type dpa --task bulk --head Omat24 --device cuda

# GRACE
.venv-grace/bin/mliport sp LGPS.vasp --model GRACE-2L-OMAT-medium-base --model-type grace --task bulk --device cuda

# UMA
.venv/bin/mliport sp LGPS.vasp --model uma-s-1p2.pt --model-type uma --task omat --device cuda
```

| Engine | Model type | Task/head | Notes |
|---|---|---|---|
| MACE | `mace` | `bulk` | float64 by default; `--dtype float32` is faster and less precise |
| DPA | `dpa` | `bulk`, `--head Omat24` | multitask checkpoint; the branch is read from the artifact |
| GRACE | `grace` | `bulk` | TensorFlow runtime; CPU and GPU commands are available |
| UMA | `uma` (`fairchem` alias) | `omat` | `--task omol` for molecules; license is non-commercial research |

DPA and GRACE cannot move a CUDA context after their framework has loaded.
When `CUDA_VISIBLE_DEVICES` is not already set, mliport restarts the command
in a fresh process with the requested GPU visible and prints a short note.
The Python API cannot restart the caller: set `CUDA_VISIBLE_DEVICES` first if
you call DPA or GRACE from Python. Per-backend manuals:
[docs/backends/index.md](docs/backends/index.md).

## Main workflows

### Relaxation (`opt`)

`--fmax` and `--max-steps` control convergence, and `--cell-opt` relaxes the
cell together with the ions.

```bash
.venv-mace/bin/mliport opt LGPS.vasp --model model.model --model-type mace \
  --fmax 0.05 --max-steps 200 --cell-opt --device cuda --output runs/lgps-opt
```

### Molecular dynamics (`md`)

`--ensemble` selects `NVE` or `NVT`, and `--thermostat` selects `LANGEVIN`,
`BUSSI` or `NHC`. `--temp`, `--timestep`, `--steps` and `--save-interval`
size the run. A trajectory can continue from a saved frame that contains
velocities with `--velocity-policy preserve --no-pre-relax`. NHC requires
`--com-policy none`, because constrained NHC cannot run with automatic
center-of-mass removal.

```bash
.venv-mace/bin/mliport md LGPS.vasp --model model.model --model-type mace \
  --ensemble NVT --thermostat LANGEVIN --temp 800 --timestep 1.0 \
  --steps 10000 --save-interval 100 --device cuda --output runs/lgps-md
```

### NEB and CI-NEB (`neb`)

The endpoint files define the band, `--images` sets the number of
intermediates, and `--climb` enables the climbing image. Current checkpoints
have not had their energy-gradient consistency validated, so the run requires
the explicit experimental opt-in and reports `physical_barrier=not_claimed`:

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --images 5 --climb --allow-unvalidated-neb \
  --model model.model --model-type mace --device cuda --output runs/lgps-neb
```

Resume a stopped band with `--resume runs/lgps-neb --output runs/lgps-neb`.
The recorded step budgets are part of the run identity, so changing
`--max-steps` during a resume is rejected in this beta. Advanced explicit
atom mapping/image-shift control is available through API/direct CLI/TUI only.

### Batch (`batch`)

One directory, one model and one calculation type per sweep.

```bash
.venv-mace/bin/mliport batch structures/ --pattern "*.vasp" --calc-type sp \
  --model model.model --model-type mace --device cuda --output runs/batch
```

### INCAR-style runs (`run`)

Generate a template, edit it, then run it.

```bash
.venv-mace/bin/mliport template sp --output INCAR.mliport
# set MODEL_PATH, MODEL_TYPE, TASK, DEVICE (and HEAD for DPA)
.venv-mace/bin/mliport run -i INCAR.mliport -s LGPS.vasp -o runs/incar-sp
```

### Queue (`queue`, `jobs`, `kill`, `clean`)

Put several calculations in a JSON task file, start the scheduler in the
background, then inspect the jobs.

```bash
.venv-mace/bin/mliport queue submit tasks.json
.venv-mace/bin/mliport queue start
.venv-mace/bin/mliport jobs
.venv-mace/bin/mliport queue stop
```

### TUI

`mliport tui` opens the interactive application. It needs a terminal and
exits with an error when there is none.

### Python API

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("examples/structures/li10gep2s12_primitive.vasp", "model.model", model_type="mace", device="cuda")
result = run_single_point("examples/structures/li10gep2s12_primitive.vasp", "model.model", model_type="mace", device="cuda")
```

## Analyze a trajectory

`analyze` accepts an mliport run directory or an ASE trajectory:

```bash
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md msd --mobile Li
.venv-mace/bin/mliport analyze runs/lgps-md rdf --center Li --neighbor S
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md vacf
.venv-mace/bin/mliport analyze runs/lgps-md spectrum
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
.venv-mace/bin/mliport analyze runs/lgps-md transport --mobile Li --charge 1 --fit-start-ps 100
.venv-mace/bin/mliport analyze runs/lgps-md electrolyte --mobile Li --discover-sites-from-density
```

Analysis results are written under `runs/lgps-md/analysis/<task>/<id>/` as
JSON plus CSV and, where a figure makes sense, PNG and SVG. Plots render
headlessly, so no X11 display is required.

Short trajectories are accepted but not over-interpreted: `msd` and
`transport` return an explicit insufficient-window/insufficient-sampling
status instead of a diffusion coefficient, and `transport` refuses a lag grid
that would exceed its memory guard unless you pass explicit
`--lag-step-ps`/`--lag-stop-ps`. `arrhenius` fits independent
temperature/diffusivity pairs produced by `transport`; it never invents a
value for a failed point. See [docs/analysis/index.md](docs/analysis/index.md)
for the per-task options and semantics.

## Configuration

- Backend selection is explicit: `--model-type mace|dpa|grace|uma` plus
  `--model`, or a configured alias/profile. `fairchem` is accepted as an
  alias for `uma`.
- Strict configuration is the default. Unknown or cross-backend keys are
  fatal; `--lenient-config` downgrades them for exploratory runs.
- `mliport config` inspects and manages settings. A project `settings.ini`
  can define model aliases and profiles; an explicit `--settings PATH` wins
  over both the user and project files.
- Every run writes `resolved_config.json` with the sources of each value, so
  the effective configuration can be audited after the fact.

## Validation and limitations

The current release is a beta candidate (2.0.0b3). It has been installed from
these instructions on a clean V100 host and exercised through the LGPS
workflow and analysis matrix; the generated status block below links the
reports.

INCAR-style input covers the common controls, not every interface. The
expressibility of each option per entry point (API, direct CLI, INCAR, TUI) is
recorded in `validation/science/capability_matrix.json`; a `no` or `partial`
entry means that entry point cannot express the option and no silent
translation happens.

Model accuracy belongs to the upstream checkpoint. Short MD runs verify the
workflow, not equilibration. NEB on the current checkpoints is experimental:
it requires `--allow-unvalidated-neb` and does not claim a physical barrier.
Transport analyses fail closed with an explicit insufficient-sampling status
instead of returning a number.

## Documentation

- [Installation](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Backends](docs/backends/index.md)
- [Workflows](docs/workflows/index.md)
- [Analysis](docs/analysis/index.md)
- [Output files](docs/outputs.md)
- [Configuration](docs/configuration.md)
- [Models and licenses](docs/models.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Validation](docs/validation.md)

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

Validated hardware: NVIDIA V100-SXM2-16GB (Volta, 16 GiB), driver 580.173.02, Rocky Linux 9.8. CPU installs are smoke-tested on the same host.

Current status: 2.0.0b3 beta. The four backends are installed and exercised on an LGPS structure through single point, relaxation, MD, NEB, batch, INCAR-style runs, queue, TUI, the Python API and the analysis modules.

- Full beta validation report: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md)
- LGPS fresh-install acceptance: [LGPS_ACCEPTANCE_REPORT.md](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- Model identities: [model_manifest.json](validation/science/model_manifest.json)

Per-workload tables, tier scopes and limitations live in the reports; the README keeps only this status summary.
<!-- END GENERATED -->

## Citation

See [docs/citations.md](docs/citations.md) and [`CITATION.cff`](CITATION.cff).
If you use a backend model, cite the upstream model and its license terms.

## License

The mliport package is distributed under the license in
[`LICENSE.md`](LICENSE.md). Model checkpoints and upstream frameworks keep their
own licenses.
