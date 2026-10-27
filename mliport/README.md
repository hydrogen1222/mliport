<!-- Generated from the repository root README.md by scripts/generate_docs.py; DO NOT EDIT. -->

# mliport

**English** | [简体中文](https://github.com/hydrogen1222/mliport/blob/main/README_CN.md)

Run machine-learned interatomic potentials — MACE, DPA, GRACE and UMA — from one
command line, one terminal UI and one Python API. Give it a structure and a
checkpoint; it relaxes, runs molecular dynamics, finds NEB barriers, searches adsorbate
configurations on a slab or analyzes a trajectory, and writes the files a VASP user expects: `OUTCAR`, `OSZICAR`,
`CONTCAR`, `XDATCAR`.

mliport is a workflow layer, not a model and not a DFT code. The files are
VASP-shaped, but there is no SCF and no POTCAR, and whether a potential is right for
your chemistry depends on the checkpoint you choose.

What you get:

- **The same workflow on four backends.** Single point, fixed- and variable-cell
  relaxation, NVE/NVT MD (Langevin, Bussi, Nosé–Hoover chain), NEB and CI-NEB,
  MD-based configuration search for an adsorbate on a slab, batch sweeps and a job
  queue, from the CLI, an INCAR-style file, the TUI or Python.
- **No silent guessing.** There is no default backend, and a misspelled option, or
  one that belongs to another backend, stops the run. Configuration errors fail
  closed.
- **A paper trail.** Every run records the model file and its SHA-256, the requested
  and the actual device, the resolved configuration and the mliport version.
- **Analysis that says when the data is too short.** Trajectory validation,
  thermodynamics, RDF, RMSD/RMSF, MSD, VACF, spectra, mobile-ion density, kinisi
  transport, GEMDAT electrolyte analysis and Arrhenius fits.

## Documentation

The full documentation is at <https://hydrogen1222.github.io/mliport/> (English and
中文); the same pages are in [`docs/`](docs/) and read fine on GitHub.

| I want to … | Read |
|---|---|
| run my first calculation | [Quickstart](https://github.com/hydrogen1222/mliport/blob/main/docs/quickstart.md) |
| check a model on a system with a known answer | [Test systems](https://github.com/hydrogen1222/mliport/blob/main/docs/test-systems.md) |
| install one or more backends, on GPU or CPU | [Installation](https://github.com/hydrogen1222/mliport/blob/main/docs/installation.md) |
| understand what MACE, DPA, GRACE and UMA are and pick one | [Backends](https://github.com/hydrogen1222/mliport/blob/main/docs/backends/index.md), [Models](https://github.com/hydrogen1222/mliport/blob/main/docs/models.md) |
| set options through the CLI, an INCAR-style file or `settings.ini` | [Configuration](https://github.com/hydrogen1222/mliport/blob/main/docs/configuration.md) |
| single point, relaxation, MD, NEB, configuration search, batch, queue | [Workflows](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/index.md) |
| MSD, diffusion, conductivity, RDF, spectra, Arrhenius | [Analysis](https://github.com/hydrogen1222/mliport/blob/main/docs/analysis/index.md) |
| know what each output file contains | [Output files](https://github.com/hydrogen1222/mliport/blob/main/docs/outputs.md) |
| fix an error | [Troubleshooting](https://github.com/hydrogen1222/mliport/blob/main/docs/troubleshooting.md) |
| cite mliport and the models | [Citations](https://github.com/hydrogen1222/mliport/blob/main/docs/citations.md) |

## Install

Linux and WSL2. You need `git` and [uv](https://docs.astral.sh/uv/); the installer
makes one Python environment per backend and a launcher for each under `bin/`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you do not have uv yet
git clone https://github.com/hydrogen1222/mliport
cd mliport

# NVIDIA GPU (needs a working CUDA driver)
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto

# or CPU only
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

The installer pins the upstream backend versions, installs the analysis extras and
runs `mliport doctor` for each engine. Python 3.12 is the default for the
four-engine install; the package supports Python 3.10-3.13, and if you pass
`--python`, every backend you request must accept it (UMA needs 3.11 or newer). The
install was verified on a V100 with driver 580.173.02; older and newer GPU families
use other wheel routes, listed in [docs/installation.md](https://github.com/hydrogen1222/mliport/blob/main/docs/installation.md),
together with CPU, offline and proxy setups and uninstalling.

| Engine | Environment | Launcher |
|---|---|---|
| MACE | `.venv-mace` | `bin/mliport-mace` |
| DPA | `.venv-dpa` | `bin/mliport-dpa` |
| GRACE | `.venv-grace` | `bin/mliport-grace` |
| UMA | `.venv` | `bin/mliport-uma` |

Run mliport from the environment of the backend you want, or through its launcher:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
./bin/mliport-mace --help
```

A good result ends with `Environment checks passed for MACE on CUDA.` On Windows each
environment has `Scripts\mliport.exe` and the installer writes `bin\mliport-*.cmd`
launchers. `mliport doctor --json` gives the same checks for scripts.

### Models

mliport does not download checkpoints. Fetch one with the upstream tool and pass its
path with `--model`; a local directory works offline.

| Engine | Checkpoint | Where from |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |

The hashes the validation suite uses are pinned in
[`validation/science/model_manifest.json`](https://github.com/hydrogen1222/mliport/blob/main/validation/science/model_manifest.json).
Licenses belong to the upstream projects; UMA's checkpoint is under a non-commercial
research license. Hugging Face mirrors and SOCKS proxies are covered in
[docs/models.md](https://github.com/hydrogen1222/mliport/blob/main/docs/models.md).

## A first calculation

The repository carries a 50-atom Li10GeP2S12 (LGPS) cell, small enough for a CPU
smoke test:

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda \
  --output runs/lgps-mace-sp
```

In `runs/lgps-mace-sp/` you get `mliport_results.json` (energy, forces, stress,
timing), `resolved_config.json` (what was really used), `OUTCAR`, `CONTCAR`,
`run.log` and, for a relaxation or MD run, `OSZICAR`. A run refuses an output
directory that already holds results and locks it while it works; `--overwrite`
replaces them on purpose. The device in the result is the one the backend reported,
not the one you asked for.

## Choosing a backend

The backend comes from `--model-type` or a configured alias. Without one the run
fails; mliport never falls back to another model on its own.

```bash
.venv-mace/bin/mliport  sp LGPS.vasp --model mace-omat-0-medium.model --model-type mace  --task bulk --device cuda
.venv-dpa/bin/mliport   sp LGPS.vasp --model DPA-3.1-3M.pt --model-type dpa --task bulk --head Omat24 --device cuda
.venv-grace/bin/mliport sp LGPS.vasp --model GRACE-2L-OMAT-medium-base --model-type grace --task bulk --device cuda
.venv/bin/mliport       sp LGPS.vasp --model uma-s-1p2.pt --model-type uma --task omat --device cuda
```

| Engine | `--model-type` | Task / head | Good to know |
|---|---|---|---|
| MACE | `mace` | `bulk` | float64 by default; `--dtype float32` is faster, less precise |
| DPA | `dpa` | `bulk`, `--head Omat24` | multitask checkpoint; a missing head is an error, not a guess |
| GRACE | `grace` | `bulk` | TensorFlow runtime, CPU and GPU |
| UMA | `uma` (`fairchem` is an alias) | `omat` | `--task omol` for molecules; non-commercial research license |

DPA and GRACE cannot change GPU after their framework has loaded, so when
`CUDA_VISIBLE_DEVICES` is unset the CLI restarts the command in a fresh process with
the right GPU visible. From Python, set `CUDA_VISIBLE_DEVICES` before you import
them. More in [docs/backends/index.md](https://github.com/hydrogen1222/mliport/blob/main/docs/backends/index.md).

## The main workflows

**Relaxation.** `--fmax` and `--max-steps` set convergence; `--cell-opt` relaxes the
cell with the ions.

```bash
.venv-mace/bin/mliport opt LGPS.vasp --model model.model --model-type mace \
  --fmax 0.05 --max-steps 200 --cell-opt --device cuda --output runs/lgps-opt
```

**Molecular dynamics.** `--ensemble` is `NVE` or `NVT`, `--thermostat` is `LANGEVIN`,
`BUSSI` or `NHC`. A run can continue from a saved frame that has velocities with
`--velocity-policy preserve --no-pre-relax`. With NHC the default `--com-policy auto`
zeroes the center-of-mass momentum once, so the cell does not drift and bias
MSD-based transport.

```bash
.venv-mace/bin/mliport md LGPS.vasp --model model.model --model-type mace \
  --ensemble NVT --thermostat LANGEVIN --temp 800 --timestep 1.0 \
  --steps 10000 --save-interval 100 --device cuda --output runs/lgps-md
```

**NEB and CI-NEB.** The two endpoint files define the band, `--images` sets how many
intermediates, `--climb` turns on the climbing image. Before the first step mliport
checks, on your own band, whether the model's forces are the gradient of its energy
(noise-aware finite differences) and refuses to go on if they are not; a model whose
predictor reports direct (non-gradient) forces is always refused. The measurement is saved as
`gradient_preflight`. It is a spot check of consistency, not a statement about
accuracy.

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --images 5 --climb \
  --model model.model --model-type mace --device cuda --output runs/lgps-neb
```

Resume with `--resume runs/lgps-neb --output runs/lgps-neb`. The step budgets are
part of the run's identity, so changing `--max-steps` while resuming is rejected.
Advanced explicit atom mapping/image-shift control is available through API/direct
CLI/TUI only.

**Configuration search.** `confsearch` looks for low-energy configurations of one
adsorbate on one side of a periodic slab: many independent starts (your structures,
or random placements with `--slab`/`--molecule`), each run through heat–hold–cool
Langevin cycles at fixed cell, with every snapshot quenched, classified (molecules
whose covalent skeleton broke are set aside) and deduplicated. The result is a ranked
list of the lowest minima found, with saturation evidence and POSCAR/CP2K files for
DFT re-ranking — never a claim of a global minimum. `--dry-run` checks the inputs
and the budget without loading a model.

```bash
.venv-mace/bin/mliport confsearch run --slab slab.vasp --molecule conformers.xyz \
  --n-starts 20 --cycles 5 \
  --model model.model --model-type mace --task bulk --device cuda --output runs/ads
```

Interrupted runs continue with `confsearch resume`, finished ones grow with
`confsearch extend --cycles N`, and comparable runs combine with `confsearch merge`.
Details, defaults and limits: [docs/workflows/confsearch.md](https://github.com/hydrogen1222/mliport/blob/main/docs/workflows/confsearch.md).

**Batch, INCAR-style runs, queue, TUI, Python.**

```bash
.venv-mace/bin/mliport batch structures/ --pattern "*.vasp" --calc-type sp \
  --model model.model --model-type mace --device cuda --output runs/batch

.venv-mace/bin/mliport template sp --output INCAR.mliport     # edit MODEL_PATH, MODEL_TYPE, ...
.venv-mace/bin/mliport run -i INCAR.mliport -s LGPS.vasp -o runs/incar-sp

.venv-mace/bin/mliport queue submit tasks.json                # then: queue start, jobs, queue stop
```

`mliport tui` opens the interactive application (it needs a terminal). From Python:

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("LGPS.vasp", "model.model", model_type="mace", device="cuda")
result = run_single_point("LGPS.vasp", "model.model", model_type="mace", device="cuda")
```

## Analyzing a trajectory

`analyze` takes an mliport run directory or an ASE trajectory:

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

Results go to `runs/lgps-md/analysis/<task>/<id>/` as JSON, CSV and, where it makes
sense, PNG and SVG; plots render headlessly. A short trajectory is accepted but not
over-read: `msd` and `transport` return an explicit insufficient-sampling status
instead of a diffusion coefficient, and `arrhenius` fits only the points you give it.
Options for each task are in [docs/analysis/index.md](https://github.com/hydrogen1222/mliport/blob/main/docs/analysis/index.md).

## Configuration

The backend is always explicit (`--model-type mace|dpa|grace|uma` plus `--model`, or
an alias). Configuration is strict: unknown and cross-backend keys are errors, and
`--lenient-config` downgrades them for exploratory runs. `mliport config` inspects
settings; a project `settings.ini` can define aliases, and `--settings PATH` wins
over the user and project files. Each run writes `resolved_config.json` with the
source of every value.

## Status and limits

This is a beta candidate (2.0.0b4). It has been installed from these instructions on a
clean V100 machine and run through the LGPS workflow and analysis matrix; the block
below is generated from the validation reports.

INCAR-style input covers the common controls, not every interface. Which option each
entry point (API, direct CLI, INCAR, TUI) can express is recorded in
`validation/science/capability_matrix.json`; a `no` or `partial` there means that
entry point cannot say it, and nothing is translated silently.

Model accuracy belongs to the upstream checkpoint. Short MD runs check the workflow,
not equilibration. NEB's pre-flight shows the forces match the energy, which says
nothing about how accurate the barrier is. Transport analyses fail closed with an
insufficient-sampling status instead of returning a number.

The configuration search (`confsearch`) is new: it is covered by synthetic-model
tests in CI but has not yet been through a real-model GPU validation campaign, and
its deduplication tolerances are calibrated on synthetic geometries only. Treat its
ranking as a screening for DFT, not as a result.

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

Scientific campaign target: commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).
LGPS fresh-install acceptance target: commit `51d2a78` ([acceptance report](https://github.com/hydrogen1222/mliport/blob/main/validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)).
Current release: 2.0.0b4 (package metadata; the release tag records the exact commit).

Validated hardware: NVIDIA V100-SXM2-16GB (Volta, 16 GiB), driver 580.173.02, Rocky Linux 9.8. CPU installs are smoke-tested on the same host.

Current status: 2.0.0b4 beta. The four backends are installed and exercised on an LGPS structure through single point, relaxation, MD, NEB, batch, INCAR-style runs, queue, TUI, the Python API and the analysis modules.

- Full beta validation report: [BETA_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md)
- LGPS fresh-install acceptance: [LGPS_ACCEPTANCE_REPORT.md](https://github.com/hydrogen1222/mliport/blob/main/validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- Model identities: [model_manifest.json](https://github.com/hydrogen1222/mliport/blob/main/validation/science/model_manifest.json)

Per-workload tables, tier scopes and limitations live in the reports; the README keeps only this status summary.
<!-- END GENERATED -->

## Citation

See [docs/citations.md](https://github.com/hydrogen1222/mliport/blob/main/docs/citations.md) and [`CITATION.cff`](https://github.com/hydrogen1222/mliport/blob/main/CITATION.cff). If you
use a backend model, cite the upstream model and respect its license.

## License

The mliport package is under the license in [`LICENSE.md`](https://github.com/hydrogen1222/mliport/blob/main/LICENSE.md). Model
checkpoints and the upstream frameworks keep their own licenses.
