# mliport

**English** | [简体中文](README_CN.md)

**mliport** is a unified workflow layer for machine-learned interatomic potentials (MLIPs).
It lets you run five MLIP backends — **MACE**, **DPA**, **GRACE**, **UMA** and **NEP** —
through one command line, one terminal UI, one INCAR-style input file and one Python API.

Give it a structure file and a model checkpoint, and it handles single-point energy
calculations, geometry relaxation, molecular dynamics (NVE / NVT / NPT), NEB barrier
searches, adsorbate configuration searches on slabs, batch sweeps and trajectory analysis.
Every result is written in VASP-style format (`OUTCAR`, `OSZICAR`, `CONTCAR`,
`XDATCAR`) alongside a machine-readable JSON record so you always know exactly how a
number was produced.

> **mliport is a workflow layer, not a model and not a DFT code.**
> The output files look like VASP, but there is no SCF cycle, no k-point mesh and no
> POTCAR. Whether a potential is accurate for your chemistry depends on the checkpoint
> you choose and the validation you do.

📖 **Full documentation:** <https://hydrogen1222.github.io/mliport/>
(English and 中文; the same pages are in [`docs/`](docs/) and read fine on GitHub)

## Why mliport?

| Problem | How mliport solves it |
|---|---|
| Each MLIP backend has its own Python API and CLI | One set of commands works for all five backends |
| Switching backends means rewriting scripts | Change `--model-type` and `--model`; everything else stays the same |
| Hard to tell what configuration was actually used | Every run writes `resolved_config.json` recording every option and its source |
| Misspelled options are silently ignored | Strict mode by default — unknown or cross-backend keys are errors |
| Short trajectories produce misleading diffusion coefficients | Analysis reports an explicit "insufficient sampling" status instead of a bad number |
| Reproducing someone else's calculation | SHA-256 of the model file, actual device, mliport version — all recorded automatically |

## Feature overview

- **Unified workflows** — Single point, relaxation (fixed / variable cell), NVE / NVT / NPT
  molecular dynamics, NEB / CI-NEB, configuration search, batch sweeps, job queue
- **Five backends** — MACE, DPA (DeePMD-kit), GRACE, UMA (Meta / FairChem), NEP (GPUMD / Calorine)
- **VASP-style output** — `OUTCAR`, `OSZICAR`, `CONTCAR`, `XDATCAR`, readable by VESTA, OVITO,
  pymatgen and any VASP post-processing tool
- **Trajectory analysis** — MSD, tracer diffusivity, ionic conductivity (kinisi), RDF, RMSD/RMSF,
  VACF, vibrational spectra, mobile-ion density, GEMDAT electrolyte analysis, Arrhenius fitting
- **NPT dynamics** — Isotropic MTK NPT for ASE-based backends (MACE, DPA, GRACE, UMA)
- **DFT-D3 dispersion** — Optional DFT-D3 correction on any backend
- **Python API** — Six functions for scripting: `run_single_point`, `run_optimization`, `run_md`,
  `run_neb`, `run_confsearch`, `calculate_energy`
- **Provenance** — SHA-256, device, configuration source and version recorded for every run

## Quick navigation

| I want to… | Read |
|---|---|
| Run my first calculation | [Quickstart](docs/quickstart.md) |
| Install one or more backends (GPU or CPU) | [Installation](docs/installation.md) |
| Check a model on a known system | [Test systems](docs/test-systems.md) |
| Understand the five backends and pick one | [Backends](docs/backends/index.md), [Models](docs/models.md) |
| Set options (CLI, INCAR-style, `settings.ini`) | [Configuration](docs/configuration.md) |
| Single point, relaxation, MD, NEB, confsearch, batch, queue | [Workflows](docs/workflows/index.md) |
| MSD, diffusion, conductivity, RDF, spectra, Arrhenius | [Analysis](docs/analysis/index.md) |
| Use mliport from Python | [Python API](docs/api.md) |
| Know what each output file contains | [Output files](docs/outputs.md) |
| Fix an error | [Troubleshooting](docs/troubleshooting.md) |
| Cite mliport and the models | [Citations](docs/citations.md) |

## Install

**Platforms:** Linux and WSL2. **Requirements:** `git` and [uv](https://docs.astral.sh/uv/).

The installer creates one isolated Python environment per backend and a launcher script for
each under `bin/`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you don't have uv yet
git clone https://github.com/hydrogen1222/mliport
cd mliport

# NVIDIA GPU (needs a working CUDA driver)
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto

# or CPU only
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu

# NEP is opt-in (add it explicitly)
./scripts/install_mliport.sh --engines nep --device cpu
```

| Engine | Environment | Launcher |
|---|---|---|
| MACE | `.venv-mace` | `bin/mliport-mace` |
| DPA | `.venv-dpa` | `bin/mliport-dpa` |
| GRACE | `.venv-grace` | `bin/mliport-grace` |
| UMA | `.venv` | `bin/mliport-uma` |
| NEP | `.venv-nep` | `bin/mliport-nep` |

You can also invoke the environment directly (without the launcher):

```bash
.venv-mace/bin/mliport sp structure.cif --model model.model --model-type mace
.venv/bin/mliport sp structure.cif --model uma.pt --model-type uma
# Windows
.venv-mace\Scripts\mliport.exe sp structure.cif --model model.model --model-type mace
```

The installer pins upstream backend versions, installs the analysis extras and runs
`mliport doctor` for each engine. Python 3.12 is the default; mliport supports 3.10-3.13
(UMA needs ≥ 3.11). Verified on a V100 with driver 580.173.02; other GPU families use
different wheel routes — see [Installation](docs/installation.md) for GPU / CPU / offline /
proxy / uninstall details.

<details>
<summary>NEP setup notes</summary>

NEP is not in the default engine list. The installer puts Calorine and an editable mliport
into `.venv-nep`, which is enough for the CPU ASE runtime (`CPUNEP`, float64). The native
runtime and `GPUNEP` also need a `gpumd` executable that the installer does **not** build
(it needs your CUDA toolkit and `-arch=sm_XX`). Use `--device cpu` until you have built
`gpumd`; see [NEP backend](docs/backends/nep.md) for the build instructions.

</details>

### Verify the installation

```bash
./bin/mliport-mace doctor --engine mace --device auto
```

A good result ends with `Environment checks passed for MACE on CUDA.`
(`mliport doctor --json` gives machine-readable output for scripts.)

### Get a model checkpoint

mliport does not download checkpoints. Fetch one from the upstream project and pass its
path with `--model`:

| Engine | Checkpoint | Source |
|---|---|---|
| MACE | `mace-omat-0-medium.model` | [mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_omat_0) |
| DPA | `DPA-3.1-3M.pt` | [deepmodelingcommunity/DPA-3.1-3M](https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M) |
| GRACE | `GRACE-2L-OMAT-medium-base` | [AMS-ICAMS-RUB/grace-foundation-models](https://huggingface.co/AMS-ICAMS-RUB/grace-foundation-models) |
| UMA | `uma-s-1p2.pt` | [facebook/UMA](https://huggingface.co/facebook/UMA) |
| NEP | your own `nep.txt` | trained for your elements and conditions; mliport ships none |

Validation hashes: [`validation/science/model_manifest.json`](validation/science/model_manifest.json).
Licenses belong to the upstream projects; UMA is non-commercial research only. Hugging Face
mirrors and SOCKS proxies: [Models](docs/models.md).

## A first calculation

The repository carries a 50-atom Li₁₀GeP₂S₁₂ (LGPS) cell, small enough for a CPU smoke test:

```bash
./bin/mliport-mace sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda \
  --output runs/lgps-mace-sp
```

In `runs/lgps-mace-sp/` you get:

| File | Contents |
|---|---|
| `mliport_results.json` | Energy, forces, stress, timing, provenance |
| `resolved_config.json` | Every configuration value and its source |
| `OUTCAR` | VASP-style output |
| `CONTCAR` | Final structure |
| `run.log` | Full log |

A run refuses an output directory that already holds results (use `--overwrite` to replace
them). The device in the result is the one the backend actually used, not the one you requested.

## Choosing a backend

The backend comes from `--model-type` (or a configured alias). Without one the run fails —
mliport never falls back to another model.

```bash
./bin/mliport-mace  sp LGPS.vasp --model mace-omat-0-medium.model --model-type mace  --task bulk --device cuda
./bin/mliport-dpa   sp LGPS.vasp --model DPA-3.1-3M.pt --model-type dpa --task bulk --head Omat24 --device cuda
./bin/mliport-grace sp LGPS.vasp --model GRACE-2L-OMAT-medium-base --model-type grace --task bulk --device cuda
./bin/mliport-uma   sp LGPS.vasp --model uma-s-1p2.pt --model-type uma --task omat --device cuda
./bin/mliport-nep   sp LGPS.vasp --model nep.txt --model-type nep --nep-runtime ase --device cpu
```

| Engine | `--model-type` | Task / head | Notes |
|---|---|---|---|
| MACE | `mace` | `bulk` | float64 by default; `--dtype float32` is faster |
| DPA | `dpa` | `bulk`, `--head Omat24` | Multitask checkpoint — must specify head |
| GRACE | `grace` | `bulk` | TensorFlow runtime; CPU and GPU |
| UMA | `uma` (alias: `fairchem`) | `omat` | `--task omol` for molecules; non-commercial research license |
| NEP | `nep` + `--nep-runtime ase\|gpumd` | `bulk` (PBC only) | Runtime never inferred; `ase` = Calorine, `gpumd` = native |

DPA and GRACE cannot change GPU after loading. Set `CUDA_VISIBLE_DEVICES` before import.
See [Backends](docs/backends/index.md) for details.

## Main workflows

**Relaxation** — minimize forces, optionally also the cell:

```bash
./bin/mliport-mace opt LGPS.vasp --model model.model --model-type mace \
  --fmax 0.05 --max-steps 200 --cell-opt --device cuda --output runs/lgps-opt
```

**Molecular dynamics** — NVE, NVT (Langevin / Bussi / NHC) or NPT (ASE backends only):

```bash
./bin/mliport-mace md LGPS.vasp --model model.model --model-type mace \
  --ensemble NVT --thermostat LANGEVIN --temp 800 --timestep 1.0 \
  --steps 10000 --save-interval 100 --device cuda --output runs/lgps-md
```

**NEB / CI-NEB** — minimum-energy path with gradient pre-flight check:

```bash
./bin/mliport-mace neb --initial initial.vasp --final final.vasp \
  --images 5 --climb \
  --model model.model --model-type mace --device cuda --output runs/lgps-neb
```

**Configuration search** — low-energy adsorbate configurations on a slab:

```bash
./bin/mliport-mace confsearch run --slab slab.vasp --molecule conformers.xyz \
  --n-starts 20 --cycles 5 \
  --model model.model --model-type mace --task bulk --device cuda --output runs/ads
```

**Batch, queue, TUI and Python:**

```bash
./bin/mliport-mace batch structures/ --pattern "*.vasp" --calc-type sp \
  --model model.model --model-type mace --device cuda --output runs/batch

./bin/mliport-mace queue submit tasks.json       # then: queue start, jobs, queue stop
./bin/mliport-mace tui                           # interactive terminal UI
```

```python
from mliport.api import calculate_energy, run_single_point

energy = calculate_energy("LGPS.vasp", "model.model", model_type="mace", device="cuda")
result = run_single_point("LGPS.vasp", "model.model", model_type="mace", device="cuda")
```

See [Workflows](docs/workflows/index.md) for the full reference.

## Analyzing a trajectory

`analyze` takes an mliport run directory or an ASE trajectory:

```bash
./bin/mliport-mace analyze runs/lgps-md validate
./bin/mliport-mace analyze runs/lgps-md thermo
./bin/mliport-mace analyze runs/lgps-md msd --mobile Li
./bin/mliport-mace analyze runs/lgps-md rdf --center Li --neighbor S
./bin/mliport-mace analyze runs/lgps-md transport --mobile Li --charge 1 --fit-start-ps 100
./bin/mliport-mace analyze runs/lgps-md electrolyte --mobile Li --discover-sites-from-density
./bin/mliport-mace analyze runs/lgps-md vacf
./bin/mliport-mace analyze runs/lgps-md spectrum
./bin/mliport-mace analyze runs/lgps-md density --mobile Li
```

Results go to `runs/lgps-md/analysis/<task>/<id>/` as JSON, CSV, PNG and SVG. A short
trajectory is accepted but not over-read: `msd` and `transport` return an explicit
insufficient-sampling status instead of a diffusion coefficient, and `arrhenius` fits only
the points you give it. See [Analysis](docs/analysis/index.md) for the full reference.

## Configuration

The backend is always explicit (`--model-type` plus `--model`, or an alias; `nep` also
needs `--nep-runtime`). Configuration is strict: unknown and cross-backend keys are errors.
`--lenient-config` downgrades them for exploratory runs. `mliport config` inspects settings;
a project `settings.ini` can define aliases, and `--settings PATH` overrides the user and
project files. Every run writes `resolved_config.json` recording the source of each value.
See [Configuration](docs/configuration.md).

## Status and limits

This is a beta candidate (2.0.0b5). It has been installed from these instructions on a
clean V100 machine and run through the LGPS workflow and analysis matrix; the block
below is generated from the validation reports.

INCAR-style input covers the common controls, not every interface. Advanced explicit atom mapping/image-shift control is available through API/direct CLI/TUI only. Which option each
entry point (API, direct CLI, INCAR, TUI) can express is recorded in
`validation/science/capability_matrix.json`; a `no` or `partial` there means that
entry point cannot say it, and nothing is translated silently.

NEP limits: you supply the `nep.txt`, and mliport cannot say whether it fits your
material. The native `gpumd` runtime has no CPU path and refuses NEB, batch and
confsearch instead of switching to the ASE runtime; DFT-D3 is rejected; `qNEP`,
`TNEP` and `nep4-vdw` models are unsupported; partial PBC is refused. NEP is not part
of the validation campaign summarised below: its real-hardware evidence is a V100 with
GPUMD 5.9.1 and CUDA 12.9, and other GPUs and toolkits are unverified. Details:
[docs/backends/nep.md](docs/backends/nep.md).

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
LGPS fresh-install acceptance target: commit `51d2a78` ([acceptance report](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)).
Current release: 2.0.0b5 (package metadata; the release tag records the exact commit).

Validated hardware: NVIDIA V100-SXM2-16GB (Volta, 16 GiB), driver 580.173.02, Rocky Linux 9.8. CPU installs are smoke-tested on the same host.

Current status: 2.0.0b5 beta. The four backends are installed and exercised on an LGPS structure through single point, relaxation, MD, NEB, batch, INCAR-style runs, queue, TUI, the Python API and the analysis modules.

- Full beta validation report: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md)
- LGPS fresh-install acceptance: [LGPS_ACCEPTANCE_REPORT.md](validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- Model identities: [model_manifest.json](validation/science/model_manifest.json)

Per-workload tables, tier scopes and limitations live in the reports; the README keeps only this status summary.

<!-- END GENERATED -->

**Known limits:**

- INCAR-style input covers the common controls, not every option.
- NEP: you supply the `nep.txt`; mliport cannot say whether it fits your material. The native
  `gpumd` runtime has no CPU path and refuses NEB, batch and confsearch. Not part of the
  validation campaign. See [NEP](docs/backends/nep.md).
- Model accuracy belongs to the upstream checkpoint.
- Configuration search (`confsearch`) is new: CI-tested with synthetic models, not yet
  GPU-validated with real models. Treat its ranking as a screening for DFT.

## Citation

See [Citations](docs/citations.md) and [`CITATION.cff`](CITATION.cff). If you use a backend
model, cite its upstream paper and respect its license.

## License

mliport is under the license in [`LICENSE.md`](LICENSE.md). Model checkpoints and upstream
frameworks keep their own licenses.
