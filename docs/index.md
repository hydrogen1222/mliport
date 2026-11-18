# mliport

mliport is a **workflow layer** for machine-learned interatomic potentials. It
lets you run five MLIP backends — MACE, DPA, GRACE, UMA and NEP — through one
command line, one INCAR-style input file, one terminal UI and one Python API.
Hand it a structure and a checkpoint, and it does the rest: single-point
calculations, geometry relaxation, molecular dynamics (NVE, NVT and NPT), NEB
barrier searches, adsorbate configuration searches on slabs, batch sweeps
and trajectory analysis. Every result is written in VASP-compatible format
(`OUTCAR`, `OSZICAR`, `CONTCAR`, `XDATCAR`) alongside a machine-readable JSON
record of exactly what was run.

mliport is **not** a model, not a trainer and not a DFT code. The output files
look like VASP output, but there is no SCF cycle, no k-point mesh and no
POTCAR. Whether a potential is accurate for your chemistry depends entirely on
the checkpoint you choose and the validation you run on it.

## Design principles

**One workflow, every backend.**
The same `mliport sp`, `mliport opt`, `mliport md` and `mliport neb` commands
work for all five backends. You switch models by changing `--model-type` and
`--model`; nothing else in your workflow needs to change. This makes it
straightforward to compare backends on the same structure, or to try a new
model on a calculation you have already set up.

**No silent guessing.**
There is no default backend. If you misspell an option, use one that belongs to
a different backend, or forget to specify the head for a multi-task DPA
checkpoint, the run stops with a clear error. Configuration is strict by
default — mliport never silently ignores an unknown key or falls back to
another model.

**Reproducibility built in.**
Every run writes a `resolved_config.json` that records the model file path and
its SHA-256 hash, which device was requested vs. actually used, where each
configuration value came from (default, settings.ini, INCAR file, or CLI), and
the mliport version. You can always reconstruct how a result was produced.

**Analysis that knows its limits.**
The analysis pipeline — MSD, tracer diffusivity, ionic conductivity, RDF, VACF,
vibrational spectra, GEMDAT electrolyte analysis, Arrhenius fitting — reports an
explicit "insufficient sampling" status when a trajectory is too short for a
reliable result, rather than returning a number you should not trust.

## Feature overview

| Feature | Description |
|---|---|
| Unified workflow | Single point, relaxation (fixed/variable cell), NVE/NVT/NPT MD, NEB/CI-NEB, configuration search, batch, job queue |
| Five backends | MACE, DPA (DeepPotential), GRACE, UMA (Meta), NEP (GPUMD) |
| VASP-compatible output | `OUTCAR`, `OSZICAR`, `CONTCAR`, `XDATCAR` — readable by any tool that parses VASP files |
| INCAR-style input | Write an `INCAR.mliport` file with familiar VASP-like syntax; run it with `mliport run` |
| Strict configuration | Unknown keys, cross-backend keys and type mismatches are errors, not warnings |
| NPT molecular dynamics | Isotropic MTK NPT ensemble for ASE-based backends (MACE, DPA, GRACE, UMA) |
| DFT-D3 dispersion | Optional DFT-D3 correction on any backend |
| Trajectory analysis | MSD, diffusivity, conductivity, RDF, VACF, spectra, density, GEMDAT, Arrhenius |
| Python API | Six functions (`run_single_point`, `run_optimization`, `run_md`, `run_neb`, `run_confsearch`, `calculate_energy`) for scripting |
| Model provenance | SHA-256, device, configuration source and version recorded for every run |

## Quick start

The repository ships a 50-atom Li₁₀GeP₂S₁₂ (LGPS) cell for smoke tests. With
MACE installed and a checkpoint on disk:

```bash
./bin/mliport-mace sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk
```

The [Quickstart](quickstart.md) walks through this step by step, including
installation and how to get the checkpoint.

## Where to go next

| I want to… | Read |
|---|---|
| Install and run my first calculation | [Quickstart](quickstart.md) |
| Understand environments, Python versions and GPUs | [Installation](installation.md) |
| Check a model on a system with a known answer | [Test systems](test-systems.md) |
| Pick a model, or see which ones were validated | [Models](models.md), [Backends](backends/index.md) |
| Set options through the CLI, INCAR file or `settings.ini` | [Configuration](configuration.md) |
| Run a relaxation, MD, NEB, configuration search or batch | [Workflows](workflows/index.md) |
| Analyze a trajectory (MSD, RDF, conductivity, …) | [Analysis](analysis/index.md) |
| Use mliport from Python scripts | [Python API](api.md) |
| Know what each output file contains | [Output files](outputs.md) |
| See what has been validated and how | [Validation](validation.md) |
| Fix an error | [Troubleshooting](troubleshooting.md) |
| Move from the old `mlipx` name | [Migration](migration-from-mlipx.md) |
| Cite mliport and the models it runs | [Citations](citations.md) |
| Work on mliport itself | [Development](development.md) |

## Platforms and requirements

mliport runs on **Linux** and **WSL2** with Python **3.10–3.13**. GPU
acceleration requires an NVIDIA GPU with a working CUDA driver. The installer
uses [uv](https://docs.astral.sh/uv/) to manage Python environments. macOS is
not supported.

Source code: [github.com/hydrogen1222/mliport](https://github.com/hydrogen1222/mliport)
