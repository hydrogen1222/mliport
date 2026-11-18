# FAQ & Recipes

Answers to common questions and worked examples for recurring tasks.

## Installation & setup

### Which Python versions are supported?

Python **3.10 through 3.13** on **Linux** and **WSL2**. macOS is not
supported. GPU acceleration requires an NVIDIA GPU with a working CUDA driver.

### Do I need to install all five backends?

No. Install only the backends you plan to use. Each backend lives in its own
Python virtual environment, so they do not interfere with each other. The
installer creates one environment per `--engines` argument:

```bash
# Just MACE
./scripts/install_mliport.sh --engines mace --device auto

# MACE and UMA together (two separate environments)
./scripts/install_mliport.sh --engines mace uma --device auto
```

### How do I check whether my installation is working?

Run `mliport doctor`:

```bash
./bin/mliport-mace doctor --engine mace --device auto
```

This checks that the backend's Python packages are importable, that the
requested device is available, and (optionally) that a model file can be
loaded. Pass `--model` and `--task` to also probe a specific checkpoint.

### Can I use mliport in a conda environment?

The installer uses [uv](https://docs.astral.sh/uv/) and creates its own
venvs. If you need mliport inside an existing conda environment, you can
install it manually with pip — see [Installation](installation.md) for
the manual install procedure.

### How do I update mliport?

Pull the latest code and re-run the installer for each backend you use:

```bash
git pull
./scripts/install_mliport.sh --engines mace --device auto
```

The installer detects the existing environment and updates it.

---

## Models & backends

### How do I choose a backend?

It depends on your use case:

| Scenario | Recommended backend |
|---|---|
| General-purpose bulk materials | MACE (OMAT-0) or UMA |
| Molecules and molecular systems | UMA (`--task omol`) or MACE (`--task omol`) |
| Custom system-specific potential | NEP (train your own `nep.txt`) |
| Need TensorFlow ecosystem | GRACE |
| DeePMD ecosystem / multi-task | DPA |

All five backends support all workflow types (single point, optimization, MD,
NEB, configuration search). The main differences are in the models available,
licensing terms, and ecosystem compatibility.

See [Backends](backends/index.md) for a detailed comparison.

### Where do I download model files?

mliport never downloads or bundles model weights. You fetch them from each
project's release page. The [Models](models.md) page lists all known
checkpoints with download links, SHA-256 hashes, and license terms.

### What does `--task` do?

Multi-head models (MACE, UMA) are trained on datasets with different physical
semantics. `--task` selects which head to use:

- `omat` / `bulk` — periodic bulk materials (OMat24 dataset)
- `omol` — molecules and molecular systems
- `oc20` / `oc25` — Open Catalyst datasets (adsorbates on surfaces)

The exact set of available tasks depends on the checkpoint. Using the wrong
task produces physically meaningless results (wrong reference energies), so
mliport requires you to specify it explicitly.

### What is `--head` vs `--task`?

`--task` is the MACE/UMA concept for selecting a model head by the dataset it
was trained on. `--head` is the DPA/DeepMD concept for selecting a branch in
a multi-task checkpoint. They serve the same purpose — selecting which output
channel of a multi-task model to use — but the naming differs by backend.

### Can I mix absolute energies from different models?

**No.** Different models (and even different tasks within one model) use
different reference energy levels. Comparing or subtracting absolute energies
across models or tasks is physically meaningless. Compare energy *differences*
(reaction energies, barriers, relative stabilities) within a single
model+task combination.

---

## Running calculations

### Why does mliport refuse to run without `--model-type`?

There is no default backend. mliport requires you to say which model produced
every number, so results are always unambiguously attributed. You can avoid
typing `--model-type` every time by setting up a model alias in
`settings.ini` — see [Configuration](configuration.md).

### How do I reuse settings across runs?

Create a `settings.ini` file:

```ini
[defaults]
device = cuda
strict_config = true

[model:mace-omat]
model_path = /models/mace-omat-0-medium.model
model_type = mace
task = bulk

[profile:production-md]
ensemble = NVT
thermostat = BUSSI
bussi_tau = 100
temperature = 600
timestep = 2.0
steps = 100000
save_interval = 100
```

Then use the alias and profile:

```bash
mliport md structure.vasp --model-alias mace-omat --profile production-md \
  --output runs/md-001
```

### What is `mliport run` for?

`mliport run` reads an INCAR-style input file (`INCAR.mliport`) that contains
all calculation parameters in VASP-like syntax:

```
CALC_TYPE = md
MODEL_PATH = /models/mace-omat-0-medium.model
MODEL_TYPE = mace
TASK = bulk
MD_ENSEMBLE = NVT
TEMPERATURE = 600
TIMESTEP = 2.0
STEPS = 10000
```

```bash
mliport run --incar INCAR.mliport --output runs/md-from-incar
```

This is useful for users familiar with VASP input files, or for documenting
a calculation as a single self-contained file.

### How do I run on multiple structures?

Use `mliport batch`:

```bash
mliport batch sp structures/*.cif \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --output runs/batch-sp
```

Each structure gets its own subdirectory under the output path.

### Can I resume a failed MD run?

MD runs can be resumed by providing the same output directory — mliport picks
up from the last saved checkpoint. NEB runs support explicit `--resume`. For
configuration searches, use `--resume` with the original output directory.

### What is pre-relaxation in MD?

The `--pre-relax` flag (or `pre_relax=True` in the API) runs a quick geometry
optimization of atomic positions *before* starting MD. This reduces large
initial forces that can cause instabilities in the first few MD steps. It
optimizes positions only — it does not relax the cell or eliminate stress.

---

## NPT molecular dynamics

### Which backends support NPT?

NPT uses the ASE `IsotropicMTKNPT` integrator, so it works with all
ASE-based backends: **MACE, DPA, GRACE, and UMA**. NEP with the `gpumd`
runtime does not support NPT (it uses native GPUMD integrators that do not
include this ensemble). NEP with `--nep-runtime ase` does support NPT.

### What NPT parameters do I need?

At minimum, you need:

```bash
--ensemble NPT --temperature 300 --pressure-gpa 0.0 \
--npt-tdamp 50 --npt-pdamp 500
```

- `--pressure-gpa`: target pressure in GPa (0 = ambient, positive =
  compressive)
- `--npt-tdamp`: thermostat damping time in fs (typically 25–100 fs)
- `--npt-pdamp`: barostat damping time in fs (typically 250–1000 fs, longer
  than tdamp)

### Do NPT trajectories have variable cells?

Yes. The cell shape and volume change at each step. The trajectory (`XDATCAR`)
records the cell at each frame. Analysis tools that assume fixed cells (like
standard RDF binning) should be used with care on NPT trajectories.

---

## Analysis

### What does "insufficient sampling" mean?

mliport's analysis pipeline checks whether a trajectory is long enough to
produce reliable statistics. If it detects that the trajectory is too short
(for example, the MSD hasn't reached the diffusive regime, or the RDF hasn't
converged), it reports an explicit "insufficient sampling" status rather than
returning an unreliable number. Run a longer trajectory or increase sampling.

### How do I analyze an MD trajectory?

```bash
# Mean squared displacement and tracer diffusivity
mliport analyze msd runs/md/XDATCAR --timestep 2.0

# Radial distribution function
mliport analyze rdf runs/md/XDATCAR --pairs Li-S --nbins 200

# Velocity autocorrelation function and vibrational spectrum
mliport analyze vacf runs/md/XDATCAR --timestep 2.0

# Ionic conductivity and transport properties
mliport analyze transport runs/md/XDATCAR --timestep 2.0 --temperature 600

# Arrhenius fit from multiple temperatures
mliport analyze arrhenius results_300K results_400K results_500K
```

See [Analysis](analysis/index.md) for the full list of available analyses.

### Can I analyze trajectories from other codes?

mliport reads `XDATCAR` files in VASP format. If your trajectory is in
another format, convert it to XDATCAR first (for example with ASE's
`write` function).

---

## Output & reproducibility

### What files does each calculation produce?

| File | All types | Description |
|---|---|---|
| `mliport_results.json` | ✓ | Energy, forces, stress, timing, provenance |
| `resolved_config.json` | ✓ | Every config value and its source layer |
| `run.log` | ✓ | Full run log |
| `OUTCAR` | sp, opt | VASP-style forces and energy |
| `OSZICAR` | opt, md | Step-by-step convergence / thermodynamics |
| `CONTCAR` | sp, opt | Final structure in POSCAR format |
| `XDATCAR` | md | Trajectory in VASP format |

See [Output files](outputs.md) for complete details.

### How do I know which model produced a result?

Every `mliport_results.json` records the model file path and its SHA-256 hash,
the requested vs. actual device, where each configuration value came from
(default, settings.ini, INCAR, or CLI), and the mliport version. The
`resolved_config.json` file adds the full configuration with source
annotations.

### Why does mliport refuse to overwrite existing results?

To protect you from accidentally destroying a completed calculation. If you
deliberately want to replace results, pass `--overwrite` on the CLI or
`overwrite=True` in the API.

---

## NEP-specific questions

### What is `--nep-runtime`?

NEP has two execution runtimes:

- **`ase`** — uses the Calorine Python package to run NEP through ASE. Uses
  `CPUNEP` on CPU and `GPUNEP` on CUDA. Supports all mliport features
  including NPT.
- **`gpumd`** — launches the native GPUMD executable directly. GPU-only,
  highest performance for large-scale MD, but does not support NPT and has
  a different feature set.

You must always specify `--nep-runtime` when using NEP. mliport never guesses
which runtime to use.

### Do I need `--nep-gpumd-command`?

Only when using `--nep-runtime gpumd` or when the Calorine `GPUNEP` calculator
needs it. Point it to the `gpumd` binary on your system:

```bash
--nep-runtime gpumd --nep-gpumd-command /usr/local/bin/gpumd
```

---

## Troubleshooting shortcuts

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: mace` | MACE not installed in this environment | Re-run installer with `--engines mace` |
| `Unknown config key: xxx` | Typo or wrong-backend option with strict mode | Check spelling; see [Configuration](configuration.md) |
| `CUDA out of memory` | System too large for GPU VRAM | Use a smaller model, reduce batch, or use CPU |
| `Output directory exists` | Previous run's results still there | Use `--overwrite` or a new `--output` path |
| `No model_type specified` | Missing `--model-type` and no alias | Add `--model-type` or set up a model alias |

For more, see [Troubleshooting](troubleshooting.md).
