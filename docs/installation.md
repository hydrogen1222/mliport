# Installation

## Why one environment per backend?

Machine-learning interatomic potentials depend on deep-learning frameworks — PyTorch,
TensorFlow, JAX — and those frameworks pin specific versions of CUDA libraries, NumPy
and other low-level packages. When two backends need *different* versions of the same
library (MACE wants PyTorch 2.8, DPA needs 2.11, GRACE runs on TensorFlow), putting
them in one environment creates unsolvable version conflicts.

mliport solves this by building **one virtual environment per backend**. Each
environment contains mliport itself plus that backend's stack, and the `mliport`
command inside it can only call that backend. This keeps every dependency chain clean
and every run reproducible.

| Backend | Environment | What it can run |
|---|---|---|
| UMA (FAIRChem) | `.venv` | UMA checkpoints only |
| MACE | `.venv-mace` | MACE checkpoints only |
| DPA (DeePMD-kit) | `.venv-dpa` | DPA checkpoints only |
| GRACE | `.venv-grace` | GRACE checkpoints only |
| NEP (Calorine / GPUMD) | `.venv-nep` | NEP potentials only |

You only need the backends you use. If you only work with MACE, install only MACE.

## Prerequisites

You need two things before running the installer:

1. **uv** — a fast Python package manager that handles virtual environments and
   dependency resolution. It replaces `pip` and `venv` in one tool and is
   significantly faster:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. **The mliport repository**:

    ```bash
    git clone https://github.com/hydrogen1222/mliport
    cd mliport
    ```

On a GPU machine you also need a working NVIDIA driver. The installer was tested on
a V100-SXM2-16GB with driver 580.173.02.

## Running the installer

### GPU machine (recommended)

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

`--device auto` detects whether a GPU is available and picks the right CUDA wheel
channel for your GPU architecture. You can install a subset — `--engines mace dpa`
skips GRACE and UMA.

### CPU-only machine

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

CPU installs use CPU PyTorch wheels and work everywhere, but inference is slower.
The GRACE TensorFlow stack may still pull CUDA runtime libraries because of
TensorPotential's dependency contract, even when selected for CPU execution. CPU
installation is useful for post-processing, analysis, and testing on laptops.

### Windows (PowerShell)

The `scripts/install_mliport.sh` bootstrap is Bash-only. From the repository root,
run the Python-native installer directly in PowerShell:

```powershell
$env:PYTHONPATH = Join-Path $PWD "mliport"
uv run --no-project --with packaging --python 3.12 -- python -m mliport.install --engines mace --device auto
```

This creates Windows virtual environments under `.venv-*/Scripts/` and native
`bin/mliport-*.cmd` launchers. Use `.venv-mace\Scripts\mliport.exe` or
`bin\mliport-mace.cmd` to run MACE. Backend build and GPU support remain subject
to each upstream project's Windows support; the presence of a launcher alone is
not evidence that a backend's GPU runtime is available.

### What the installer does

For each engine you name, the installer:

1. Creates a virtual environment (`.venv-mace`, `.venv-dpa`, etc.).
2. Installs mliport in editable mode with the analysis extras.
3. Installs the backend's framework at the pinned version.
4. Writes a platform-native launcher into `./bin/` (e.g. `./bin/mliport-mace` on POSIX or `./bin/mliport-mace.cmd` on Windows).
5. Runs `mliport doctor` to verify the installation (unless `--skip-doctor`).

The whole process is deterministic: the same command on the same machine produces the
same environments, because the framework versions are pinned in the compatibility
registry (`mliport/mliport/install/compatibility.py`).

### Useful installer flags

| Flag | What it does |
|---|---|
| `--dry-run` | Print every step without executing anything |
| `--clean` | Delete and rebuild the environments of the named engines |
| `--skip-doctor` | Skip the verification step at the end |
| `--python 3.12` | Force a specific Python version (must satisfy every named backend) |
| `--source offline` | Use a prepared wheel cache instead of downloading |

## NEP: an opt-in backend

NEP is not in the default engine list because its native GPU runtime needs a
separately compiled `gpumd` binary that the installer cannot build for you.

**CPU-only NEP** works immediately — the Calorine ASE runtime needs no compilation:

```bash
./scripts/install_mliport.sh --engines nep --device cpu
```

**GPU NEP** requires you to build `gpumd` yourself with CUDA Toolkit 12.9 (or
compatible), then either put it on `PATH` or set `MLIPORT_GPUMD_COMMAND`:

```bash
./scripts/install_mliport.sh --engines nep --device auto
# The closing doctor will fail until gpumd is available.
# Build gpumd, then re-run the installer, or use --skip-doctor
# and call .venv-nep/bin/mliport directly.
```

See [NEP](backends/nep.md) for build instructions and runtime details.

## Which environment runs what

<!-- BEGIN GENERATED: backend-matrix -->
| Backend | Environment | Python | Runtime stack |
|---|---|---|---|
| UMA (FAIRChem) | `.venv` | 3.11, 3.12, 3.13 | `fairchem-core==2.25.0` (torch 2.13.0) |
| MACE | `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |
| DPA (DeepMD-kit) | `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit @ git+https://github.com/deepmodeling/deepmd-kit.git@0edeaef5617e9878edeb2346bf37bb6a035a4f51` (torch 2.11.0) |
| NEP (Calorine / GPUMD) | `.venv-nep` | 3.10, 3.11, 3.12, 3.13 | `calorine==4.1` (NumPy/ASE, resolved by the installer; gpumd built with CUDA Toolkit 12.9) |
| GRACE | `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |
<!-- END GENERATED -->

The `Python` column shows the versions the installer will accept for that backend.
This is the intersection of mliport's own `requires-python`, the backend's constraints,
and the framework wheels available. The table is auto-generated from the compatibility
registry — the authoritative source of pins.

You can force an interpreter with `--python`, but it must satisfy *every* backend
you asked for. An impossible combination like `--python 3.10 --engines uma` (UMA
needs 3.11+) is refused before anything is installed.

## Verifying the installation

After the installer finishes, verify each backend:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
.venv-dpa/bin/mliport  doctor --engine dpa  --device auto
.venv-grace/bin/mliport doctor --engine grace --device auto
.venv/bin/mliport       doctor --engine uma  --device auto
.venv-nep/bin/mliport   doctor --engine nep  --device cpu     # if you installed NEP
```

Each command exits 0 when the required checks pass and reports optional features
(TUI, analysis, transport, electrolyte, plotting). To test with a real model, add
`--model PATH --task TASK --structure FILE`. For machine-readable output (useful for
cluster provisioning scripts), use `doctor --json`.

## Running mliport

### Using environment paths directly

Call `mliport` from the environment that owns the backend:

```bash
.venv-mace/bin/mliport  sp structure.cif --model model.model --model-type mace
.venv-dpa/bin/mliport   sp structure.cif --model model.pt   --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model saved_model --model-type grace
.venv/bin/mliport       sp structure.cif --model uma.pt     --model-type uma
.venv-nep/bin/mliport   sp structure.cif --model nep.txt    --model-type nep --nep-runtime ase --device cpu
```

### Using launchers

The installer writes convenience scripts into `./bin/` that select the right
environment for you:

```bash
./bin/mliport-mace sp structure.cif --model model.model --model-type mace
./bin/mliport-uma  sp structure.cif --model uma.pt --model-type uma
```

A launcher only picks the Python environment. It never sets a model, a dtype or
any scientific option for you.

On Windows, each environment has `Scripts\mliport.exe`, and the installer writes
native `bin\mliport-*.cmd` launchers. POSIX shell launchers are written only on POSIX.

### DPA and GRACE: GPU device isolation

DPA (DeePMD-kit) and GRACE (TensorFlow) cannot move a CUDA context once their
framework has loaded. When `CUDA_VISIBLE_DEVICES` is not set, the CLI automatically
restarts the same command in a fresh process with the requested GPU visible, and
prints a note saying so.

The Python API cannot restart its caller. If you use DPA or GRACE from Python, set
`CUDA_VISIBLE_DEVICES` before importing the framework:

```bash
export CUDA_VISIBLE_DEVICES=0          # for GPU
export CUDA_VISIBLE_DEVICES=""         # for CPU
```

A GPU's UUID comes from `nvidia-smi --query-gpu=uuid --format=csv,noheader`.

## Models

mliport does not download checkpoints. Fetch the model you need from the upstream
project and pass its path with `--model`. A local file works offline. Where to get
checkpoints and the pinned file names used by the validation suite are on the
[Models](models.md) page.

If you download from Hugging Face through a SOCKS proxy, install `httpx[socks]` in
the download environment and keep `NO_PROXY` free of bracketed entries like `[::1]`.
`HF_ENDPOINT=https://hf-mirror.com` selects the mirror some sites use.

## GPU architectures

PyTorch and TensorFlow ship different CUDA builds for different GPU generations. The
installer detects your GPU and picks the right wheel channel automatically:

- **Turing and newer** (RTX 20xx/30xx/40xx, A100, H100): use `cu128` or `cu130`.
- **Volta** (V100): stays on `cu126`, which works with older drivers.
- **Maxwell and Pascal**: `cu126` legacy channel; marked `experimental` upstream.

When more than one NVIDIA GPU is visible, the installer checks that the selected
framework wheel covers **every** visible GPU. If no single route covers the full
set, `--device cuda` fails before creating environments and `--device auto`
falls back to CPU. Use `CUDA_VISIBLE_DEVICES` to expose only a compatible subset.
The official TensorFlow 2.20 wheel used by GRACE has no Maxwell GPU kernel, so
GRACE follows the same fail-closed/CPU-fallback rule on Maxwell; other experimental
profiles are not automatically treated as unsupported.

The CUDA 13 channel (`cu130`, used by UMA's torch 2.13) requires driver 580+ and
carries no kernels below Turing, so Volta stays on `cu126`.

<!-- BEGIN GENERATED: gpu-routes -->
| Engine | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| MACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| DPA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| NEP | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | experimental |
| GRACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | experimental |
<!-- END GENERATED -->

Read the cells carefully: `experimental` means the upstream framework does not test
that GPU family, and `needs runtime smoke test` means a route exists but mliport has
not recorded a real run on that hardware. Only the V100 has hardware evidence. The
detailed architecture analysis is in the
[architecture report](https://github.com/hydrogen1222/mliport/blob/main/validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md).

Pre-Turing GPUs depend on the CUDA 12.6 channel. PyTorch stops publishing it from
2.15, so keep the installer's pins and do not upgrade the framework in those
environments.

`mliport setup` prints the GPU it detected and the route it would take.

## Re-running the installer

The installer has a clear contract for existing environments:

- **Compatible environment** (a real venv of the requested Python, without another
  backend's packages): reused; pinned packages are re-resolved in place.
- **Incompatible environment** (a plain directory, wrong Python, or another backend's
  framework present): **refused**. The refusal names the reason and tells you to add
  `--clean` to delete and rebuild.

`--clean` only deletes environments of the engines you name. If you only need to
rebuild MACE, `--clean --engines mace` leaves the others untouched.

## Offline machines

For air-gapped clusters, prepare a wheel cache on a connected machine and point the
installer at it:

```bash
./scripts/install_mliport.sh --engines mace --device cpu --source offline
```

Only two things need to move: uv's wheel cache (`UV_CACHE_DIR`) and the model files.
Set `UV_PYTHON_DOWNLOADS=never` to stop the installer from fetching a managed
interpreter.

## Disk space and removal

A four-engine GPU install takes roughly 25–35 GB, most of it PyTorch and TensorFlow.
A CPU install is smaller. To remove a backend, delete its environment directory
(`.venv-mace`, `.venv-dpa`, `.venv-grace`, `.venv-nep` or `.venv`). The repository
and model files are untouched. The launchers go with the `bin/` directory.

## Installing by hand

Every backend environment needs mliport itself plus that engine's stack. The
authoritative pins are in `mliport/mliport/install/compatibility.py`. If you do it
by hand:

1. Create one environment per backend.
2. Install `./mliport` into each.
3. Run `mliport doctor` in each environment.
4. Do not trust results until every check passes.

Installing mliport once and expecting it to reach into other environments will not
work — each environment is self-contained by design.

## Common pitfalls

**"The installer exits before installing anything."** An unknown engine name, a
`--python` that a backend cannot use, or `--device cuda` on a machine without a
supported GPU. Run with `--dry-run` to see the plan and the constraint that failed.

**"A launcher is missing or points at the wrong environment."** The installer did
not finish, or mliport was installed by hand. Check with `ls bin/` and
`readlink -f bin/mliport-mace`. Re-run the installer for that engine.

**"I can't install all backends — the versions conflict."** That is exactly why
separate environments exist. Install each backend independently; they do not need
to coexist in one environment.

See [Troubleshooting](troubleshooting.md) for more installation issues.
