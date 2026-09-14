# Installation

mliport installs **inside each backend's own Python environment**. It is not a
cross-environment dispatcher: the CLI in `.venv-mace` can only run MACE, the
CLI in `.venv` can only run UMA, and so on. Pick the backend you need, install
its environment, and run `mliport` from that environment.

## Recommended: the installer

Install `uv` first, then clone the repository:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hydrogen1222/mliport
cd mliport
```

On an NVIDIA GPU host (tested on V100-SXM2-16GB with driver 580.173.02):

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

On a CPU-only host:

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

`--dry-run` prints every step without executing anything. Unless
`--skip-doctor` is given, the installer runs `mliport doctor` in each
environment after installing it. The one-command install includes mliport's
analysis, transport and electrolyte features; they are not a separate step.

The installer also writes thin launchers into `./bin/`:

```bash
./bin/mliport-mace sp structure.cif --model model.model --model-type mace
./bin/mliport-uma  sp structure.cif --model uma.pt --model-type uma
```

A launcher only selects the Python runtime. It never sets model defaults,
dtype or scientific configuration. DPA and GRACE commands can restart
themselves in a process with isolated GPU visibility; see below.

## Verify the install

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
.venv-dpa/bin/mliport  doctor --engine dpa  --device auto
.venv-grace/bin/mliport doctor --engine grace --device auto
.venv/bin/mliport       doctor --engine uma  --device auto
```

Each command exits 0 when the required checks pass and ends with a line such
as `Environment checks passed for MACE on CUDA.`. The report also lists
optional features (TUI, analysis, transport, electrolyte, plotting) and a
model probe is available with `--model PATH --task TASK --structure FILE`.
Use `doctor --json` for a machine-readable report.

## Models

mliport does not download checkpoints. Fetch the checkpoint with the upstream
tool and pass its path with `--model`; local files work offline. The pinned
artifact names, URLs and SHA-256 values used by the validation suite are in
[models.md](models.md) and `validation/science/model_manifest.json`.

Behind a SOCKS proxy, install `httpx[socks]` in the environment you use for
Hugging Face downloads, and keep `NO_PROXY` free of malformed bracket entries
that httpx rejects. `HF_ENDPOINT=https://hf-mirror.com` selects the Hugging
Face mirror used by some sites. The installer itself goes through `uv`, which
handles the same proxy variables.

## Offline hosts

Point `--source offline` at a prepared wheel cache, or install on a connected
host and copy the environments. uv's cache (`UV_CACHE_DIR`) and the model
directory are the only moving parts; `UV_PYTHON_DOWNLOADS=never` keeps the
installer from fetching a managed interpreter.

## Disk usage and uninstall

A four-engine GPU install is roughly 25-35 GB (PyTorch and TensorFlow
dominate); a CPU install is smaller. Remove an environment by deleting its
directory (`.venv-mace`, `.venv-dpa`, `.venv-grace`, `.venv`); the repository
itself and model checkpoints are not touched. `./bin/mliport-*` launchers can
be removed with the `bin/` directory. Re-run the installer to recreate an
environment.

<!-- BEGIN GENERATED: backend compatibility matrix -->
| Backend | Environment | Python | Runtime stack |
|---|---|---|---|
| UMA (FAIRChem) | `.venv` | 3.11, 3.12 | `fairchem-core==2.21.0` (torch 2.8.0) |
| MACE | `.venv-mace` | 3.10, 3.11, 3.12 | `mace-torch==0.3.16` (torch 2.8.0) |
| DPA (DeepMD-kit) | `.venv-dpa` | 3.10, 3.11, 3.12 | `deepmd-kit==3.1.3` (torch 2.10.0) |
| GRACE | `.venv-grace` | 3.10, 3.11, 3.12 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |
<!-- END GENERATED -->

## DPA and GRACE device isolation

DPA and GRACE cannot move a CUDA context after their framework loads. When
`CUDA_VISIBLE_DEVICES` is not already set, the mliport CLI restarts the same
command in a fresh process with the requested GPU visible and prints a short
note. The Python API cannot restart the caller: set `CUDA_VISIBLE_DEVICES`
(or `""` for CPU) before importing those frameworks.

## Python version selection

Python is chosen **per backend**, from the pinned compatibility registry
(`mliport/mliport/install/compatibility.py`). The `Python` column above is the
intersection of mliport's supported versions, the backend's own
`requires-python` and the framework wheel's Python range.

`--python` must satisfy every requested backend. An incompatible combination
(for example `--python 3.10 --engines uma`) is rejected before anything is
installed; the installer never silently substitutes a different version.

## Running mliport

Run the CLI from the environment that owns the backend:

```bash
.venv-mace/bin/mliport  sp structure.cif --model model.model --model-type mace
.venv-dpa/bin/mliport   sp structure.cif --model model.pt   --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model saved_model --model-type grace
.venv/bin/mliport       sp structure.cif --model uma.pt     --model-type uma
```

Equivalently, use the generated launchers (`./bin/mliport-mace`, ...), which
exec exactly those commands.

On Windows, each environment exposes `Scripts\mliport.exe`:

```bat
.venv-mace\Scripts\mliport.exe sp structure.cif --model model.model --model-type mace
```

and the installer writes `bin\mliport-mace.cmd` launchers next to the POSIX
ones.

## Manual installation

Each backend environment needs mliport itself plus that engine's stack. The
authoritative pins live in `mliport/mliport/install/compatibility.py`; the
installer is the only path that keeps them consistent with your GPU
architecture. If you install manually, create one environment per backend,
install `./mliport` **into that environment**, then run `mliport doctor` and
confirm every check passes before trusting results. Never install mliport once
and expect it to reach into other backend environments.

## GPU architecture compatibility

The installer and `mliport doctor` choose the correct PyTorch/TensorFlow build
for your GPU family (Maxwell/Pascal/Volta use the cu126 legacy channel,
Turing and newer use cu128+). `mliport setup` prints the detected hardware and
the matching route.

| Engine | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| MACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| DPA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| GRACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | experimental |

`experimental` means the upstream framework does not test that family;
`needs runtime smoke test` means a route exists but mliport has not promoted
a runtime record for it. Neither state is a "verified" claim.
