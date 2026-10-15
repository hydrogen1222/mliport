# Installation

mliport does not live in one big environment. Each backend (MACE, DPA, GRACE,
UMA) pulls in its own deep-learning stack, and those stacks do not always get
along, so the installer builds **one environment per backend** and installs
mliport into each. The `mliport` you run from `.venv-mace` can only run MACE;
the one in `.venv` can only run UMA, and so on. Pick the backends you need,
install those, and call `mliport` from the matching environment.

## The installer

Install [uv](https://docs.astral.sh/uv/) and clone the repository:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/hydrogen1222/mliport
cd mliport
```

On a machine with an NVIDIA GPU and a working driver (this route was tested on
a V100-SXM2-16GB with driver 580.173.02):

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

On a machine without one:

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu
```

Pass fewer engines if you only need some. `--dry-run` prints every step without
executing anything. Unless you give `--skip-doctor`, the installer runs
`mliport doctor` in each environment when it finishes. The analysis, transport
and electrolyte features come with the same install; there is no second step.

The installer also writes small launchers into `./bin/`:

```bash
./bin/mliport-mace sp structure.cif --model model.model --model-type mace
./bin/mliport-uma  sp structure.cif --model uma.pt --model-type uma
```

A launcher only picks the Python environment. It never sets a model, a dtype or
any scientific option for you.

## Which environment runs what

<!-- BEGIN GENERATED: backend-matrix -->
| Backend | Environment | Python | Runtime stack |
|---|---|---|---|
| UMA (FAIRChem) | `.venv` | 3.11, 3.12, 3.13 | `fairchem-core==2.21.0` (torch 2.8.0) |
| MACE | `.venv-mace` | 3.10, 3.11, 3.12, 3.13 | `mace-torch==0.3.16` (torch 2.8.0) |
| DPA (DeepMD-kit) | `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit==3.1.3` (torch 2.10.0) |
| GRACE | `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |
<!-- END GENERATED -->

The `Python` column is what the installer will accept for that backend: the
intersection of the Python versions mliport supports, the backend's own
`requires-python`, and the Python range of the framework wheels it needs. The
table is generated from the compatibility registry
(`mliport/mliport/install/compatibility.py`), which is also where the exact
pins live.

You can force an interpreter with `--python`, but it has to satisfy *every*
backend you asked for. A combination that cannot work, such as
`--python 3.10 --engines uma`, is refused before anything is installed; the
installer never quietly picks a different version.

## Check the install

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
.venv-dpa/bin/mliport  doctor --engine dpa  --device auto
.venv-grace/bin/mliport doctor --engine grace --device auto
.venv/bin/mliport       doctor --engine uma  --device auto
```

Each command exits 0 when the required checks pass and ends with a line such as
`Environment checks passed for MACE on CUDA.` The report also lists optional
features (TUI, analysis, transport, electrolyte, plotting). To try a real
model, add `--model PATH --task TASK --structure FILE`. `doctor --json` gives
the same report in a machine-readable form, handy for cluster provisioning.

## Running mliport

Call the CLI from the environment that owns the backend:

```bash
.venv-mace/bin/mliport  sp structure.cif --model model.model --model-type mace
.venv-dpa/bin/mliport   sp structure.cif --model model.pt   --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model saved_model --model-type grace
.venv/bin/mliport       sp structure.cif --model uma.pt     --model-type uma
```

The launchers in `./bin/` run exactly these commands. On Windows each
environment has a `Scripts\mliport.exe`, and the installer writes
`bin\mliport-*.cmd` launchers next to the POSIX ones:

```bat
.venv-mace\Scripts\mliport.exe sp structure.cif --model model.model --model-type mace
```

### DPA and GRACE and the GPU

DPA and GRACE cannot move a CUDA context once their framework has loaded. When
`CUDA_VISIBLE_DEVICES` is not set, the CLI restarts the same command in a fresh
process with the GPU you asked for visible, and prints a short note saying so.
The Python API cannot restart its caller, so if you use DPA or GRACE from
Python, set `CUDA_VISIBLE_DEVICES` (or `""` for CPU) before importing the
framework.

## Models

mliport does not download checkpoints. Fetch one with the upstream tool and pass
its path with `--model`; a local file works offline. Where to get the
checkpoints, and the pinned file names and SHA-256 values used by the validation
suite, are on the [Models](models.md) page.

If you download from Hugging Face through a SOCKS proxy, install `httpx[socks]`
in the environment you download with, and keep `NO_PROXY` free of bracketed
entries such as `[::1]`, which httpx rejects. `HF_ENDPOINT=https://hf-mirror.com`
selects the Hugging Face mirror some sites use. The installer itself goes
through `uv`, which reads the same proxy variables.

## GPU architectures

PyTorch and TensorFlow ship different CUDA builds for different GPU
generations, and the installer picks the right one for yours: Maxwell, Pascal
and Volta use the cu126 legacy channel, Turing and newer use cu128 or later.
`mliport setup` prints the GPU it detected and the route it would take.

<!-- BEGIN GENERATED: gpu-routes -->
| Engine | Maxwell | Pascal | Volta / V100 | Ada / RTX 4090 | Other Turing+ | Hopper / Blackwell |
|---|---|---|---|---|---|---|
| UMA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| MACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| DPA | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test |
| GRACE | experimental | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | needs runtime smoke test | experimental |
<!-- END GENERATED -->

Read the cells carefully: `experimental` means the upstream framework does not
test that GPU family, and `needs runtime smoke test` means a route exists but
mliport has not recorded a real run on that hardware. Neither means "verified".
Only the V100 has hardware evidence. The other families are classified from a
close inspection of the wheels and native extensions: the compiled architecture
lists, the `+cu126`/`+cu128` wheel hashes and the `cuobjdump` output are in the
[architecture report](https://github.com/hydrogen1222/mliport/blob/main/validation/compatibility/GPU_ARCHITECTURE_THEORY_REPORT.md).
`mliport doctor` prints that classification for the GPU it finds instead of
claiming a verification.

Pre-Turing GPUs depend on the CUDA 12.6 channel, and PyTorch stops publishing
it from 2.15. In those environments keep the installer's pins and do not
upgrade the framework.

## Offline machines

Point `--source offline` at a prepared wheel cache, or install on a connected
machine and copy the environments over. Only two things move: uv's cache
(`UV_CACHE_DIR`) and the model directory. `UV_PYTHON_DOWNLOADS=never` stops the
installer from fetching a managed interpreter.

## Disk space and removal

A four-engine GPU install takes roughly 25–35 GB, most of it PyTorch and
TensorFlow; a CPU install is smaller. To remove a backend, delete its
environment directory (`.venv-mace`, `.venv-dpa`, `.venv-grace` or `.venv`).
The repository and your model files are untouched, and the `./bin/mliport-*`
launchers go with the `bin/` directory. Run the installer again to recreate an
environment.

## Installing by hand

Every backend environment needs mliport itself plus that engine's stack. The
authoritative pins are in `mliport/mliport/install/compatibility.py`, and the
installer is the only path that keeps them consistent with your GPU
architecture. If you do it by hand anyway, create one environment per backend,
install `./mliport` into each, and run `mliport doctor` there; do not trust
results until every check passes. Installing mliport once and expecting it to
reach into the other environments will not work.
