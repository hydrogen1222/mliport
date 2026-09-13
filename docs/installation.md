# Installation

mliport installs **inside each backend's own Python environment**. It is not a
cross-environment dispatcher: the CLI in `.venv-mace` can only run MACE, the
CLI in `.venv` can only run UMA, and so on. Pick the backend you need, install
its environment, and run `mliport` from that environment.

## Recommended: the installer

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

`--dry-run` prints every step without executing anything. Unless
`--skip-doctor` is given, the installer runs `mliport doctor` in each
environment after installing it.

The installer also writes thin launchers into `./bin/`:

```bash
./bin/mliport-mace sp structure.cif --model model.model --model-type mace
./bin/mliport-uma  sp structure.cif --model uma.pt --model-type uma
```

A launcher only selects the Python runtime; it never changes
`CUDA_VISIBLE_DEVICES`, model defaults, dtype or scientific configuration.

<!-- BEGIN GENERATED: backend compatibility matrix -->
| Backend | Environment | Python | Runtime stack |
|---|---|---|---|
| UMA (FAIRChem) | `.venv` | 3.11, 3.12 | `fairchem-core==2.21.0` (torch 2.8.0) |
| MACE | `.venv-mace` | 3.10, 3.11, 3.12 | `mace-torch==0.3.16` (torch 2.8.0) |
| DPA (DeepMD-kit) | `.venv-dpa` | 3.10, 3.11, 3.12 | `deepmd-kit==3.1.3` (torch 2.10.0) |
| GRACE | `.venv-grace` | 3.10, 3.11, 3.12 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |
<!-- END GENERATED -->

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
