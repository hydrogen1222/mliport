# Backends

mliport runs exactly one backend per calculation. The backend comes from
`--model-type`, from a model alias or profile in your settings, or the run stops
with `No MLIP backend was selected`. Nothing falls back to something else.

| Backend | Page | `--model-type` | Checkpoint used in the examples |
|---|---|---|---|
| MACE | [MACE](mace.md) | `mace` | `mace-omat-0-medium.model` |
| DPA (DeepMD-kit) | [DPA](dpa.md) | `dpa` | `DPA-3.1-3M.pt` with `--head Omat24` |
| GRACE | [GRACE](grace.md) | `grace` | `GRACE-2L-OMAT-medium-base` |
| UMA (FAIRChem) | [UMA](uma.md) | `uma` (or `fairchem`) | `uma-s-1p2.pt` with `--task omat` |

Each backend lives in its own environment (`.venv-mace`, `.venv-dpa`,
`.venv-grace`, `.venv`) because their frameworks and Python requirements differ;
see [Installation](../installation.md) for how that works. To check that one of
them can really run, ask `doctor`:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

DPA and GRACE need their GPU chosen before the framework loads. The command line
handles this by restarting itself in a fresh process with `CUDA_VISIBLE_DEVICES`
set; from the Python API you have to set it in the calling process yourself.

Every page below follows the same outline: how the backend is installed, which
model profiles mliport has pinned, how to run it, and what to watch out for.
The environment and profile tables are generated from the compatibility registry
and the validation manifest, so they cannot drift from the code.
