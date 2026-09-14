# Backends

mliport selects exactly one backend per run. The backend comes from
`--model-type`, from a configured alias/profile, or the run stops with
`No MLIP backend was selected`.

| Backend | Manual | Model type | Typical checkpoint |
|---|---|---|---|
| MACE | [mace.md](mace.md) | `mace` | `mace-omat-0-medium.model` |
| DPA (DeepMD-kit) | [dpa.md](dpa.md) | `dpa` | `DPA-3.1-3M.pt` with `--head Omat24` |
| GRACE | [grace.md](grace.md) | `grace` | `GRACE-2L-OMAT-medium-base` |
| UMA (FAIRChem) | [uma.md](uma.md) | `uma` or `fairchem` | `uma-s-1p2.pt` with `--task omat` |

Each backend has its own environment (`.venv-mace`, `.venv-dpa`,
`.venv-grace`, `.venv`) because their framework and Python requirements
differ. Check a runtime with:

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

DPA and GRACE need CUDA device isolation before their framework loads. The
CLI restarts such commands in a fresh process with `CUDA_VISIBLE_DEVICES`
set; the Python API requires you to set it in the calling process.
