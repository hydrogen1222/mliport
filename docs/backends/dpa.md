# DPA (DeepMD-kit)

DPA checkpoints are loaded through `deepmd-kit`. Many of them are multi-task
models with several named branches, and mliport will not guess which branch you
mean: you pass it explicitly as the head, or the run fails closed.

## Install

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit==3.1.3` (torch 2.10.0) |

```bash
./scripts/install_mliport.sh --engines dpa --device auto
```
<!-- END GENERATED -->

The model loader needs the `deepmd-kit[torch]` metadata, and the installer
brings along the matching `mpich` package.

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `dpa_omat` | DPA-3.1-3M, branch Omat24 | upstream/model-defined | bulk | Omat24 | `86dd3a804d78...` |
<!-- END GENERATED -->

## Run it

```bash
.venv-dpa/bin/mliport sp structure.cif \
  --model /path/to/DPA-3.1-3M.pt --model-type dpa --head Omat24
```

To see which branches a checkpoint offers, ask DeePMD-kit, then pass the one you
want as `--head` (`HEAD` in an INCAR file):

```bash
dp --pt show DPA-3.1-3M.pt model-branch
```

## Things to know

- **Leaving out the head on a multi-task checkpoint is a fatal configuration
  error.** There is no silent default branch.
- **Precision** is defined by the model. mliport records the dtype the backend
  actually reports and never substitutes its own idea of it.
- **Stress** is reported only if the checkpoint and branch you loaded expose it.
- **DPA needs per-process GPU isolation.** The CLI restarts a command with
  `CUDA_VISIBLE_DEVICES` when it is unset; from the Python API, set the variable
  in the calling process before importing the framework.
- DPA has no `inference_mode`. UMA-only options are rejected under strict
  configuration rather than ignored.

mliport is MIT-licensed. The upstream software and model must be cited
separately; see [Citations](../citations.md).
