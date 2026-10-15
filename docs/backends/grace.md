# GRACE

GRACE models are exported TensorFlow SavedModels, loaded through
`tensorpotential`. The model path is the SavedModel **directory**, not a single
file.

## Install

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-grace` | 3.10, 3.11, 3.12, 3.13 | `tensorpotential==0.6.0` (tensorflow 2.20.0) |

```bash
./scripts/install_mliport.sh --engines grace --device auto
```
<!-- END GENERATED -->

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
| `grace_omat` | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | bulk | None | `d497af4c7d9a...` |
| `grace_omat_fp64` | GRACE-2L-OMAT-medium-base fp64 build | float64 | bulk | None | `53b945efce37...` |
<!-- END GENERATED -->

## Run it

```bash
.venv-grace/bin/mliport sp structure.cif \
  --model /path/to/saved_model --model-type grace
```

Official exported SavedModels come from the upstream AMS-ICAMS-RUB release
channels; point `--model` (`MODEL_PATH`) at the directory you unpack.

## Things to know

- **No heads.** GRACE models do not expose an mliport-level head; `--task` is the
  explicit periodic-boundary semantic (`bulk` or `molecule`).
- **GPU isolation.** TensorFlow's GPU visibility has to be fixed per process. The
  CLI restarts a command with `CUDA_VISIBLE_DEVICES` when it is unset; the Python
  API and the queue worker need the same isolation.
- **GPU memory and neighbor lists** are tunable. TensorFlow normally reserves
  nearly all visible memory, so memory growth is on by default
  (`--gpu-memory-growth`, `--gpu-memory-limit-mb`). A Verlet-style neighbor
  cache (`--neighbor-cache`, `--neighbor-skin`, default skin 1.5 Å) is on by
  default and rebuilds the table when an atom moves more than half the skin.
- **Precision** follows the model: the pinned profiles include the upstream
  default build and an fp64 build, as listed above.
- **Stress** depends on the checkpoint.

mliport is MIT-licensed. The upstream software and model must be cited
separately; see [Citations](../citations.md).
