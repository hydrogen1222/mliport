# Configuration

mliport resolves one configuration from several layers. Later layers
override earlier ones:

```text
built-in defaults
  < settings.ini
  < model alias ([model:NAME])
  < profile ([profile:NAME])
  < INCAR / job file
  < CLI flags / API arguments
```

Higher layers do not silently merge semantics: an option is resolved to
exactly one source and the provenance is recorded in
`resolved_config.json` (see [outputs.md](outputs.md)).

## Required configuration

| Item | How to provide it |
|---|---|
| calculation type | the subcommand (`sp`, `opt`, `md`, `neb`, `batch`) or `CALC_TYPE`/`CALCULATION` |
| backend | `--model-type`/`MODEL_TYPE`, or a model alias with `engine` |
| model path | `--model`/`MODEL_PATH`, or an alias `path` |
| structure | positional argument, or `--structure`/INCAR structure keys for `run` |
| DPA branch | `--head`/`HEAD` when the checkpoint is multi-task |
| NEB endpoints | `--initial`/`--final` or `NEB_INITIAL`/`NEB_FINAL` |

There is **no implicit UMA default**. With no backend from any layer the
resolver fails closed:

```text
Error: No MLIP backend was selected.

Choose one of:
  --model-type mace
  --model-type dpa
  --model-type grace
  --model-type uma

or use a configured model alias/profile.
```

## Strict configuration (default)

Scientific configuration is **strict by default**: a misspelled key or a key
belonging to another backend is a fatal configuration error, not a warning.
This prevents "typo -> warning -> run anyway".

```bash
mliport sp structure.cif --model m.model --model-type mace --inference-mode turbo
# Error: Option 'inference_mode' is not applicable to engine 'mace' ...
```

The legacy warning behaviour is an explicit opt-in, never a default:

```bash
mliport sp ... --lenient-config
```

```ini
[general]
strict_config = false
```

Use `mliport config validate settings.ini` and
`mliport config explain <KEY>` to inspect and debug configuration.

## settings.ini

Search paths (highest priority first) are printed by
`mliport config paths`. The environment variable `MLIPORT_SETTINGS` points at
an explicit file; the global `--settings PATH` flag overrides it.

```ini
[general]
strict_config = true

[model:mace_omat]
engine = mace
path = /models/mace-omat-0-medium.model
task = bulk

[md]
ensemble = NVT
temperature = 300

[profile:quick]
calc_type = sp
device = cuda
```

- `[model:NAME]` must declare `engine` (`uma`/`mace`/`dpa`/`grace`; `fairchem`
  is an alias for `uma`).
- `[profile:NAME]` bundles calculator/run options without a model.
- Section names such as `[sp]`, `[md]`, `[neb]` scope defaults to one
  calculation type; `[general]`, `[resources]`, `[output]`, `[safety]` apply
  globally.

## INCAR-style jobs

`mliport run --incar INCAR.mliport` accepts the VASP-shaped key/value file
(INCAR-style, **not** a DFT input):

```text
CALC_TYPE = SP
MODEL_TYPE = MACE
MODEL_PATH = /models/mace-omat-0-medium.model
DEVICE = cuda
```

Generate a template and fill in the backend explicitly:

```bash
mliport template sp --engine mace --output INCAR.mace
mliport template sp --output INCAR.neutral   # backend-neutral, MODEL_TYPE=REQUIRED
```

## Backend-specific options

| Option | Backend | Meaning |
|---|---|---|
| `TASK` / `--task` | UMA | UMA task family (`omat`, `omol`, `oc20`, ...) |
| `TASK` / `--task` | MACE/DPA/GRACE | explicit PBC semantic (`bulk`/`molecule`) |
| `HEAD` / `--head` | DPA, MACE | checkpoint branch/head (`default` for many MACE models) |
| `INFERENCE_MODE` | UMA | `default`/`turbo`/`trained` inference preset |
| `ACTIVATION_CHECKPOINTING` | UMA | GPU memory saving |
| `DEFAULT_DTYPE` | MACE | model precision selection |
| `GPU_MEMORY_GROWTH`, `GPU_MEMORY_LIMIT_MB` | GRACE | TensorFlow GPU memory policy |
| `NEIGHBOR_CACHE`, `NEIGHBOR_SKIN` | GRACE | neighbor-list cache policy |
| `TORCH_NUM_THREADS` | UMA/MACE/DPA | CPU intra-op threads |

Supplying a backend-specific option to a different backend is an error under
strict configuration; the option is never silently dropped.

## Scientific defaults

Scientific defaults (temperature, timestep, steps, thermostat, optimizer,
fmax, NEB settings, ...) live in mliport's built-in defaults, are scoped per
calculation type, and are recorded per run in `resolved_config.json`. Inspect
the authoritative list with:

```bash
mliport config schema
mliport config show --settings settings.ini
```

## Output control

| Key | Meaning |
|---|---|
| `DEVICE` / `--device` | `cpu`, `cuda`, `cuda:N` (requested device is recorded separately from the actual one) |
| `OUTPUT_DIR` / `--output` | run directory |
| `WRITE_OUTCAR`, `WRITE_XDATCAR`, `WRITE_TRAJECTORY` | output files |
| `WRITE_FORCES`, `WRITE_STRESS` | result arrays/columns |

NEB rejects disabling its mandatory outputs rather than producing a silently
incomplete result.

## Environment variables

| Variable | Effect |
|---|---|
| `MLIPORT_SETTINGS` | explicit settings.ini path |
| `MLIPORT_JOBS_DIR` | queue/job registry directory |
| `MLIPORT_CANCEL_GRACE_SECONDS` | cancellation grace period |
| `MLIPORT_LEASED_GPU_UUID` | single-GPU lease used by queue workers |
| `CUDA_VISIBLE_DEVICES` | device visibility; set by the queue when it leases a GPU — do not override it inside a running job |

## TUI

`mliport tui` exposes the same configuration (including an explicit backend
selection) and blocks a run until a backend and model are chosen. The TUI
submits through the same queue helper as the CLI, so the two entry points
cannot drift.
