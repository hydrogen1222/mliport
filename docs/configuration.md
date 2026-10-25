# Configuration

Most of the time you will set everything on the command line. When runs get
more involved you can move settings into a `settings.ini`, a model alias or an
INCAR-style job file. They all feed one resolver, which applies the layers in
this order, each overriding the one before:

```text
built-in defaults
  < settings.ini
  < model alias ([model:NAME])
  < profile ([profile:NAME])
  < INCAR / job file
  < CLI flags / API arguments
```

The resolver never merges two layers' ideas of an option. Each option ends up
with exactly one value from exactly one source, and that source is written to
`resolved_config.json` (see [Output files](outputs.md)). When you wonder why a
run behaved as it did, that file has the answer.

## What a run needs

| What | Where it comes from |
|---|---|
| calculation type | the subcommand (`sp`, `opt`, `md`, `neb`, `batch`, `confsearch`), or `CALC_TYPE` / `CALCULATION` |
| backend | `--model-type` / `MODEL_TYPE`, or a model alias that declares `engine` |
| model path | `--model` / `MODEL_PATH`, or the alias's `path` |
| structure | the positional argument, or `--structure` / the INCAR structure keys with `run` |
| DPA branch | `--head` / `HEAD`, when the checkpoint is multi-task |
| NEB endpoints | `--initial` / `--final`, or `NEB_INITIAL` / `NEB_FINAL` |
| confsearch starts | `CS_STARTS` + `CS_ADSORBATE_INDICES`, or `--slab` / `--molecule` / `--n-starts` |

There is no implicit UMA default, or any other. If no layer names a backend,
the resolver stops:

```text
Error: No MLIP backend was selected.

Choose one of:
  --model-type mace
  --model-type dpa
  --model-type grace
  --model-type uma

or use a configured model alias/profile.
```

## Strict by default

A misspelled key, or a key that belongs to another backend, is a fatal error.
The alternative — a warning and a run that carries on with the wrong settings —
is exactly how a wrong result gets produced and trusted.

```bash
mliport sp structure.cif --model m.model --model-type mace --inference-mode turbo
# Error: Option 'inference_mode' is not applicable to engine 'mace' ...
```

For exploratory work you can switch to the old warn-and-continue behavior, but
only on purpose:

```bash
mliport sp ... --lenient-config
```

```ini
[general]
strict_config = false
```

To look at your configuration, use `mliport config validate settings.ini` to
check a file and `mliport config explain <KEY>` to see where one value came
from. `mliport config paths` lists where `settings.ini` is searched for,
`mliport config init` writes a starter file, `mliport config show` prints the
resolved configuration, and `mliport config schema` lists every recognized key.

## settings.ini

The search paths, highest priority first, are printed by `mliport config paths`.
Set `MLIPORT_SETTINGS` to point at one explicit file; the global
`--settings PATH` flag beats even that.

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

- A `[model:NAME]` section must declare `engine` (`uma`, `mace`, `dpa` or
  `grace`; `fairchem` is another name for `uma`).
- A `[profile:NAME]` section bundles calculator and run options without
  naming a model.
- Sections named after a calculation type (`[sp]`, `[md]`, `[neb]`, …) set
  defaults for that type only. `[general]`, `[resources]`, `[output]` and
  `[safety]` apply everywhere.

## INCAR-style jobs

`mliport run --incar INCAR.mliport` reads a key/value file in the style of a
VASP INCAR. It is only the style: this is not a DFT input.

```text
CALC_TYPE = SP
MODEL_TYPE = MACE
MODEL_PATH = /models/mace-omat-0-medium.model
DEVICE = cpu                        # explicit cuda / cuda:0 for GPU runs
```

Let mliport write a template, and fill in the backend yourself:

```bash
mliport template sp --engine mace --output INCAR.mace
mliport template sp --output INCAR.neutral   # backend-neutral, MODEL_TYPE=REQUIRED
```

## Options that belong to one backend

| Option | Backend | What it does |
|---|---|---|
| `TASK` / `--task` | UMA | task family (`omat`, `omol`, `oc20`, …) |
| `TASK` / `--task` | MACE, DPA, GRACE | explicit PBC semantic (`bulk` / `molecule`) |
| `HEAD` / `--head` | DPA, MACE | checkpoint branch or head (`default` for many MACE models) |
| `INFERENCE_MODE` | UMA | `default`, `turbo` or `trained` inference preset |
| `ACTIVATION_CHECKPOINTING` | UMA | saves GPU memory |
| `DEFAULT_DTYPE` | MACE | model precision |
| `GPU_MEMORY_GROWTH`, `GPU_MEMORY_LIMIT_MB` | GRACE | TensorFlow GPU memory policy |
| `NEIGHBOR_CACHE`, `NEIGHBOR_SKIN` | GRACE | neighbor-list cache policy |
| `TORCH_NUM_THREADS` | UMA, MACE, DPA | CPU intra-op threads |

Handing one of these to a different backend is an error under strict
configuration. The option is never silently dropped.

## Dispersion correction (DFT-D3)

An MLIP checkpoint does not contain a dispersion correction. A model trained on
plain PBE data (MACE `omat` heads, UMA `omat`, most materials models) has learned
the PBE potential energy surface, which has no long-range dispersion. `--d3`
adds Grimme's DFT-D3 energy, forces and stress on top of the model, on every
backend, through `torch-dftd`:

| Option | Default | What it does |
|---|---|---|
| `D3` / `--d3` | off | add DFT-D3 to the model |
| `D3_XC` / `--d3-xc` | `pbe` | reference functional of the model's training data; selects the D3 parameters |
| `D3_DAMPING` / `--d3-damping` | `bj` | damping function: `zero`, `bj`, `zerom` or `bjm` |

```bash
uv pip install --python .venv-mace/bin/python torch-dftd
mliport opt slab.vasp --model model.model --model-type mace --task bulk --d3
```

Only add D3 to a model whose training reference has no dispersion, and name
that reference with `--d3-xc`; mliport cannot read it from a MACE, DPA or GRACE
checkpoint. For UMA the reference of each task is known: D3 is accepted for
`omat` (PBE) and `oc20` (RPBE) only, and refused for `omol`, `omc`, `odac` and
`oc25`, whose references already contain dispersion or are not verified to be
dispersion-free. `D3_XC` or `D3_DAMPING` without `D3` is an error rather than a
setting that silently does nothing. NEB does not accept D3 yet, and the TUI
does not expose it.

The correction is recorded with the run: `dispersion` in the model metadata of
the JSON output and a `Dispersion:` line in the OUTCAR-style header. The
`torch-dftd` package needs PyTorch, which the MACE, DPA and UMA environments
already have.

## Scientific defaults

Defaults for temperature, timestep, step counts, thermostat, optimizer, `fmax`,
NEB settings and so on live in mliport's built-in defaults, are scoped to the
calculation type, and are recorded for every run in `resolved_config.json`.
The authoritative list is one command away:

```bash
mliport config schema
mliport config show --settings settings.ini
```

## Output control

| Key | Meaning |
|---|---|
| `DEVICE` / `--device` | `cpu`, `cuda` or `cuda:N`; the requested device is recorded separately from the one actually used |
| `OUTPUT_DIR` / `--output` | the run directory |
| `WRITE_OUTCAR`, `WRITE_XDATCAR`, `WRITE_TRAJECTORY` | which files are written |
| `WRITE_FORCES`, `WRITE_STRESS` | which arrays and columns are written |

NEB refuses to switch off its mandatory outputs, rather than hand you a result
that quietly lacks them.

## Environment variables

| Variable | Effect |
|---|---|
| `MLIPORT_SETTINGS` | explicit `settings.ini` path |
| `MLIPORT_JOBS_DIR` | queue and job registry directory |
| `MLIPORT_CANCEL_GRACE_SECONDS` | grace period when cancelling a job |
| `MLIPORT_LEASED_GPU_UUID` | the single GPU leased to a queue worker |
| `CUDA_VISIBLE_DEVICES` | device visibility; the queue sets it when it leases a GPU, so do not override it inside a running job |

## The TUI

`mliport tui` exposes the same configuration, including an explicit backend
choice, and will not start a run until a backend and a model are chosen. It
submits through the same queue helper as the CLI, so the two cannot drift apart.
