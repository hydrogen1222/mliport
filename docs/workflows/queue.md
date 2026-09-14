# Queue

**Purpose.** Submit several computations to a local queue so a single GPU is
used by one job at a time, with a device lease, cancellation and status
tracking.

**Minimal commands.**

```bash
.venv-mace/bin/mliport queue submit job.json
.venv-mace/bin/mliport queue start            # scheduler (background by default)
.venv-mace/bin/mliport queue status
.venv-mace/bin/mliport queue stop
```

**Important options.**

| Command | Option | Meaning |
|---|---|---|
| `queue submit` | task-file (positional) | JSON task spec: calc type, structure(s), model, backend, options |
| `queue start` | `--max-concurrent N` | jobs allowed to run at once (default 1 for a single GPU) |
| `queue start` | `--poll SECONDS` | scheduler poll interval (default 5) |
| `queue start` | `--foreground` | run the scheduler in the foreground (Ctrl-C stops it) |
| `queue status` | — | pending/running/finished jobs with their device lease |
| `queue stop` / `pause` / `resume` | — | stop the scheduler / pause / resume dispatching |

The TUI Run screen submits through the same queue/task-spec helper, so TUI
and CLI jobs cannot drift.

**Inputs.** A JSON task file describing the run (see `mliport queue submit`
and the TUI "Jobs" screen for the fields). Model paths and backend selection
must be explicit; the job inherits the environment of the scheduler process.

**Outputs.** A job registry under `MLIPORT_JOBS_DIR` (default: a
platform-appropriate user/jobs directory), per-job logs and run directories,
and a physical-GPU lease recorded per job.

**Failure semantics.** A job that cannot acquire a device stays pending
rather than sharing the GPU; a crash/exception is recorded as failed with its
reason, and the device lease is released. Stale claims are recovered on
scheduler start. `CUDA_VISIBLE_DEVICES` is resolved to the leased physical
GPU UUID by the queue — do not override it inside a job.

**Scientific caveats.** Queue ordering does not change the science, but jobs
in one queue share one backend environment and one configuration surface.
For transport/NEB campaigns, keep the model identity and timestep/save
interval in the job spec so the resulting records stay comparable.

**Example.**

```bash
cat > job.json <<'JSON'
{
  "calc_type": "md",
  "structure": "structure.cif",
  "model": "uma.pt",
  "model_type": "uma",
  "task": "omat",
  "device": "cuda",
  "options": {"ensemble": "NVT", "temp": 800, "steps": 50000}
}
JSON
.venv/bin/mliport queue submit job.json
.venv/bin/mliport queue start --max-concurrent 1
```

## Task file and lifecycle

A task file is one JSON object. Paths must exist; `calc_type` is one of
`sp`, `opt`, `md` or `neb`:

```json
{
  "max_concurrent": 1,
  "tasks": [
    {
      "name": "lgps-sp",
      "calc_type": "sp",
      "structure": "/data/LGPS.vasp",
      "model": "/data/mace-omat-0-medium.model",
      "model_type": "mace",
      "task": "bulk",
      "device": "cuda",
      "output_dir": "/data/runs/lgps-sp"
    }
  ]
}
```

```bash
# tested example
.venv-mace/bin/mliport queue submit tasks.json
.venv-mace/bin/mliport queue start
.venv-mace/bin/mliport jobs
.venv-mace/bin/mliport queue status
.venv-mace/bin/mliport queue stop
```

`queue start` owns a background scheduler and returns immediately; poll
`jobs` (or `queue status`) until the job reaches `done` or `failed`, then stop
the scheduler. Each job writes the usual run directory, including
`mliport_results.json`. `queue pause`/`queue resume` control pending work,
while `kill` and `clean` operate on individual job records.
