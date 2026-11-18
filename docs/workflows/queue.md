# Job queue

The queue lets you submit several calculations, start a local scheduler, and
walk away. Each job gets its own GPU lease so two jobs never trample one
card. The scheduler tracks status and handles cancellation.

## Quick start

```bash
# Submit jobs from a task file
mliport queue submit tasks.json

# Start the scheduler (background by default)
mliport queue start

# Check progress
mliport jobs          # or: mliport queue status

# Stop the scheduler when done
mliport queue stop
```

`queue start` launches a background scheduler and returns immediately. Poll
`mliport jobs` until your jobs are `done` or `failed`. The TUI's Run screen
submits through the same helper, so jobs from either interface behave
identically.

## The task file

A task file is one JSON object with a `tasks` list. Each task specifies a
workflow, a structure, a model and its backend:

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
    },
    {
      "name": "uma-md-800K",
      "calc_type": "md",
      "python": "/data/mliport/.venv/bin/python",
      "structure": "/data/LGPS.vasp",
      "model": "/data/uma-s-1p2.pt",
      "model_type": "uma",
      "task": "omat",
      "device": "cuda",
      "options": {
        "ensemble": "NVT",
        "temperature": 800,
        "steps": 50000
      }
    }
  ]
}
```

### Key fields

- `calc_type`: one of `sp`, `opt`, `md` or `neb`
- `options`: run options passed to the workflow (same as CLI flags)
- `python`: optional path to a different backend's Python interpreter, so a
  single queue can mix engines (e.g. MACE in one venv and DPA in another)
- Model path and backend must always be explicit

## Scheduler commands

| Command | Option | Meaning |
|---|---|---|
| `queue submit` | task file | Enqueue every task in it |
| `queue start` | `--max-concurrent N` | How many jobs may run at once (default 1) |
| `queue start` | `--max-jobs-per-gpu N` | Jobs sharing one physical GPU (default 1 = exclusive) |
| `queue start` | `--poll SECONDS` | Poll interval (default 5 s) |
| `queue start` | `--foreground` | Run in foreground (Ctrl-C stops it) |
| `queue status` | — | Show pending, running and finished jobs |
| `queue stop` | — | Stop the background scheduler |
| `queue pause` / `resume` | optional job id | Pause/resume one job or all pending dispatch |

`mliport kill` and `mliport clean` operate on individual job records.

## GPU management

Each job gets a physical GPU lease. The scheduler sets `CUDA_VISIBLE_DEVICES`
to the leased GPU — do not override it inside a job.

- A job that cannot get a device stays pending; it does not squeeze onto a GPU
  beyond `--max-jobs-per-gpu`.
- A crash or exception marks the job failed, records the reason, and releases
  the lease.
- A restarted scheduler recovers stale claims.

Raise `--max-jobs-per-gpu` only for small systems that fit in GPU memory
together.

## Where things go

The job registry lives under `MLIPORT_JOBS_DIR` (a user-level directory by
default). Each job also gets its normal run directory with `run.log` and
`mliport_results.json`.

## Error handling

A job that fails is marked failed with its reason and the lease is released.
The queue continues with the next job. The scheduler itself is resilient:
killing and restarting it picks up where it left off.

## Tips for campaigns

- Queue order does not change the science, but jobs in one queue share a
  configuration surface. For transport or NEB campaigns, keep the model
  identity, timestep and save interval consistent in the task specs so
  results stay comparable.
- For large campaigns, use `--max-concurrent` to match your available GPUs
  and `--max-jobs-per-gpu 1` (the default) to avoid memory contention.
- The `mliport state` command shows the current state of the job registry
  across all queues.
