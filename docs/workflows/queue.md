# Queue

The queue lets you hand mliport several calculations and walk away. A local
scheduler runs them for you, gives each job its own GPU lease so two jobs don't
trample one card, tracks status and handles cancellation.

```bash
.venv-mace/bin/mliport queue submit tasks.json
.venv-mace/bin/mliport queue start
.venv-mace/bin/mliport jobs
.venv-mace/bin/mliport queue stop
```

`queue start` launches a background scheduler and returns right away. Poll
`mliport jobs` (or `queue status`) until your jobs are `done` or `failed`, then
stop the scheduler. The TUI's Run screen submits through the same helper as the
CLI, so jobs from either behave identically.

## The task file

A task file is one JSON object with a `tasks` list. `calc_type` is one of `sp`,
`opt`, `md` or `neb`, and the paths must exist:

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
      "options": {"ensemble": "NVT", "temperature": 800, "steps": 50000}
    }
  ]
}
```

Run options go in `options`. An optional `python` runs that one task in another
backend's environment, so a single queue can mix engines. The model path and the
backend must always be explicit.

## Controlling the scheduler

| Command | Option | Meaning |
|---|---|---|
| `queue submit` | the task file | enqueue every task in it |
| `queue start` | `--max-concurrent N` | how many jobs may run at once (default 1, for a single GPU) |
| `queue start` | `--max-jobs-per-gpu N` | how many jobs may share one physical GPU (default 1, an exclusive lease); raise it only for small systems that fit in GPU memory together |
| `queue start` | `--poll SECONDS` | scheduler poll interval (default 5) |
| `queue start` | `--foreground` | run the scheduler in the foreground (Ctrl-C stops it) |
| `queue status` | | pending, running and finished jobs with their device lease |
| `queue stop` | | stop the background scheduler |
| `queue pause` / `queue resume` | optional job id | pause or resume one pending job, or all pending dispatch |

`kill` and `clean` work on individual job records.

## Where things go

The job registry lives under `MLIPORT_JOBS_DIR` (a user-level directory by
default). Each job also gets its normal run directory with `run.log` and
`mliport_results.json`, and the physical GPU lease is recorded per job.

## What happens when things go wrong

A job that cannot get a device stays pending; it does not squeeze onto a GPU
beyond `--max-jobs-per-gpu`. A crash or an exception marks the job failed with
its reason and releases the lease. A scheduler that is restarted recovers stale
claims. The queue turns `CUDA_VISIBLE_DEVICES` into the leased physical GPU, so
don't override it inside a job.

Queue order does not change the science, but jobs in one queue share a
configuration surface. For transport or NEB campaigns, keep the model identity,
timestep and save interval in the task spec so the resulting records stay
comparable.
