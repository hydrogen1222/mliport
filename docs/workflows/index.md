# Workflows

mliport organizes every calculation into a **workflow** — a self-contained unit
that takes a structure and a model, runs the physics, and writes a complete
record of what happened. All workflows share the same model selection logic,
the same [configuration layers](../configuration.md), and the same habit of
writing `resolved_config.json` next to their results so you can always
reconstruct exactly what was run.

## Choosing a workflow

| If you need to… | Use | Command |
|---|---|---|
| Check that a model loads and produces sensible forces | [Single point](single-point.md) | `mliport sp` |
| Find the nearest local minimum of the potential energy surface | [Relaxation](optimization.md) | `mliport opt` |
| Sample thermodynamic ensembles or generate trajectories for transport analysis | [Molecular dynamics](md.md) | `mliport md` |
| Estimate a reaction or migration barrier between two states | [NEB / CI-NEB](neb.md) | `mliport neb` |
| Screen low-energy adsorbate configurations on a slab | [Configuration search](confsearch.md) | `mliport confsearch` |
| Run the same calculation over many structures with one model load | [Batch](batch.md) | `mliport batch` |
| Queue several jobs on shared GPUs and walk away | [Queue](queue.md) | `mliport queue` |

A typical project follows these stages: **single point** → confirm the model
loads and forces look physical → **relaxation** → obtain the equilibrium
structure → **MD** → generate a trajectory for property extraction →
**analysis**. NEB and configuration search are specialized workflows for
barrier finding and surface adsorption screening respectively.

## Three ways to start a workflow

Every workflow can be launched from:

1. **The command line** — `mliport sp`, `mliport opt`, `mliport md`, etc.
2. **An INCAR-style file** — `mliport run INCAR` reads a keyword file
   (`CALC_TYPE = sp`, `MODEL_TYPE = mace`, …) and dispatches the workflow.
   `mliport template` writes a commented template to start from.
3. **The Python API** — `from mliport.api import run_single_point` and friends,
   for embedding calculations in scripts and notebooks.
4. **The terminal UI** — `mliport tui` opens an interactive application that
   exposes the same workflows with guided menus.

## Exit codes

The CLI, the TUI and the queue all use the same exit code convention, so
wrapper scripts can tell "finished" from "finished but did not converge"
without parsing log files:

| Exit code | Meaning |
|---|---|
| `0` | Completed successfully |
| `1` | Failed — bad configuration, runtime error, or missing input |
| `2` | Finished but did not converge; partial results and the reason are preserved |
| `130` | Cancelled by interrupt or cooperative cancellation |

A relaxation or NEB result without a convergence flag is treated as **not
converged**. mliport will not call a run successful just because the process
exited cleanly.

## Following a run

Every run writes a live `run.log` in its output directory and prints a
`tail -f` command you can copy. Long runs — NEB and MD in particular —
report progress there as they go, so a quiet terminal does not mean a
stuck job.

## Model identity and reproducibility

Every workflow records a **model identity** in `run_context.json`: the
backend, model path, file hash, task, head, dtype, inference mode, device
and any DFT-D3 correction. Two results are only comparable when their model
identities match. Absolute energies from different models, or from the same
model with different heads or tasks, cannot be mixed.
