# Workflows

Everything mliport computes is one of a handful of workflows. They share the
same model selection, the same configuration layers and the same habit of
writing a `resolved_config.json` next to their results.

| Workflow | Command | What it is for |
|---|---|---|
| [Single point](single-point.md) | `mliport sp` | energy, forces and stress of one structure |
| [Relaxation](optimization.md) | `mliport opt` | minimize forces, optionally with the cell |
| [Molecular dynamics](md.md) | `mliport md` | NVE or NVT trajectories for analysis |
| [NEB / CI-NEB](neb.md) | `mliport neb` | a reaction path and its barrier |
| [Configuration search](confsearch.md) | `mliport confsearch` | low-energy adsorbate configurations on a slab |
| [Batch](batch.md) | `mliport batch` | one calculation over many structures |
| [Queue](queue.md) | `mliport queue` | several jobs, one GPU at a time |

`mliport run` executes an INCAR-style file and can start any of the workflows
above; `mliport template` writes a commented template to start from.
`mliport tui` opens the same workflows in an interactive terminal application.

## Exit codes

The CLI, the TUI and the queue agree on how a run ended, so a script can tell
"finished" from "ran but did not converge":

| Exit code | Meaning |
|---|---|
| `0` | completed |
| `1` | failed (error, bad configuration, runtime failure) |
| `2` | finished but did not converge; partial results and the reason are kept |
| `130` | cancelled (interrupt or cooperative cancellation) |

A relaxation or NEB result without a convergence flag is treated as not
converged. mliport will not call a run a success because the process happened
to exit cleanly.

## Following a run

Every run writes a live `run.log` in its output directory, and prints a command
for following it (`tail -f`). Long runs, NEB included, report progress there as
they go, so a quiet terminal does not mean a stuck job.
