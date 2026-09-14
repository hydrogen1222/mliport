# Workflows

| Workflow | Command | Manual |
|---|---|---|
| Single point | `mliport sp` | [single-point.md](single-point.md) |
| Relaxation | `mliport opt` | [optimization.md](optimization.md) |
| Molecular dynamics | `mliport md` | [md.md](md.md) |
| NEB / CI-NEB | `mliport neb` | [neb.md](neb.md) |
| Batch sweeps | `mliport batch` | [batch.md](batch.md) |
| Queue | `mliport queue` | [queue.md](queue.md) |

`mliport run` executes an INCAR-style file and can start any of the workflows
above; `mliport template` writes a commented template. Every workflow records
`resolved_config.json` next to its results.
