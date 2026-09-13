# mliport documentation

mliport is a **backend-neutral workflow layer for machine-learned
interatomic potentials (MLIPs)**. It runs third-party models — UMA/fairchem,
MACE, DPA/DeePMD-kit and GRACE/tensorpotential — through one CLI, TUI and
Python API for single points, relaxation, molecular dynamics, NEB and
trajectory analysis.

It is **not** a model, a trainer, or a DFT code. Absolute energies are only
comparable inside one model/head/reference combination; model accuracy is not
mliport accuracy.

## Start here

| I want to... | Read |
|---|---|
| install mliport and run a first calculation in 5-10 minutes | [quickstart.md](quickstart.md) |
| understand installation options, environments and Python constraints | [installation.md](installation.md) |
| choose or understand a model/backend | [models.md](models.md), [backends/](backends/mace.md) |
| configure a run (CLI, INCAR, settings.ini, strictness) | [configuration.md](configuration.md) |
| run a workflow | [workflows/](workflows/single-point.md) |
| analyse a trajectory | [analysis/](analysis/overview.md) |
| understand the output files and provenance | [outputs.md](outputs.md) |
| know what has actually been validated | [validation.md](validation.md) |
| fix a problem | [troubleshooting.md](troubleshooting.md) |
| migrate from the old `mlipx` name | [migration-from-mlipx.md](migration-from-mlipx.md) |
| cite mliport and its upstream projects | [citations.md](citations.md) |
| contribute | [development.md](development.md) |

## Backend pages

- [MACE](backends/mace.md)
- [DPA (DeepMD-kit)](backends/dpa.md)
- [GRACE](backends/grace.md)
- [UMA (FAIRChem)](backends/uma.md)

Each page lists the pinned profiles, install environment, minimal command,
precision/head/stress semantics, upstream citation and known limitations.

## Workflows

- [Single point](workflows/single-point.md)
- [Relaxation](workflows/optimization.md)
- [Molecular dynamics](workflows/md.md)
- [NEB / CI-NEB](workflows/neb.md)
- [Batch processing](workflows/batch.md)
- [Queue](workflows/queue.md)

## Analysis

- [Overview](analysis/overview.md)
- [MSD and local exponent](analysis/msd.md)
- [Transport (kinisi)](analysis/transport.md)
- [GEMDAT electrolyte mechanism](analysis/electrolyte-gemdat.md)
- [RDF](analysis/rdf.md)
- [VACF and spectrum](analysis/vacf-spectrum.md)
- [Arrhenius](analysis/arrhenius.md)
