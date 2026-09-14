# Analysis overview

mliport analyses are explicit, provenance-recording subcommands:

```bash
mliport analyze <RUN_DIR|trajectory> <task> [task options]
```

| Task | Question it answers | Page |
|---|---|---|
| `msd` | how far do the mobile species move, and in which regime? | [msd.md](msd.md) |
| `transport` | diffusivity / ionic conductivity with uncertainty | [transport.md](transport.md) |
| `electrolyte` | site-to-site jumps and mechanisms (GEMDAT) | [electrolyte-gemdat.md](electrolyte-gemdat.md) |
| `rdf` | local structure around a species | [rdf.md](rdf.md) |
| `vacf` / `spectrum` | velocity correlations and vibrational spectrum | [vacf-spectrum.md](vacf-spectrum.md) |
| `arrhenius` | temperature trend of diffusivity | [arrhenius.md](arrhenius.md) |
| `thermo`, `density`, `rmsd` | energy/temperature/density and structural drift diagnostics | this page |

## Inputs

- an mliport run directory (preferred: its own metadata is trusted), an ASE
  trajectory (`.traj`) or an XDATCAR;
- for external trajectories: `--positions-convention` (wrapped/unwrapped) and
  `--frame-interval-fs` — mliport refuses to guess when trusted metadata
  conflicts with a stated convention;
- the mobile species / centre / neighbour selections required by each task.

## Outputs

Each analysis writes to `<run>/analysis/<analysis_id>/` with:

- `analysis.json` (requested parameters, resolved provenance, results),
- plots/tables where applicable,
- the software commit, model identity and input artifact hashes used.

Analyses are **diagnostics**, not automatic scientific verdicts. An MSD
exponent, a diffusivity or a jump count must be interpreted with the sampling
and model caveats on the relevant page.

## Sampling sanity checks

- Use the production part of the trajectory (`--start-frame`,
  `--include-equilibration` only when you mean it).
- Know the **saved** frame spacing (`--frame-interval-fs` for external
  trajectories); it is not the integration timestep.
- Longer, well-equilibrated trajectories are required for transport
  coefficients than for RDF/thermo diagnostics.
- Re-running with different windows is expected and recorded via `--force`
  and analysis IDs.

## Acceptance-verified invocations

The LGPS acceptance suite exercises these commands on real trajectories:

```bash
# tested example
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
```

`density` needs `--mobile`. `validate` and `thermo` run on short trajectories
as well as long ones; they report metadata problems instead of producing a
number when the trajectory is not suitable for a later task. Every task
writes JSON plus CSV and, where useful, PNG/SVG figures, all headless.
