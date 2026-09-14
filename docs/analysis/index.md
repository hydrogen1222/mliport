# Analysis

`mliport analyze <run-or-trajectory> <task>` reads an mliport run directory or
an ASE trajectory. Results are written to
`<run>/analysis/<task>/<analysis-id>/` as JSON, CSV and, when a figure is
useful, PNG and SVG.

| Task | Manual | What it reports |
|---|---|---|
| `validate` | [overview.md](overview.md) | semantics, frame interval, eligibility |
| `thermo` | [overview.md](overview.md) | temperature, energy drift, pressure |
| `rdf` | [rdf.md](rdf.md) | partial RDF and coordination |
| `rmsd` | [overview.md](overview.md) | periodic-displacement RMSD/RMSF |
| `msd` | [msd.md](msd.md) | directional windowed MSD and alpha |
| `vacf` | [vacf-spectrum.md](vacf-spectrum.md) | velocity autocorrelation |
| `spectrum` | [vacf-spectrum.md](vacf-spectrum.md) | VACF-derived spectrum |
| `density` | [overview.md](overview.md) | mobile-ion density map |
| `transport` | [transport.md](transport.md) | kinisi tracer diffusion and NE conductivity |
| `electrolyte` | [electrolyte-gemdat.md](electrolyte-gemdat.md) | GEMDAT sites, jumps and percolation |
| `arrhenius` | [arrhenius.md](arrhenius.md) | Arrhenius fit of independent D(T) points |

Short trajectories are valid inputs. Tasks that would over-interpret them
return an explicit insufficient-window or insufficient-sampling status rather
than a number.
