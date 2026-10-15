# Analysis

`mliport analyze <run-or-trajectory> <task>` reads an mliport run directory or an
ASE trajectory and writes its results to `<run>/analysis/<task>/<analysis-id>/` as
JSON, CSV and, where a figure helps, PNG and SVG. Plots render headlessly, so no
display is needed.

| Task | What it tells you | Page |
|---|---|---|
| `validate` | whether the trajectory is usable, and for which tasks | [Overview](overview.md) |
| `thermo` | temperature, energy drift and pressure | [Overview](overview.md) |
| `rmsd` | RMSD and RMSF from periodic displacements | [Overview](overview.md) |
| `density` | a 3-D density map of the mobile ion | [Overview](overview.md) |
| `rdf` | partial radial distribution and coordination | [RDF](rdf.md) |
| `msd` | directional mean-squared displacement and its local exponent | [MSD](msd.md) |
| `vacf` | velocity autocorrelation | [VACF and spectrum](vacf-spectrum.md) |
| `spectrum` | the vibrational spectrum derived from the VACF | [VACF and spectrum](vacf-spectrum.md) |
| `transport` | tracer diffusion and Nernst–Einstein conductivity (kinisi) | [Transport](transport.md) |
| `electrolyte` | sites, jumps and percolation (GEMDAT) | [GEMDAT](electrolyte-gemdat.md) |
| `arrhenius` | an Arrhenius fit to diffusivities from several temperatures | [Arrhenius](arrhenius.md) |

Short trajectories are valid input, but a task that would over-interpret one
says so. It returns an explicit "insufficient window" or "insufficient sampling"
status instead of a number.
