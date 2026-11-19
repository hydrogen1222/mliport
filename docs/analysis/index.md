# Trajectory analysis

`mliport analyze <run-directory-or-trajectory> <task>` reads an mliport run
directory or a standalone ASE trajectory and writes results to
`<run>/analysis/<task>/<analysis-id>/`. Every analysis produces a JSON summary; most also output CSV
or NPZ data files. Tasks that plot also output PNG and SVG. No display server is needed.

## Available tasks

| Task | Physical quantity | Page |
|---|---|---|
| `validate` | Trajectory integrity: coordinate convention, frame interval, sampling readiness | [Validate](validate.md) |
| `thermo` | Temperature, energy drift and pressure over time | [Thermo](thermo.md) |
| `rmsd` | Root-mean-square displacement and fluctuation (periodic) | [RMSD](rmsd.md) |
| `density` | 3-D number density of a mobile species | [Density](density.md) |
| `rdf` | Radial distribution function g(r) and coordination number | [RDF](rdf.md) |
| `msd` | Mean-squared displacement and local exponent α(t) | [MSD](msd.md) |
| `vacf` | Velocity autocorrelation function | [VACF & spectra](vacf-spectrum.md) |
| `spectrum` | Vibrational density of states from the VACF | [VACF & spectra](vacf-spectrum.md) |
| `transport` | Tracer diffusivity and Nernst–Einstein conductivity (via kinisi) | [Transport](transport.md) |
| `electrolyte` | Crystallographic site mapping, jump analysis and percolation (via GEMDAT) | [GEMDAT](electrolyte-gemdat.md) |
| `arrhenius` | Arrhenius fit of diffusivity vs. temperature | [Arrhenius](arrhenius.md) |

## Which analysis do I need?

The right analysis depends on what question you are asking about your MD
trajectory.

```
After running MD
│
├─ "Did the simulation stay healthy?"
│   └─ validate → thermo → rmsd
│
├─ "Is there ion transport?"
│   └─ msd → transport (if α ≈ 1)
│
├─ "What is the ionic conductivity?"
│   └─ transport → arrhenius (multiple temperatures)
│
├─ "How do the ions move?"
│   └─ electrolyte (site mapping, jump analysis)
│       └─ density (visualize ion pathways)
│
├─ "What is the local structure?"
│   └─ rdf (pair correlations, coordination)
│
└─ "What are the vibrational modes?"
    └─ vacf → spectrum
```

A typical solid-electrolyte workflow runs `validate` → `thermo` → `rmsd` →
`msd` → `transport` → `electrolyte`, adding `arrhenius` once you have
diffusion coefficients at three or more temperatures.

## Sampling considerations

Short trajectories can be analyzed — mliport does not refuse them. However,
if a trajectory is too short to support a reliable result, the analysis returns
an explicit **"insufficient sampling"** or **"insufficient window"** status
instead of a number you should not trust.

General guidelines:

- **Thermodynamics** (`thermo`, `rmsd`): useful even for short runs (hundreds
  of frames). Good for quick sanity checks.
- **Structural** (`rdf`, `density`): need enough frames for statistical
  averaging. A few thousand frames usually suffice.
- **Transport** (`msd`, `transport`, `electrolyte`): need long, well-
  equilibrated trajectories. The required length depends on the diffusion
  coefficient — slower diffusion needs longer runs. For solid electrolytes,
  plan for at least 100 ps of production at each temperature.
- **Arrhenius**: needs converged diffusion coefficients at ≥ 3 temperatures.
  Each temperature is an independent MD run.

See [Overview](overview.md) for input conventions and output formats that
apply to all tasks.
