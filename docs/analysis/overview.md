# Overview

Every analysis is an explicit command that records where its numbers came from:

```bash
mliport analyze <RUN_DIR|trajectory> <task> [task options]
```

Analyses are diagnostics, not verdicts. An MSD exponent, a diffusivity or a jump
count has to be read together with the sampling and model caveats on its own page.

## What analysis does

mliport's analysis tools turn an MD trajectory into physical quantities —
pair correlations, diffusion coefficients, activation energies, vibrational
spectra, site-resolved jump statistics. The trajectory is the raw material;
the analysis tasks are the instruments that read it.

All analyses share a few invariants:

- **MLIP trajectories are not DFT trajectories.** The forces come from the
  potential, and the potential is only as good as its training data. An analysis
  can tell you what the model predicts, not what the real material does.
- **Analysis is separate from the run.** The trajectory is written once; you
  can analyze it as many times as you like, with different windows, species,
  and parameters. Each request gets its own analysis directory.
- **Provenance is automatic.** Every result records the software version, the
  model identity, the input file hashes and the exact request, so you can
  reconstruct what produced a number without searching through logs.

## What goes in

### mliport run directories (preferred)

Prefer an mliport run directory: its metadata — frame spacing, coordinate
convention, which frames are equilibration, the model identity and the run
configuration — is trusted and does not need to be restated.

The analysis reader finds `raw/trajectory.traj` (or `raw/md.csv` for scalar
time series) and `artifacts.json` (the manifest that records units, observable
definitions and file hashes). It knows which frames are production and which
are equilibration from the run's own metadata.

### External trajectories

An ASE trajectory (`.traj`) or a VASP XDATCAR also works. For those external
files you tell mliport what it cannot know:

- `--positions-convention` — `wrapped`, `unwrapped` or `unknown`. If you state
  a convention that contradicts trusted metadata, mliport refuses rather than
  guess. The convention matters because transport analyses need unwrapped
  coordinates, while structural analyses (RDF) work with either.
- `--frame-interval-fs` — the time between **saved** frames, which is not the
  MD integration step. This sets the physical time axis for every time-
  dependent analysis (MSD, VACF, transport). Getting it wrong shifts every
  diffusion coefficient and every spectral frequency.

Each task also needs its own selections, such as the mobile species, a center
and neighbor pair, or a temperature.

### XDATCAR and PBC caveats

Standard VASP XDATCAR cannot encode the ASE `pbc` mask. For a non-3D-periodic
MD input the export warns that the mask is lost, while `artifacts.json` records
the original `trajectory.pbc` array. When loading a matching mliport
`vasp/XDATCAR`, the analysis reader restores that mask from the manifest. A
standalone or older XDATCAR without matching source-PBC metadata has
**unverified PBC**; even if ASE parses it with `pbc=[True,True,True]`, mliport
does not certify it for bulk RDF/MSD or transport. Analyze the original
`raw/trajectory.traj` when available.

## What comes out

Each analysis gets its own directory, `<run>/analysis/<task>/<analysis-id>/`.
The id is derived from the request, so asking the same question of the same
trajectory again reuses the stored result; add `--force` to recompute.

- `results.json` — the requested parameters and the results,
- `request.json` and `provenance.json` — the exact request and where everything
  came from (software commit, model identity, hashes of the input artifacts),
- CSV tables and PNG/SVG figures where they make sense,
- `attempts/` — an immutable record of each attempt, with `latest.json`
  pointing at the newest.

### Analysis IDs and caching

The analysis ID encodes the task, the species, and the key options (fit window,
bin count, etc.). Changing any of these gives a different ID and a separate
result directory. This means you can sweep over parameters without overwriting
earlier results — and it means that repeating the exact same command is free.

Use `--force` to recompute an existing ID, for example after extending the
trajectory or updating the analysis code.

## Sampling sense

- Analyze the production part of the trajectory. Equilibration frames are left
  out unless you pass `--include-equilibration` on purpose, and
  `--start-frame` / `--stop-frame` narrow the window where a task accepts them.
- Know the saved frame spacing. For external trajectories that is
  `--frame-interval-fs`.
- Transport coefficients need longer, better equilibrated trajectories than RDF
  or thermodynamic diagnostics do.
- Running again with different windows is normal; each distinct request gets
  its own analysis id.

### How much trajectory is enough?

| Analysis | Minimum useful | Recommended |
|---|---|---|
| `validate`, `thermo` | Hundreds of frames | Any production length |
| `rmsd`, `rdf`, `density` | A few thousand frames | 10+ ps |
| `msd` | Enough to reach α ≈ 1 | 50+ ps at the target temperature |
| `transport` | Diffusive regime visible | 100+ ps per temperature |
| `electrolyte` | Enough jumps | 100+ ps; longer for slow diffusers |
| `arrhenius` | 3+ temperatures | 3–5 temperatures, each converged |

These are rough guidelines. Slower diffusers need proportionally longer runs.

## The tasks without their own page

### validate

**`thermo`** reports temperature, energy drift and pressure. `--block-size-frames`
sets the block size for the block statistics, and for an NVE run the energy drift
is the thing to look at first. The summary statistics use the finite frames only
(NaN and ±Inf alike, in the summary and in the block means),
so the result also carries `summary.data_integrity`: per column, how many frames
existed, how many were finite, and which columns had non-finite values at all
(`raw/md.csv` cells that are empty or unparsable become NaN). A mean computed
from a partially corrupted column is visible instead of looking healthy.

```bash
mliport analyze runs/md validate
```

A trajectory that passes `validate` is not guaranteed to be scientifically
meaningful — it only means the metadata is self-consistent and the file is
well-formed.

### thermo

`thermo` reports temperature, energy drift and pressure over time.
`--block-size-frames` sets the block size for the block statistics.

```bash
mliport analyze runs/md thermo --block-size-frames 100
```

For an NVE run, the energy drift is the first thing to look at: it measures
how well the integrator conserves energy. A drift above ~1 meV/atom/ps
suggests the timestep is too large or the model's forces are noisy.

For NVT, check that the temperature fluctuates around the target with the
expected variance. For NPT, the pressure and volume traces show whether the
barostat has equilibrated.

### rmsd

`rmsd` computes RMSD and RMSF from periodic displacements — a quick check
that a crystal stays a crystal. If it melts or transforms, the RMSD grows
without bound.

```bash
mliport analyze runs/md rmsd --species Li
mliport analyze runs/md rmsd --drift-reference nonmobile
```

`--species` restricts to one species and `--drift-reference
none|nonmobile|indices` chooses how framework drift is removed. For solid
electrolytes, `--drift-reference nonmobile` removes the slow drift of the
immobile sublattice so that you see only the mobile-ion displacement.

### density

`density` builds a periodic 3-D number-density map of a mobile species.
The output is a volumetric data file that can be visualized in VESTA or
similar tools.

```bash
mliport analyze runs/md density --mobile Li --spacing 0.2 --stride 5
```

`--spacing` sets the voxel size in Å (default 0.25 Å), `--smoothing-sigma-A`
adds Gaussian smoothing, and `--stride` thins the frames for faster
computation. The density map shows where the mobile ions spend their time —
channels, cages, and preferred sites become visible.

## Commands the acceptance suite runs

The LGPS acceptance suite exercises these on real trajectories:

```bash
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
```

`validate` and `thermo` work on short trajectories as well as long ones. Every
task writes JSON plus CSV and, where useful, PNG and SVG figures, headlessly.
