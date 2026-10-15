# Overview

Every analysis is an explicit command that records where its numbers came from:

```bash
mliport analyze <RUN_DIR|trajectory> <task> [task options]
```

Analyses are diagnostics, not verdicts. An MSD exponent, a diffusivity or a jump
count has to be read together with the sampling and model caveats on its own page.

## What goes in

Prefer an mliport run directory: its metadata (frame spacing, coordinate
convention, whether frames belong to equilibration) is trusted. An ASE trajectory
(`.traj`) or an XDATCAR also works. For those external files you tell mliport what
it cannot know:

- `--positions-convention` — `wrapped`, `unwrapped` or `unknown`. If you state a
  convention that contradicts trusted metadata, mliport refuses rather than guess.
- `--frame-interval-fs` — the time between **saved** frames, which is not the MD
  integration step.

Each task also needs its own selections, such as the mobile species, a center and
neighbor pair, or a temperature.

## What comes out

Each analysis gets its own directory, `<run>/analysis/<task>/<analysis-id>/`. The
id is derived from the request, so asking the same question of the same
trajectory again reuses the stored result; add `--force` to recompute.

- `results.json` — the requested parameters and the results,
- `request.json` and `provenance.json` — the exact request and where everything
  came from (software commit, model identity, hashes of the input artifacts),
- CSV tables and PNG/SVG figures where they make sense,
- `attempts/` — an immutable record of each attempt, with `latest.json` pointing
  at the newest.

## Sampling sense

- Analyze the production part of the trajectory. Equilibration frames are left
  out unless you pass `--include-equilibration` on purpose, and
  `--start-frame` / `--stop-frame` narrow the window where a task accepts them.
- Know the saved frame spacing. For external trajectories that is
  `--frame-interval-fs`.
- Transport coefficients need longer, better equilibrated trajectories than RDF or
  thermodynamic diagnostics do.
- Running again with different windows is normal; each distinct request gets its
  own analysis id.

## The tasks without their own page

**`validate`** checks the trajectory's semantics: coordinate convention, frame
interval, which frames are production, and which later tasks it is eligible for.
It reports metadata problems rather than letting a later task produce a number
from a trajectory it should not trust.

**`thermo`** reports temperature, energy drift and pressure. `--block-size-frames`
sets the block size for the block statistics, and for an NVE run the energy drift
is the thing to look at first.

**`rmsd`** computes RMSD and RMSF from periodic displacements, a quick check
that a crystal stays a crystal. `--species` restricts it to one species and
`--drift-reference none|nonmobile|indices` chooses how drift is removed.

**`density`** builds a periodic 3-D density map of a mobile species and needs
`--mobile`. `--spacing` sets the voxel size (default 0.25 Å),
`--smoothing-sigma-A` adds smoothing, and `--stride` thins the frames.

## Commands the acceptance suite runs

The LGPS acceptance suite exercises these on real trajectories:

```bash
.venv-mace/bin/mliport analyze runs/lgps-md validate
.venv-mace/bin/mliport analyze runs/lgps-md thermo
.venv-mace/bin/mliport analyze runs/lgps-md rmsd
.venv-mace/bin/mliport analyze runs/lgps-md density --mobile Li
```

`validate` and `thermo` work on short trajectories as well as long ones. Every task
writes JSON plus CSV and, where useful, PNG and SVG figures, headlessly.
