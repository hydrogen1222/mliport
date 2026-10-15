# RDF

`rdf` computes the radial distribution function between a center species and a
neighbor species, together with a coordination number.

```bash
mliport analyze runs/md rdf --center Li --neighbor S --rmax 6.0 --bins 200
```

Both species are required. For LGPS the useful pairs are Li–Li, Li–S and P–S.

## Options you will use

| Option | Meaning |
|---|---|
| `--center SPECIES`, `--neighbor SPECIES` | the pair (both required) |
| `--rmax` | maximum radius in Å |
| `--bins` | histogram bins (default 200) |
| `--cn-cutoff` | the cutoff in Å used for the coordination number |
| `--stride` | use every Nth frame (default 1) |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | the analysis window |
| `--positions-convention`, `--frame-interval-fs` | metadata for external trajectories |

`--start-frame`, `--stop-frame` and `--stride` keep the cost down on long
trajectories:

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns rdf \
  --center Li --neighbor S --start-frame 0 --stop-frame 2000 --stride 5
```

## What comes out

The partial RDF, the coordination number and CSV, PNG and SVG artifacts under
`<run>/analysis/rdf/<analysis-id>/`. An unknown species, or too few valid frames,
fails closed, and an external trajectory needs its coordinate convention stated.

## Reading it

An RDF is a structural average, not a bond definition. Report the cutoff you used
for the coordination number and the window you averaged over. A single snapshot is
a diagnostic; a converged RDF needs enough frames and a homogeneous cell.

```bash
mliport analyze runs/md-800K rdf --center Li --neighbor O \
  --rmax 8 --bins 240 --cn-cutoff 3.0 --stride 5
```
