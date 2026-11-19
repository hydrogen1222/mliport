# LGPS solid-electrolyte workflow

This tutorial runs a complete MLIP-based simulation and analysis workflow
for Li₁₀GeP₂S₁₂ (LGPS) — one of the fastest known solid-state lithium
conductors. By the end you will have:

- An optimised LGPS structure
- A 1 ns NVT trajectory at 600 K
- Diffusion coefficient *D*_Li and estimated ionic conductivity

The backend used here is **MACE**; swap `mliport-mace` for `mliport-dpa` etc.
to use another engine.

## Prerequisites

A working mliport+MACE install and an LGPS CIF. If you need a structure,
the ICSD experimental geometry (space group *P4₂/nmc*, a ≈ 8.72 Å,
c ≈ 12.70 Å) is freely available from the Materials Project (mp-696128).

```bash
./scripts/install_mliport.sh --engines mace --device auto
./bin/mliport-mace doctor --engine mace
```

## Step 1 — Geometry optimisation

Relax the LGPS structure before MD. This removes residual force imbalances
from the experimental geometry.

```bash
./bin/mliport-mace opt \
  --structure lgps.cif \
  --model-type mace \
  --model-file path/to/mace-checkpoint.pt \
  --fmax 0.01 \
  --max-steps 500 \
  --cell-opt \
  --output runs/lgps-opt
```

The optimised structure is written to `runs/lgps-opt/CONTCAR`. Use this for
the MD run, not the original CIF.

## Step 2 — NVT molecular dynamics

Run 1 ns (1 000 000 steps at 1 fs) of NVT-MD at 600 K. Use an equilibration
phase of 50 ps to let the system reach thermal equilibrium before recording
the production trajectory.

```bash
./bin/mliport-mace md \
  --structure runs/lgps-opt/CONTCAR \
  --model-type mace \
  --model-file path/to/mace-checkpoint.pt \
  --ensemble NVT \
  --temp 600 \
  --timestep 1.0 \
  --steps 1000000 \
  --equilibration-steps 50000 \
  --thermostat BUSSI \
  --bussi-tau 1000 \
  --output runs/lgps-nvt-600K
```

!!! note "Thermostat choice"
    `BUSSI` (stochastic velocity rescaling) gives canonical sampling with
    minimal perturbation to the dynamics, making it a good default for
    transport calculations. `LANGEVIN` is also valid but introduces friction
    that slightly broadens VACF peaks.

The trajectory is written to `runs/lgps-nvt-600K/trajectory/`.

## Step 3 — Validate the trajectory

Always validate before analysis to catch truncated or corrupted trajectories.

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K validate
```

The report lists which analysis tasks can safely run. If `fixed_cell`,
`uniform_time_axis`, and `task_msd` all pass, you're ready.

## Step 4 — Thermodynamic diagnostics

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K thermo
```

Check:
- Mean temperature close to 600 K with reasonable variance (~10–20 K for
  a ~400-atom cell)
- No monotonic energy drift in the production window

## Step 5 — Structural stability (RMSD)

Confirm the LGPS framework didn't melt or transform during MD.

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K rmsd \
  --drift-reference nonmobile
```

The RMSD should plateau (< 1 Å for a stable crystal). If it grows without
bound, the temperature may be too high or the model poorly-transferable.

## Step 6 — Mean-square displacement

Extract the Li diffusion coefficient from the production trajectory.

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K msd \
  --mobile Li \
  --axes xyz \
  --drift-reference nonmobile \
  --drift-mode mass_weighted_com
```

The output `msd.csv` contains MSD(t) in Å². The diffusion coefficient *D* is
extracted from the slope in the linear (diffusive) regime; `summary.json`
reports the fitted value.

## Step 7 — Ionic conductivity (transport)

The `transport` task uses kinisi for a Bayesian fit to the MSD with
propagated uncertainty.

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K transport \
  --mobile Li \
  --frame-interval-fs 1.0 \
  --drift-reference nonmobile \
  --drift-mode mass_weighted_com \
  --temperature 600
```

`summary.json` reports *D* (m² s⁻¹), its 95 % credible interval, and
ionic conductivity σ (S cm⁻¹) via the Nernst–Einstein relation.

## Step 8 — Li–S radial distribution function

Verify the local coordination environment is physically reasonable.

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K rdf \
  --center Li \
  --neighbor S \
  --rmax 6.0 \
  --bins 200
```

The first peak near 2.5 Å corresponds to the Li–S nearest-neighbour distance.

## Step 9 — 3-D density map

Visualise the Li diffusion pathways.

```bash
./bin/mliport-mace analyze runs/lgps-nvt-600K density \
  --mobile Li \
  --spacing 0.2 \
  --smoothing-sigma-A 0.3 \
  --stride 5
```

The output `density.npz` contains occupancy probability and number density
grids. Load with NumPy:

```python
import numpy as np
data = np.load("runs/lgps-nvt-600K/analysis/density/<id>/density.npz")
prob = data["occupancy_probability"]
cell = data["cell_A"]
```

Post-process into a VESTA-compatible Cube file with a short script to
visualise the 1-D channels in LGPS.

## Summary

| Step | Command | Key output |
|---|---|---|
| Optimise | `mliport opt` | `CONTCAR` |
| MD | `mliport md --ensemble NVT --temp 600` | `trajectory/` |
| Validate | `mliport analyze ... validate` | gate for all analysis |
| Thermo | `mliport analyze ... thermo` | energy drift, T stability |
| RMSD | `mliport analyze ... rmsd` | structural stability |
| MSD | `mliport analyze ... msd --mobile Li` | *D*_Li |
| Transport | `mliport analyze ... transport --mobile Li` | *D* + σ with CI |
| RDF | `mliport analyze ... rdf --center Li --neighbor S` | Li–S g(r) |
| Density | `mliport analyze ... density --mobile Li` | 3-D occupancy |

## See also

- [MD workflow](../workflows/md.md)
- [MSD analysis](../analysis/msd.md)
- [Transport analysis](../analysis/transport.md)
- [Literature benchmark tutorial](literature-benchmark.md)
