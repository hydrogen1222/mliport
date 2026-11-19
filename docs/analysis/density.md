# Density map

The `density` task computes a periodic 3-D occupancy probability and number
density grid for a selected mobile species. This reveals preferred diffusion
pathways and site occupancies in solid electrolytes.

```bash
mliport analyze runs/md density --mobile Li --spacing 0.2 --stride 5
```

## Output

The output is a NumPy `.npz` archive (`density.npz`) containing:

| Array | Shape | Description |
|---|---|---|
| `occupancy_probability` | grid | Fraction of time each voxel is occupied (sums to 1) |
| `number_density_A3` | grid | Number density in Å⁻³ per voxel |
| `counts_per_voxel` | grid | Raw histogram counts |
| `cell_A` | (3, 3) | Simulation cell in Ångströms |

The grid shape is determined by the cell dimensions and the requested spacing.

### Loading the output

```python
import numpy as np

data = np.load("density.npz")
probability = data["occupancy_probability"]
density = data["number_density_A3"]
cell = data["cell_A"]
```

## Options

| Option | Default | Description |
|---|---|---|
| `--mobile SPECIES` | — | Element to map (e.g. `Li`). **Required.** |
| `--spacing` | 0.25 | Voxel edge length in Å. Smaller values give finer grids. |
| `--smoothing-sigma-A` | — | Gaussian smoothing width in Å. Applied periodically. |
| `--stride` | 1 | Use every *n*-th frame (reduces memory for long trajectories) |
| `--include-equilibration` | off | Include equilibration frames |
| `--start-frame` | — | First frame index |
| `--stop-frame` | — | Last frame index (exclusive) |

## Requirements

- 3-D periodic boundary conditions
- Fixed cell (variable-cell trajectories are rejected)

These are the same as the RDF gate, plus the fixed-cell constraint.

## Physical details

Atomic positions are folded into fractional coordinates and histogrammed onto
a 3-D grid that tiles the unit cell. The grid dimensions are
⌈*L* / spacing⌉ along each lattice vector, with a minimum of 2 bins per
direction.

**Occupancy probability** is the histogram normalised so it sums to 1 — the
probability that a randomly chosen snapshot places a mobile atom in a given
voxel.

**Number density** is the count per frame divided by the voxel volume,
yielding a spatial density in Å⁻³.

When `--smoothing-sigma-A` is set, an isotropic Gaussian is applied in
reciprocal space with periodic wrapping, so the density is smooth across cell
boundaries. This is useful for visualisation but widens sharp features.

## Tips

- Start with the default spacing (0.25 Å) and refine if needed. Very fine
  grids (< 0.1 Å) produce large files and may need more frames to converge.
- Use `--stride` to reduce memory for trajectories with many frames.
- The `.npz` file can be post-processed into volumetric formats (Cube,
  CHGCAR, XDATCAR-compatible grids) with a short Python script for
  visualisation in VESTA or OVITO.
