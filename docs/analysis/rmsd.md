# RMSD and RMSF

The `rmsd` task computes the periodic-displacement root-mean-square deviation
(RMSD) and root-mean-square fluctuation (RMSF) — fast diagnostics for whether
a crystal kept its structure during MD.

```bash
mliport analyze runs/md rmsd
mliport analyze runs/md rmsd --species Li
mliport analyze runs/md rmsd --drift-reference nonmobile
```

## Physical background

### RMSD

The RMSD measures how far atoms have moved from their initial positions at
each frame, averaged over all selected atoms:

$$
\text{RMSD}(t) = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} |\Delta \mathbf{r}_i(t)|^2 }
$$

where Δ**r**ᵢ(t) is the minimum-image displacement of atom *i* from its
initial position to frame *t*. Unlike molecular RMSD, there is **no
rotational alignment** — the displacement is purely translational with
periodic boundary conditions handled via ASE's minimum-image convention.

A stable crystal shows a bounded RMSD (thermal vibrations). Unbounded growth
signals melting, a phase transition or structural collapse.

### RMSF

The RMSF measures the fluctuation of each atom around its own mean position:

$$
\text{RMSF}_i = \sqrt{ \frac{1}{T} \sum_{t=1}^{T} |\Delta \mathbf{r}_i(t) - \langle \Delta \mathbf{r}_i \rangle|^2 }
$$

High-RMSF atoms are the most mobile. Per-atom RMSF is useful for identifying
mobile sublattices or defect sites.

## Output files

| File | Contents |
|---|---|
| `rmsd.csv` | `time_ps`, `periodic_displacement_rmsd_A` |
| `rmsf.csv` | `atom_index`, `periodic_displacement_rmsf_A` |

## Options

| Option | Default | Description |
|---|---|---|
| `--species` | all | Restrict to one element (e.g. `Li`, `O`) |
| `--drift-reference` | `none` | Drift correction reference: `none`, `nonmobile`, `indices` |
| `--drift-indices` | — | Atom indices for drift correction (with `--drift-reference indices`) |
| `--include-equilibration` | off | Include equilibration frames |
| `--start-frame` | — | First frame index |
| `--stop-frame` | — | Last frame index (exclusive) |

### Drift correction

By default (`--drift-reference none`) no drift correction is applied. For
solid electrolytes, using `--drift-reference nonmobile` subtracts the drift of
the immobile sublattice, isolating the displacement of mobile ions from slow
framework drift.

## Tips

- A flat RMSD plateau confirms the crystal is stable; run `thermo` alongside
  to verify the temperature is also stable.
- Compare RMSF across species to identify which sublattice is mobile — this
  helps choose `--mobile` for MSD and transport tasks.
- For amorphous or liquid systems, RMSD grows without bound; use MSD instead
  for quantitative analysis of diffusion.
