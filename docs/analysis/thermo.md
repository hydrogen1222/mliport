# Thermodynamic diagnostics

The `thermo` task extracts temperature, energy, pressure and volume time series
from a trajectory and computes summary statistics and block averages. For NVE
simulations it also reports the total-energy drift via linear fit.

```bash
mliport analyze runs/md thermo
mliport analyze runs/md thermo --block-size-frames 500
```

## Output files

| File | Contents |
|---|---|
| `thermo.csv` | Time series of all available columns |
| `thermo_*.png` | Per-column plots (temperature, energies, pressure, volume) |

### CSV columns

All columns are per-atom where noted:

| Column | Unit | Notes |
|---|---|---|
| `time_ps` | ps | Time relative to production start |
| `absolute_time_ps` | ps | Absolute simulation time |
| `step` | — | MD step number (if available) |
| `phase` | — | `equilibration` or `production` |
| `temperature_K` | K | Instantaneous temperature |
| `potential_energy_eV_atom` | eV/atom | Potential energy per atom |
| `kinetic_energy_eV_atom` | eV/atom | Kinetic energy per atom |
| `total_energy_eV_atom` | eV/atom | Total energy per atom |
| `pressure_GPa` | GPa | Instantaneous pressure |
| `volume_A3` | Å³ | Cell volume |

Columns that are unavailable in the trajectory are simply omitted.

## Options

| Option | Default | Description |
|---|---|---|
| `--include-equilibration` | off | Include equilibration frames (production-only by default) |
| `--start-frame` | — | First frame index |
| `--stop-frame` | — | Last frame index (exclusive) |
| `--block-size-frames` | — | Non-overlapping block size for block-averaged statistics |

## NVE energy drift

When the source simulation used the NVE ensemble, `thermo` fits a linear
trend to the total energy per atom versus time. The result includes:

| Key | Unit | Description |
|---|---|---|
| `nve_total_energy_drift_eV_atom_ps` | eV/atom/ps | Slope of linear fit |
| `nve_total_energy_drift_meV_atom_ps` | meV/atom/ps | Same, in meV |
| `nve_fitted_total_drift_eV_atom` | eV/atom | End-to-end drift of the fitted line |
| `nve_relative_drift_over_production` | — | Drift relative to initial energy |

## Data integrity

Statistics are computed from finite values only — NaN and ±Inf are excluded.
To make this transparent, the summary includes a `data_integrity` section
that reports, per column:

- Total frames and finite frames
- Non-finite fraction
- Which columns contain non-finite values

If part of a trajectory's `raw/md.csv` has empty or unparsable cells, they
become NaN. The mean will still look plausible, but `data_integrity` will
reveal the missing data.

## Block statistics

With `--block-size-frames N`, each thermodynamic column is divided into
non-overlapping blocks of *N* frames. The block means and their standard
deviation indicate equilibration quality: large standard deviations or
systematic trends in block means suggest the system has not yet equilibrated.

Non-finite values within a block are excluded; blocks with no finite frames
contribute no mean. The output records how many blocks were usable and how
many frames were skipped.

## Tips

- Check `thermo` output before running any transport analysis — drifting
  temperature or pressure can invalidate diffusion coefficients.
- For NVE, the energy drift should be small compared to kinetic energy
  fluctuations. A drift above ~1 meV/atom/ps typically indicates
  integration issues.
- Use `--block-size-frames` to gauge equilibration. If block means show a
  systematic trend, consider extending the equilibration period.
