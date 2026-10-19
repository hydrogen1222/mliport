# VACF and vibrational spectrum

`vacf` computes the velocity autocorrelation function and `spectrum` its Fourier
transform, a vibrational spectral density. Use them as diagnostics of the dynamics.

```bash
mliport analyze runs/md vacf --species Li
mliport analyze runs/md spectrum --species Li --taper one-sided-cosine \
  --normalization normalized_area
```

Both require stored velocities and a uniform time axis; velocities are not
reconstructed from positions. Both run headlessly and write CSV plus PNG and SVG.
Spectrum area normalization supports both NumPy 1.x and 2.x.

## Options you will use

| Option | Meaning |
|---|---|
| `--species SPECIES` | the species to correlate (all atoms when omitted) |
| `--method fft\|direct` | the correlation algorithm (default `fft`) |
| `--taper one-sided-cosine\|none` | window applied before the transform (`spectrum`; default `one-sided-cosine`) |
| `--normalization normalized_area\|raw_spectrum` | spectrum normalization (`spectrum`; default `normalized_area`) |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | the analysis window |
| `--positions-convention`, `--frame-interval-fs` | metadata for external trajectories |

## What to watch

The **saved-frame interval** sets the Nyquist frequency, not the integration
timestep. If it is too coarse for the spectral range you asked for, mliport
reports insufficient sampling instead of silently aliasing.

Spectra depend on the frame spacing, the taper and the trajectory length, so
report all three. A spectrum from a classical MLIP trajectory is not a
quantitative phonon calculation and carries no quantum nuclear effects. Compare
spectra only inside one model and timestep protocol. A short trajectory gives a
noisy but finite spectrum; do not read its frequencies as converged vibrational
modes.

```bash
mliport analyze runs/md-300K spectrum --species O --method fft \
  --taper one-sided-cosine --normalization normalized_area
```

The acceptance suite runs the plain forms on the 1 ns GRACE trajectory:

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns vacf
.venv-grace/bin/mliport analyze results/lgps-1ns spectrum
```
