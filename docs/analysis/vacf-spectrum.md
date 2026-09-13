# VACF and vibrational spectrum

**Purpose.** Compute the velocity autocorrelation function (VACF) and its
Fourier transform (vibrational/spectral density) as diagnostics of the
dynamics and phonon-like spectrum.

**Minimal command.**

```bash
mliport analyze runs/md vacf --species Li --method fft
mliport analyze runs/md spectrum --species Li --method fft \
  --taper one-sided-cosine --normalization normalized_area
```

**Important options.**

| Option | Meaning |
|---|---|
| `--species SPECIES` | species to correlate (all atoms when omitted) |
| `--method {fft,direct}` | correlation algorithm |
| `--taper {one-sided-cosine,none}` | window applied before the transform (spectrum) |
| `--normalization {normalized_area,raw_spectrum}` | spectrum normalisation |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | analysis window |
| `--positions-convention`, `--frame-interval-fs` | external trajectory metadata |

**Inputs / outputs.** A trajectory (velocities are derived consistently with
the saved frames); `analysis.json` plus VACF/spectrum arrays and plots. The
**saved-frame interval** sets the Nyquist frequency — it is not the
integration timestep.

**Failure semantics.** A saved-frame interval that is too coarse for the
requested spectral range is reported as insufficient sampling rather than
aliased silently.

**Scientific caveats.** Spectra are sensitive to the frame spacing, taper
and trajectory length; report all three. A spectrum from a classical MLIP
trajectory is not a quantitative phonon calculation and does not include
quantum nuclear effects. Compare spectra only within one model/timestep
protocol.

**Example.**

```bash
mliport analyze runs/md-300K spectrum --species O --method fft \
  --taper one-sided-cosine --normalization normalized_area --stride 1
```
