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

## Physical background

### Velocity autocorrelation function (VACF)

The VACF measures how long an atom "remembers" its velocity. At *t* = 0 the
VACF equals the mean kinetic energy per atom (in velocity-squared units); it
decays as collisions randomize the direction of motion:

$$
C_v(t) = \left\langle \mathbf{v}_i(t_0) \cdot \mathbf{v}_i(t_0 + t) \right\rangle_{i, t_0}
$$

The normalized VACF *C*_v(*t*) / *C*_v(0) starts at 1 and decays toward 0.
How it decays tells you about the dynamics:

- **Exponential decay** — diffusive, liquid-like collisions.
- **Oscillatory with negative dips** — atoms are partially trapped in local
  cages and bounce back (characteristic of solid electrolytes and dense
  liquids).
- **Slow oscillatory decay** — vibrational modes dominate; the system behaves
  more like a crystal.

### From VACF to vibrational spectrum

The Fourier cosine transform of the VACF yields the **velocity power spectrum**
(or vibrational density of states in the classical limit):

$$
S(\nu) = \int_0^{\infty} C_v(t) \, \cos(2 \pi \nu t) \, dt
$$

In practice, mliport uses a real-valued FFT (rFFT) of the one-sided, windowed,
normalized VACF. The result is a spectral density whose peaks correspond to the
vibrational frequencies of the system.

!!! warning "Not a phonon DOS"
    The VACF-derived spectrum from a classical MD trajectory is **not** a
    quantitative phonon density of states. It lacks quantum nuclear effects
    (zero-point motion, Bose–Einstein statistics), anharmonic corrections, and
    the proper weighting by neutron cross-sections or IR activity. It is a
    classical dynamical diagnostic — useful for comparing models, checking
    for artifacts, and identifying characteristic frequencies, but not for
    direct comparison to inelastic neutron scattering or IR/Raman experiments.

### Nyquist frequency

The maximum frequency the spectrum can resolve is the Nyquist frequency, set
by the **saved-frame interval** (not the integration timestep):

$$
\nu_\text{Nyquist} = \frac{1}{2 \, \Delta t_\text{save}}
$$

If the save interval is 10 fs, the Nyquist frequency is 50 THz ≈ 1668 cm⁻¹.
Frequencies above this are aliased. To resolve high-frequency O–H stretches
(~100 THz), you need frames saved every 5 fs or less. mliport reports both the
Nyquist frequency in THz and cm⁻¹ in every VACF result.

### Windowing (tapering)

The VACF is finite in length. A sharp truncation at the end of the trajectory
introduces **spectral leakage** — artificial oscillations in the spectrum from
the discontinuity at the boundary. A window function (taper) smooths the
truncation. mliport's default is a **one-sided cosine taper** (raised cosine
from 1 at lag 0 to 0 at the last lag), which suppresses leakage with minimal
distortion of the peak positions.

The `none` taper (rectangular window) is available for comparison; its peaks
are sharper but the leakage is stronger.

### Thermostat artifacts

Thermostats modify the dynamics. A Langevin thermostat introduces friction
that damps the VACF and broadens spectral peaks. A Nosé–Hoover chain
thermostat preserves the dynamics better but can introduce low-frequency
oscillations if its coupling frequency is close to a physical mode. An NVE
trajectory gives the cleanest VACF, but drift may limit the useful length.

When comparing spectra, always use the **same thermostat and timestep
protocol**. A peak that appears in Langevin but not NHC (or vice versa) is
more likely an artifact than a physical mode.

## Options you will use

| Option | Meaning |
|---|---|
| `--species SPECIES` | the species to correlate (all atoms when omitted) |
| `--method fft\|direct` | the correlation algorithm (default `fft`) |
| `--taper one-sided-cosine\|none` | window applied before the transform (`spectrum`; default `one-sided-cosine`) |
| `--normalization normalized_area\|raw_spectrum` | spectrum normalization (`spectrum`; default `normalized_area`) |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | the analysis window |
| `--positions-convention`, `--frame-interval-fs` | metadata for external trajectories |

### VACF method

- **`fft`** (default) — O(*N* log *N*) via FFT autocorrelation. Use this.
- **`direct`** — O(*N*²) reference implementation. Useful for verifying the FFT
  result on short trajectories; too slow for production.

Both methods compute the full time-origin-averaged VACF. They give identical
results (to floating-point precision).

### Spectrum normalization

- **`normalized_area`** (default) — the signed integral of the spectrum over
  frequency equals 1. This makes spectra from different trajectory lengths
  comparable in shape.
- **`raw_spectrum`** — the rFFT real part multiplied by the frame interval.
  Useful when you need the absolute scale.

## Output files

| File | Contents |
|---|---|
| `results.json` | VACF and spectrum arrays, Nyquist frequency, negative fraction, provenance |
| `vacf.csv` | Lag time (fs) vs. raw and normalized VACF |
| `spectrum.csv` | Frequency (THz, cm⁻¹) vs. spectral density |
| `vacf.png`, `vacf.svg` | VACF plot |
| `spectrum.png`, `spectrum.svg` | Spectrum plot |

## Interpreting results

### What to report

1. **The saved-frame interval** — it sets the Nyquist limit and the frequency
   resolution. A 10 fs interval gives 50 THz Nyquist and ~1/(N×Δt) resolution.
2. **The taper** — `one-sided-cosine` or `none`.
3. **The trajectory length** — longer trajectories give narrower, better-
   resolved peaks.
4. **The thermostat** — it affects the peak shapes and widths.
5. **The species** — per-species spectra separate framework vibrations from
   mobile-ion dynamics.

### Negative spectral values

The spectrum retains negative estimates. A small negative fraction
(`negative_fraction` in `results.json`) is normal noise; a large negative
fraction suggests the VACF was not well converged or the taper was
inappropriate.

### Comparing spectra

Compare spectra only within the **same model, timestep, thermostat, and
save interval**. A shift in peak position between two models at the same
conditions is meaningful; a shift between different save intervals is an
artifact.

```bash
mliport analyze runs/md-300K spectrum --species O --method fft \
  --taper one-sided-cosine --normalization normalized_area
```

The acceptance suite runs the plain forms on the 1 ns GRACE trajectory:

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns vacf
.venv-grace/bin/mliport analyze results/lgps-1ns spectrum
```
