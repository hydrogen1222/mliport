# Arrhenius

**Purpose.** Fit an Arrhenius relation to diffusivities obtained from
separate runs at several temperatures, with optional extrapolation, and
report the implied activation energy.

**Minimal command.**

```bash
mliport analyze runs/md-800K arrhenius \
  --temperature 600 700 800 900 \
  --diffusivity 1.2e-12 3.5e-12 8.1e-12 1.6e-11 \
  --source-run-id md-600K md-700K md-800K md-900K
```

**Important options.**

| Option | Meaning |
|---|---|
| `--temperature T [T ...]` | temperatures in K (required) |
| `--diffusivity D [D ...]` | diffusivities in m²/s (required; same length as temperatures) |
| `--diffusivity-std S [S ...]` | optional uncertainties for weighting |
| `--extrapolate-temperature T [T ...]` | temperatures to extrapolate to (recorded separately) |
| `--source-run-id ID [ID ...]` | provenance links back to the analysed runs |

**Inputs.** Diffusivities measured elsewhere with their uncertainties. The
positional run/trajectory argument is used for provenance and output
location; the numbers themselves come from the explicit options.

**Outputs.** `analysis.json` with the fitted `Ea`, prefactor, fit quality,
input values and extrapolations; plots under `<run>/analysis/<id>/`.

**Failure semantics.** Mismatched array lengths, non-positive values, fewer
than two temperatures or a degenerate temperature range fail closed. A
negatively-sloped Arrhenius fit is reported as a poor/invalid fit rather than
as a negative activation energy.

**Scientific caveats.**

- Every input diffusivity must come from the same model identity and MD
  protocol (timestep, cell, ensemble, drift policy); otherwise the fit mixes
  different physics.
- Extrapolation outside the measured range is a prediction, not a
  measurement; it is recorded as extrapolated.
- Arrhenius behaviour is an assumption: check linearity and the fit quality.

**Example.**

```bash
mliport analyze runs/md series/arrhenius \
  --temperature 600 700 800 900 \
  --diffusivity 1.2e-12 3.5e-12 8.1e-12 1.6e-11 \
  --diffusivity-std 0.1e-12 0.2e-12 0.4e-12 0.8e-12 \
  --extrapolate-temperature 300 1200
```

## Independent temperature points

`arrhenius` fits explicit, independent `(T, D)` pairs. Pass the values
produced by `transport` at each temperature; the command never invents a
missing point:

```bash
# tested example
.venv-grace/bin/mliport analyze results/uma-800K.traj arrhenius \
  --frame-interval-fs 100 --positions-convention unwrapped \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 8.475718e-10 --diffusivity 3.479545e-09 --diffusivity 4.103223e-09 \
  --diffusivity-std 7.356329e-11 --diffusivity-std 2.407381e-10 --diffusivity-std 2.640917e-10
```

The source trajectory is only used as the analysis source; trajectory-source
options such as `--positions-convention` are recorded as ignored for this fit.
With three same-engine UMA LGPS points the acceptance run fits
`Ea = 0.316 eV` (r2 = 0.88). If fewer than two transports succeed, the
pipeline reports `insufficient_transport_data` instead of an activation
energy.
