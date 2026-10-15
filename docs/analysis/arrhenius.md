# Arrhenius

`arrhenius` fits an Arrhenius relation to diffusivities you obtained from separate
runs at several temperatures, optionally extrapolates, and reports the implied
activation energy. It fits the points you give it. It never invents a missing one.

```bash
mliport analyze runs/md-800K arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11
```

Repeat the flag for each value: temperatures and diffusivities are paired by
position and must have the same length. The run or trajectory you name is only the
analysis source, which decides where the output goes; the numbers come from the
options.

## Options you will use

| Option | Meaning |
|---|---|
| `--temperature T` | a temperature in K (repeat; required) |
| `--diffusivity D` | the matching diffusivity in m²/s (repeat; required) |
| `--diffusivity-std S` | an optional uncertainty for each point, used for weighting (repeat) |
| `--extrapolate-temperature T` | a temperature to extrapolate to, recorded separately (repeat) |
| `--source-run-id ID` | links back to the runs the points came from (repeat) |

## What comes out

`results.json` has the fitted activation energy `Ea`, the prefactor, the fit
quality, the inputs and any extrapolations, with plots in
`<run>/analysis/arrhenius/<analysis-id>/`.

Mismatched lengths, non-positive values, fewer than two temperatures or a
degenerate temperature range fail closed. A fit with a negative slope is reported
as a poor or invalid fit, never as a negative activation energy.

## Reading it

- Every diffusivity must come from the same model identity and MD protocol
  (timestep, cell, ensemble, drift policy), otherwise the fit mixes different
  physics.
- Extrapolating beyond the measured range is a prediction and is recorded as one.
- Arrhenius behavior is an assumption. Check the linearity and the fit quality.

```bash
mliport analyze runs/md series/arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11 \
  --diffusivity-std 0.1e-12 --diffusivity-std 0.2e-12 \
  --diffusivity-std 0.4e-12 --diffusivity-std 0.8e-12 \
  --extrapolate-temperature 300 --extrapolate-temperature 1200
```

## Feeding it from `transport`

The natural inputs are the diffusivities `transport` produced at each temperature.
The acceptance run did exactly that with three same-engine UMA LGPS points:

```bash
.venv-grace/bin/mliport analyze results/uma-800K.traj arrhenius \
  --frame-interval-fs 100 --positions-convention unwrapped \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 8.475718e-10 --diffusivity 3.479545e-09 --diffusivity 4.103223e-09 \
  --diffusivity-std 7.356329e-11 --diffusivity-std 2.407381e-10 --diffusivity-std 2.640917e-10
```

The source trajectory is only an anchor; trajectory-source options such as
`--positions-convention` are recorded as ignored for this fit. The run gave
`Ea = 0.316 eV` (r² = 0.88). If fewer than two transport runs succeed, the
pipeline reports `insufficient_transport_data` instead of an activation energy.
