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

## Physical background

### The Arrhenius equation

The Arrhenius equation describes how a thermally activated process speeds up
with temperature:

$$
D(T) = D_0 \, \exp\!\left(- \frac{E_a}{k_B T}\right)
$$

where *D*₀ is the pre-exponential factor (units of m²/s), *E*_a is the
**activation energy** (the energy barrier the mobile ion must overcome to
hop), *k*_B is Boltzmann's constant, and *T* is the absolute temperature.

Taking the natural logarithm of both sides:

$$
\ln D = \ln D_0 - \frac{E_a}{k_B} \cdot \frac{1}{T}
$$

This is a straight line in *1/T* vs. ln *D* space. The slope gives *E*_a and
the intercept gives *D*₀.

### What the activation energy means

*E*_a is the effective barrier height for ionic hopping. In a solid electrolyte:

- **Low *E*_a** (0.1–0.3 eV) → fast-ion conduction, as in the best
  superionic conductors (LGPS, Li₇La₃Zr₂O₁₂).
- **High *E*_a** (> 0.5 eV) → slow diffusion, typical of "normal" ionic
  crystals at moderate temperature.
- The apparent *E*_a from simulation depends on the temperature range, the
  model, the cell size, and the ensemble. It is a property of the simulation,
  not a material constant, until it is validated against experiment.

### When Arrhenius fails

The Arrhenius equation assumes a single dominant mechanism across the entire
temperature range. This can break down when:

- **A phase transition** occurs within the range — the slope changes
  abruptly.
- **Multiple mechanisms compete** — the effective activation energy is a
  weighted average that changes with temperature.
- **The system is not in equilibrium** — the trajectory at one temperature
  was not long enough for the diffusion coefficient to converge.

Non-Arrhenius behavior shows up as curvature in the 1/*T* vs. ln *D* plot.
Always check the linearity and the r² of the fit.

## Options you will use

| Option | Meaning |
|---|---|
| `--temperature T` | a temperature in K (repeat; required) |
| `--diffusivity D` | the matching diffusivity in m²/s (repeat; required) |
| `--diffusivity-std S` | an optional uncertainty for each point, used for weighting (repeat) |
| `--extrapolate-temperature T` | a temperature to extrapolate to, recorded separately (repeat) |
| `--source-run-id ID` | links back to the runs the points came from (repeat) |

### Uncertainty weighting

When `--diffusivity-std` is provided, the fit uses **weighted least squares**
in log-space. The weight for each point is 1 / σ_ln(D), where σ_ln(D) = σ_D / D
(error propagation through the logarithm). This gives more influence to
better-determined points, which is usually the ones at higher temperatures
where the MSD converges faster.

Without `--diffusivity-std`, the fit is an unweighted OLS, which treats all
points equally. For two points, the line is exactly determined and no
uncertainty can be estimated.

## What comes out

`results.json` has the fitted activation energy `Ea`, the prefactor, the fit
quality, the inputs and any extrapolations, with plots in
`<run>/analysis/arrhenius/<analysis-id>/`.

| Field | Contents |
|---|---|
| `activation_energy_eV` | *E*_a in electron-volts |
| `activation_energy_std_eV` | Uncertainty on *E*_a (null for two points) |
| `preexponential_factor_m2_s` | *D*₀ in m²/s |
| `r_squared` | Coefficient of determination |
| `extrapolations` | Each extrapolated temperature with its predicted *D* and whether it is outside the fitted range |
| `warnings` | Two-point fit caveats, extrapolation warnings |

Mismatched lengths, non-positive values, fewer than two temperatures or a
degenerate temperature range fail closed.

## Interpreting results

### Fit quality

- **r² close to 1** — the data is well described by a single Arrhenius line.
  With only three points, r² > 0.95 is expected even with real scatter.
- **r² < 0.9** — either the temperature range spans a mechanism change, one
  point is an outlier (unconverged trajectory), or the data is simply noisy.
- **Two points** — the fit is mathematically exact (r² = 1 by construction)
  and provides no statistical test. Use three or more temperatures.

### Reporting

- Every diffusivity must come from the same model identity and MD protocol
  (timestep, cell, ensemble, drift policy), otherwise the fit mixes different
  physics.
- Extrapolating beyond the measured range is a prediction and is recorded as
  one. The further the extrapolation, the less reliable.
- Report the temperature range, the number of points, the fit quality, and
  the activation energy with its uncertainty.

### Feeding it from `transport`

The natural inputs are the diffusivities `transport` produced at each
temperature. The acceptance run did exactly that with three same-engine points:

```bash
.venv-grace/bin/mliport analyze results/uma-800K.traj arrhenius \
  --frame-interval-fs 100 --positions-convention unwrapped \
  --temperature 600 --temperature 700 --temperature 800 \
  --diffusivity 8.475718e-10 --diffusivity 3.479545e-09 --diffusivity 4.103223e-09 \
  --diffusivity-std 7.356329e-11 --diffusivity-std 2.407381e-10 --diffusivity-std 2.640917e-10
```

The source trajectory is only an anchor; trajectory-source options such as
`--positions-convention` are recorded as ignored for this fit. The run gave
`Ea = 0.316 eV` (r² = 0.88).

### Example with extrapolation

```bash
mliport analyze runs/md arrhenius \
  --temperature 600 --temperature 700 --temperature 800 --temperature 900 \
  --diffusivity 1.2e-12 --diffusivity 3.5e-12 \
  --diffusivity 8.1e-12 --diffusivity 1.6e-11 \
  --diffusivity-std 0.1e-12 --diffusivity-std 0.2e-12 \
  --diffusivity-std 0.4e-12 --diffusivity-std 0.8e-12 \
  --extrapolate-temperature 300 --extrapolate-temperature 1200
```

If fewer than two transport runs succeed, the pipeline reports
`insufficient_transport_data` instead of an activation energy.

## Next steps

- To compute the diffusivities that feed this fit: [Transport](transport.md)
- To understand the diffusion mechanism at each temperature: [GEMDAT](electrolyte-gemdat.md)
- To check whether the trajectory is in the diffusive regime: [MSD](msd.md)
