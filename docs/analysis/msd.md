# MSD and the local exponent

`msd` computes the mean-squared displacement of the mobile species and the local
exponent `alpha(t) = d log(MSD) / d log(t)`. Treat it as a diagnostic of which
diffusion regime the trajectory is in, not as a diffusivity.

```bash
mliport analyze runs/md msd --mobile Li --axes xyz
```

`--mobile` is required. If you analyze an external trajectory, say whether its
coordinates are wrapped or unwrapped (`--positions-convention`) and give the saved
frame spacing (`--frame-interval-fs`) when the metadata lacks it; with missing
information mliport fails rather than assume.

## Options you will use

| Option | Meaning |
|---|---|
| `--mobile SPECIES` | the mobile species (required) |
| `--axes` | `x`, `y`, `z` or a combination such as `xy` or `xyz` (default `xyz`) |
| `--drift-reference`, `--drift-mode`, `--drift-indices` | how framework drift is removed: reference `none`, `nonmobile`, `all` or `indices`; mode `none`, `arithmetic_mean` or `mass_weighted_com` (default) |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | the analysis window |
| `--method fft\|direct` | the MSD algorithm (default `fft`) |
| `--fit-start-ps`, `--fit-stop-ps` | a fit window, recorded with the result |
| `--alpha-window-decades` | full width of the local log–log regression window (default 0.25) |
| `--alpha-min-points`, `--alpha-min-origins` | minimum points in a window (5) and time origins for an alpha point to count (8) |
| `--alpha-consistency-band` | half-band around `alpha = 1` accepted as diffusive (default 0.2) |
| `--alpha-max-lag-ps` | ignore longer lags for alpha only; the raw MSD is untouched |
| `--alpha-focus-band LOW HIGH`, `--alpha-log-x` | plot options: zoom the alpha figure, use a log time axis |

## Reading alpha

`alpha ≈ 1` is consistent with Fickian diffusion, `alpha < 1` with subdiffusion
and `alpha > 1` with superdiffusive or ballistic motion, but only inside the time
window you fitted. A point outside the consistency band is marked
`not_diffusive`; one with too few time origins is reported as insufficient
support, not as an exponent. Read the whole alpha curve with its band, never a
single number.

Other things that matter:

- Drift removal changes the long-time MSD, so state the drift policy.
- MSD is a statistical estimator. Report several fit windows or an uncertainty,
  not one fit.
- Alpha is not a conductivity verdict. Conductivity needs the charge, collective
  correlations and a Nernst–Einstein treatment; use [Transport](transport.md).

```bash
mliport analyze runs/md-800K msd --mobile Li --axes xyz \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --fit-start-ps 50 --fit-stop-ps 400 --alpha-focus-band 0.8 1.2
```

## Short trajectories

A 5-step smoke trajectory is accepted. The result carries an explicit
"insufficient diffusive window" status and the alpha and diagnostic fields stay
finite. The acceptance suite treats that as an expected limitation, never as a
diffusion measurement. For a real MSD give the run at least a few hundred saved
frames and let the fit window cover a diffusive regime.

The results are written to `results.json` and the MSD and alpha tables and plots
under `<run>/analysis/msd/<analysis-id>/`, including the exact fit and alpha
windows that were used.
