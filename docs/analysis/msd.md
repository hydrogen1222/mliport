# MSD and local exponent

**Purpose.** Compute the mean-squared displacement of the mobile species and
the local MSD exponent `alpha(t) = d log(MSD) / d log(t)` as a **diagnostic**
of the diffusion regime.

**Minimal command.**

```bash
mliport analyze runs/md msd --mobile Li --axes xyz --force
```

**Important options.**

| Option | Meaning |
|---|---|
| `--mobile SPECIES` | mobile species/selection (required) |
| `--axes` | `x`, `y`, `z`, or combinations (`xy`, `xyz`, ...) |
| `--positions-convention` | `wrapped`/`unwrapped`/`unknown` for external trajectories |
| `--frame-interval-fs` | saved-frame interval when metadata lacks it |
| `--drift-reference`, `--drift-mode`, `--drift-indices` | how framework drift is removed (`none`, `nonmobile`, `all`, `indices`) |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | analysis window |
| `--alpha-window-decades`, `--alpha-min-points`, `--alpha-min-origins` | local-exponent windowing/support |
| `--alpha-consistency-band`, `--alpha-max-lag-ps`, `--alpha-focus-band`, `--alpha-log-x` | alpha diagnostics/plots |
| `--method fft|direct` | MSD algorithm |
| `--fit-start-ps`, `--fit-stop-ps` | fit window recorded with the result |

**Inputs / outputs.** One trajectory and the mobile selection in;
`analysis.json` plus MSD/alpha tables and plots under
`<run>/analysis/<analysis_id>/`, including the exact fit and alpha windows
used.

**Failure semantics.** Missing or wrapped-coordinate data without a stated
convention fails closed; too few time origins for an alpha point is reported
as insufficient support rather than as an exponent.

**Scientific caveats.**

- `alpha ≈ 1` is consistent with Fickian diffusion, `alpha < 1` with
  subdiffusion, `alpha > 1` with superdiffusive/ballistic motion — but only
  inside a fitted time window. Read the alpha curve with its consistency
  band, never a single number.
- Drift removal changes the long-time MSD; state the drift policy.
- MSD is a statistical estimator: report multiple fit windows/uncertainty,
  not one fit.
- **Alpha is not a conductivity verdict.** Transport needs charge, collective
  correlations and the Nernst-Einstein treatment; use
  [transport](transport.md).

**Example.**

```bash
mliport analyze runs/md-800K msd --mobile Li --axes xyz \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --fit-start-ps 50 --fit-stop-ps 400 --alpha-focus-band 0.8 1.2
```
