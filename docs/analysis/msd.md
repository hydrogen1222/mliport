# Mean-squared displacement (MSD)

The `msd` task computes the mean-squared displacement of mobile atoms and the
local exponent α(t), which tells you what diffusion regime the trajectory is in.

```bash
./bin/mliport-mace analyze runs/lgps-md msd --mobile Li --axes xyz
```

## Physical background

### What is MSD?

The **mean-squared displacement** measures how far atoms have traveled from
their initial positions, averaged over all atoms of the selected species and
over all possible time origins:

$$
\text{MSD}(t) = \left\langle \frac{1}{N} \sum_{i=1}^{N} \left| \mathbf{r}_i(t_0 + t) - \mathbf{r}_i(t_0) \right|^2 \right\rangle_{t_0}
$$

where *N* is the number of mobile atoms, **r**ᵢ is the unwrapped position of
atom *i*, and the angle brackets denote averaging over time origins *t*₀.

In a system where atoms diffuse freely (Fickian diffusion), the MSD grows
linearly with time:

$$
\text{MSD}(t) = 2 d \, D \, t
$$

where *d* is the number of spatial dimensions (3 for `--axes xyz`, 2 for
`--axes xy`, etc.) and *D* is the **self-diffusion coefficient**. This is the
**Einstein relation** — the foundation for extracting diffusivity from MD
simulations.

!!! important "Unwrapped coordinates are essential"
    MSD requires **unwrapped** coordinates — positions that track the true
    displacement of each atom without folding it back into the simulation cell.
    mliport run directories always store unwrapped coordinates. For external
    trajectories, specify `--positions-convention unwrapped`; if the convention
    is wrong, the MSD will be artificially bounded by the cell size.

### The local exponent α(t)

The local exponent characterizes the instantaneous scaling of MSD with time:

$$
\alpha(t) = \frac{d \ln \text{MSD}}{d \ln t}
$$

Its value tells you the diffusion regime:

| α value | Regime | Physical meaning |
|---|---|---|
| α ≈ 2 | Ballistic | Very short times — atoms haven't collided yet |
| α > 1 | Superdiffusive | Correlated motion, directed flow |
| **α ≈ 1** | **Diffusive (Fickian)** | **Normal diffusion — Einstein relation holds** |
| α < 1 | Subdiffusive | Caging, trapping, confined motion |
| α ≈ 0 | Localized | Atoms vibrate in place, no net transport |

A typical solid-electrolyte trajectory shows α ≈ 2 at the very shortest
times (ballistic vibration), dropping below 1 during the "cage-rattling"
regime (atoms trapped in their coordination shells), and eventually
approaching 1 once atoms escape and diffuse. The time at which α reaches 1
is the onset of Fickian diffusion — only beyond this point can you
reliably extract a diffusion coefficient.

!!! warning "MSD alone is not a diffusivity"
    The `msd` task is a **diagnostic**. It tells you whether the trajectory is
    long enough to be in the diffusive regime. To extract a diffusion
    coefficient with proper uncertainty, use the [transport](transport.md) task,
    which applies kinisi's blocked averaging.

## Usage

### Required options

| Option | Description |
|---|---|
| `--mobile SPECIES` | The diffusing species (e.g., `Li`, `Na`, `O`). **Required.** |

### Common options

| Option | Default | Description |
|---|---|---|
| `--axes` | `xyz` | Spatial dimensions: `x`, `y`, `z`, or combinations like `xy`, `xyz` |
| `--drift-reference` | `none` | Framework for drift removal: `none`, `nonmobile`, `all`, `indices` |
| `--drift-mode` | `mass_weighted_com` | Drift calculation: `none`, `arithmetic_mean`, `mass_weighted_com` |
| `--method` | `fft` | MSD algorithm: `fft` (fast Fourier, O(N log N)) or `direct` (O(N²)) |

### Fit window

| Option | Description |
|---|---|
| `--fit-start-ps` | Start of the linear fit window (picoseconds) |
| `--fit-stop-ps` | End of the linear fit window |

The fit window is recorded with the result but does not affect the raw MSD —
it marks the region you consider diffusive for reference.

### Alpha control

| Option | Default | Description |
|---|---|---|
| `--alpha-window-decades` | 0.25 | Full width of the local log–log regression window |
| `--alpha-min-points` | 5 | Minimum data points in a window for α to be computed |
| `--alpha-min-origins` | 8 | Minimum time origins needed |
| `--alpha-consistency-band` | 0.2 | Half-band around α = 1 accepted as "diffusive" |
| `--alpha-max-lag-ps` | — | Ignore longer lags for α (raw MSD untouched) |
| `--alpha-focus-band LOW HIGH` | — | Zoom the α plot to this range |
| `--alpha-log-x` | false | Use a logarithmic time axis for the α plot |

### Analysis window

| Option | Default | Description |
|---|---|---|
| `--start-frame` | — | First frame to include |
| `--stop-frame` | — | Last frame to include |
| `--include-equilibration` | false | Include equilibration frames (excluded by default) |

## Interpreting results

### Reading the α curve

Look at the **shape of the full α(t) curve**, not a single number:

1. **Short times** (< 1 ps): α ≈ 2 (ballistic). Normal.
2. **Intermediate times** (1–50 ps): α dips below 1. This is cage rattling —
   atoms oscillating in their local environment.
3. **Long times** (> 50 ps, varies): α approaches 1. This is the onset of
   diffusion. The MSD in this regime is meaningful for diffusivity.

If α never reaches the consistency band (1 ± 0.2 by default), the trajectory
is too short or the system is not diffusive at this temperature. mliport marks
the result as `not_diffusive`.

### Drift removal

Framework drift — the slow net displacement of the crystal lattice — should be
subtracted before computing MSD of the mobile species. The default is
`--drift-reference none` (no drift correction). For solid electrolytes, use
`--drift-reference nonmobile --drift-mode mass_weighted_com` to remove the
center-of-mass motion of all non-mobile atoms. Always report which drift
policy you used. With `none`, the result carries a warning and
`framework_drift_A` is `null` (not computed), not an array of zeros.

### Example

A full MSD analysis of lithium diffusion in LGPS at 800 K:

```bash
./bin/mliport-mace analyze runs/lgps-md-800K msd \
  --mobile Li --axes xyz \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --fit-start-ps 50 --fit-stop-ps 400 \
  --alpha-focus-band 0.5 1.5 --alpha-log-x
```

## Output files

Results are written to `<run>/analysis/msd/<analysis-id>/`:

| File | Contents |
|---|---|
| `results.json` | MSD values, α values, fit parameters, sampling status |
| `msd.csv` | Time (ps) vs. MSD (Å²) table |
| `alpha.csv` | Time (ps) vs. α(t) table |
| `msd.png`, `msd.svg` | MSD plot |
| `alpha.png`, `alpha.svg` | Local exponent plot with consistency band |

## Short trajectories

A 5-step trajectory is accepted — the analysis runs and reports an explicit
"insufficient diffusive window" status. This is useful for automated testing
and pipeline validation. For a real MSD measurement, provide at least several
hundred saved frames and let the fit window cover a region where α ≈ 1.

## Next steps

- To extract a diffusion coefficient with uncertainty: [Transport](transport.md)
- To map the sites and jumps that produce the diffusion: [GEMDAT](electrolyte-gemdat.md)
- To combine diffusivities at multiple temperatures into an activation energy: [Arrhenius](arrhenius.md)
