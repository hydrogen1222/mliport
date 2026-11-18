# Transport (kinisi)

The `transport` task estimates the tracer diffusion coefficient of mobile ions
with a proper uncertainty, and converts it to a Nernst–Einstein ionic
conductivity. It uses the [kinisi](https://github.com/kinisi-dev/kinisi)
library for statistically rigorous blocked averaging of the MSD.

```bash
./bin/mliport-mace analyze runs/lgps-md transport \
  --mobile Li --charge 1 --fit-start-ps 50 --temperature-K 800
```

!!! tip "MSD first, then transport"
    Run the [MSD](msd.md) task first to check that your trajectory is in the
    diffusive regime (α ≈ 1). The `transport` task extracts a diffusion
    coefficient — if the trajectory is not diffusive, the number will be
    unreliable regardless of the statistical treatment.

## Physical background

### From MSD to diffusivity

The Einstein relation connects the long-time limit of the MSD to the self-
diffusion coefficient:

$$
D = \lim_{t \to \infty} \frac{\text{MSD}(t)}{2 \, d \, t}
$$

where *d* is the number of spatial dimensions. In practice, you fit a straight
line to the MSD in the diffusive regime and extract *D* from its slope. The
challenge is estimating the **uncertainty** on this slope.

### Why kinisi?

A naive linear fit to the MSD underestimates the uncertainty because MSD values
at neighboring lag times are **correlated** — they share time origins. kinisi
addresses this by:

1. Splitting the trajectory into blocks and computing the MSD covariance
   matrix across lag times.
2. Performing a Bayesian fit that accounts for these correlations.
3. Reporting a posterior distribution for *D*, from which a credible interval
   (uncertainty) is drawn.

This is more statistically rigorous than a simple least-squares fit and avoids
reporting artificially small error bars.

### Nernst–Einstein conductivity

The **Nernst–Einstein relation** connects the tracer diffusion coefficient to an
ionic conductivity:

$$
\sigma_\text{NE} = \frac{n \, (z \, e)^2 \, D}{k_B \, T}
$$

where *n* is the number density of mobile ions, *z* is the charge number, *e*
is the elementary charge, *k*_B is Boltzmann's constant, and *T* is the
temperature.

!!! warning "Nernst–Einstein assumes uncorrelated motion"
    The Nernst–Einstein conductivity assumes that each ion diffuses
    independently. In real materials, ion–ion correlations (the **Haven ratio**)
    can make the true conductivity differ from the NE estimate. Use
    `--collective-conductivity` to compute the charge-collective (Onsager)
    conductivity, which captures these correlations.

### Haven ratio

The Haven ratio quantifies the deviation between tracer and charge transport:

$$
H_R = \frac{D_\text{tracer}}{D_\text{charge}}
$$

*H*_R = 1 means uncorrelated motion (NE is exact). *H*_R < 1 means cooperative
motion enhances conductivity beyond what individual diffusion would predict.
*H*_R > 1 means anti-correlated motion reduces it. For many solid
electrolytes, *H*_R is in the range 0.3–0.8.

## Usage

### Required options

| Option | Description |
|---|---|
| `--mobile SPECIES` | Mobile species (e.g., `Li`, `Na`) |
| `--charge Z` | Ionic charge in units of *e* (e.g., `1` for Li⁺) |
| `--fit-start-ps` | Start of the diffusive fit window (picoseconds) |

### Common options

## What comes out

`results.json` holds `D` with its uncertainty, the conductivity terms (the
tracer-based Nernst–Einstein one and, if requested, the collective one), the lag
grid and fit window, sampling diagnostics and the provenance (model identity,
software commit). `transport_summary.csv` has the headline numbers in one row.

A trajectory that is too short, a fit window below the required sampling, an
unknown frame interval, a coordinate-convention conflict, or a memory estimate
above the limit all fail closed with the reason. The reconstructed fallback is
used only when you allow it, and it is recorded.

## Reading the result

- A diffusivity from a finite trajectory deserves trust only when the fit window
  sits in the diffusive regime and the sampling error is reported. Look at the
  kinisi uncertainty, not just the mean `D`.
- Nernst–Einstein conductivity assumes uncorrelated ion motion, so it is not a
  full conductivity unless collective conductivity is computed as well. Say which
  one you report.
- A constant-volume cell constrains the mechanism, and a cramped cell can
  suppress or bias diffusion. State the cell and whether it was relaxed.
- Framework drift has to be handled explicitly, and the drift policy changes `D`.
- For *which* sites and jumps matter, use [GEMDAT](electrolyte-gemdat.md);
  transport gives rates, not mechanisms.

## What `publication_grade` does and does not mean

`publication_grade` is a provenance flag, not a verdict on the physics. On a
transport result it is `true` only when the exact continuous displacement array
was injected into kinisi and verified against the stored frames before the fit
(`publication_grade_scope` spells this out in `results.json`). It is `false` for
the `--allow-reconstructed-fallback` diagnostic path and for results that
contain an experimental component.

The nested `publication_grade` inside `collective_conductivity` and
`jump_diffusion` answers a second, different question: the charge grouping was
left as a single kinisi statistical group (`--collective-system-particles 1`),
so cross-group correlations are preserved. Grouping `> 1` is an experimental
diagnostic and is flagged as such.

Neither flag says that the trajectory is long enough, that the fit window is in
the diffusive regime, that the result transfers to another cell size or
temperature, or that the MLIP is accurate. Those are not decided by this
adapter, so every transport result records
`quality.statistical_adequacy = "not_assessed"` with the reason. Publication
readiness has to come from your own convergence checks (lag-window `alpha`,
posterior width, replicas/windows, cell-size checks), not from this flag.

```bash
mliport analyze runs/md-800K transport --mobile Li --charge 1 \
  --fit-start-ps 50 --lag-step-ps 5 --lag-stop-ps 500 \
  --temperature-K 800 --drift-reference nonmobile \
  --drift-mode mass_weighted_com --collective-conductivity --jump-diffusion
```

## Comparing estimators

The validation audit computes several estimators on the *same* trajectory and
reports them separately. They answer different questions and are not required to
agree:

| Estimator | What it is | Window |
|---|---|---|
| `--temperature-K` | — | Temperature for Nernst–Einstein conversion |
| `--dimensions` | `xyz` | Which Cartesian axes contribute (subset of `xyz`) |
| `--lag-step-ps` | — | Lag-time grid spacing (sparsifies the covariance matrix) |
| `--lag-stop-ps` | — | Maximum lag time (give both or neither with `--lag-step-ps`) |
| `--drift-reference` | `nonmobile` | Framework drift correction |
| `--drift-mode` | `arithmetic_mean` | Drift calculation method |

### Collective conductivity

| Option | Default | Description |
|---|---|---|
| `--collective-conductivity` | false | Also compute the Onsager (charge-collective) conductivity |
| `--collective-system-particles` | 1 | Number of kinisi statistical groups (not independent MD replicas) |

### Jump diffusion

| Option | Description |
|---|---|
| `--jump-diffusion` | Also run kinisi's jump-diffusion analyzer as a diagnostic |

### Sampling controls

| Option | Default | Description |
|---|---|---|
| `--n-samples` | 1000 | Posterior samples |
| `--n-walkers` | 32 | MCMC walkers |
| `--n-burn` | 500 | Burn-in steps |
| `--n-thin` | 10 | Thinning interval |
| `--random-seed` | 0 | Random seed for reproducibility |
| `--parser-memory-limit-gib` | 4 | Memory guard for the covariance matrix |

### Fallback option

| Option | Description |
|---|---|
| `--allow-reconstructed-fallback` | Use kinisi's unverified displacement reconstruction (marks result as `publication_grade=false`) |

## Example

A complete transport analysis of lithium in LGPS at 800 K:

```bash
./bin/mliport-mace analyze runs/lgps-md-800K transport \
  --mobile Li --charge 1 \
  --fit-start-ps 50 --lag-step-ps 5 --lag-stop-ps 500 \
  --temperature-K 800 \
  --drift-reference nonmobile --drift-mode mass_weighted_com \
  --collective-conductivity --jump-diffusion
```

## Output files

Results are written to `<run>/analysis/transport/<analysis-id>/`:

| File | Contents |
|---|---|
| `results.json` | *D* with uncertainty, NE conductivity, collective conductivity (if requested), sampling diagnostics, provenance |
| `transport_summary.csv` | Headline numbers in one row |

## Interpreting results

### What to report

When reporting transport results:

1. **The diffusivity and its uncertainty** — not just the mean *D*.
2. **Which conductivity** — tracer-based Nernst–Einstein, or collective.
3. **The fit window** — where it starts and ends, and why.
4. **The drift policy** — framework drift removal changes *D*.
5. **The cell** — constant-volume NVT constrains the mechanism; NPT lets the
   cell breathe. A small cell suppresses diffusion.

### Comparing estimators

Different estimators of *D* from the same trajectory answer different questions:

| Estimator | Description | When to use |
|---|---|---|
| Plain OLS | Slope of MSD / (2d), no covariance | Quick check |
| kinisi posterior | Tracer *D* with credible interval | Publication |
| Nernst–Einstein | σ = n(ze)²D/(k_BT) | Conductivity from tracer *D* |
| Collective | Onsager conductivity from charge-weighted MSD | Full conductivity with correlations |

When the kinisi-to-OLS ratio deviates strongly from 1, the fit window may not
be in the diffusive regime — check the [MSD α curve](msd.md).

### Failure modes

The task fails closed (with a clear reason) when:

- The trajectory is too short for the requested fit window
- Frame interval is unknown (external trajectory without `--frame-interval-fs`)
- Coordinate convention conflicts with stored metadata
- Estimated memory exceeds `--parser-memory-limit-gib`

A short trajectory returns an insufficient-sampling status — mliport never
fabricates a diffusion coefficient.

## Next steps

- To understand the diffusion mechanism (which sites, which jumps): [GEMDAT](electrolyte-gemdat.md)
- To extract an activation energy from multiple temperatures: [Arrhenius](arrhenius.md)
- To go back and check the diffusion regime: [MSD](msd.md)
