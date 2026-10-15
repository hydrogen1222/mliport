# Transport (kinisi)

`transport` estimates the tracer diffusion coefficient of the mobile ion from an
MD trajectory, with an uncertainty, using the kinisi analyzer, and converts it to
a Nernst–Einstein ionic conductivity for charged species.

```bash
mliport analyze runs/md transport --mobile Li --charge 1 \
  --fit-start-ps 50 --temperature-K 800
```

You need a production trajectory with a known **saved-frame** spacing
(`--frame-interval-fs` for external files), the mobile species, its charge in
units of e, and the start of the diffusive fit. Equilibration frames are left out.

## Options you will use

| Option | Meaning |
|---|---|
| `--mobile SPECIES` | the mobile species (required) |
| `--charge` | ionic charge in units of e (required; used for conductivity) |
| `--fit-start-ps` | where the diffusion fit starts (required) |
| `--lag-step-ps`, `--lag-stop-ps` | an explicit, sparser kinisi lag grid; give both or neither |
| `--dimensions` | which Cartesian axes enter the estimate, as a subset of `xyz` (default `xyz`) |
| `--temperature-K` | temperature for the Nernst–Einstein conversion |
| `--drift-reference`, `--drift-mode`, `--drift-indices` | framework-drift correction (mode default `arithmetic_mean`) |
| `--collective-conductivity`, `--collective-system-particles` | also compute the charge-collective (Onsager) conductivity; kinisi's index-ordered statistical groups (default 1; they are not independent MD replicas) |
| `--jump-diffusion` | also run kinisi's jump-diffusion analyzer as a diagnostic |
| `--n-samples`, `--n-walkers`, `--n-burn`, `--n-thin`, `--random-seed` | kinisi sampling controls (defaults 1000, 32, 500, 10, 0) |
| `--parser-memory-limit-gib` | refuse to parse when the estimated memory peak exceeds this (default 4) |
| `--allow-reconstructed-fallback` | diagnostic only: fall back to kinisi's unverified displacement reconstruction; the result is marked `publication_grade=false` |

The lag grid sparsifies lag times, not trajectory frames. On a long trajectory a
sparse grid is what keeps kinisi's covariance matrix inside the memory guard.

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
| plain OLS baseline | `D = slope(MSD)/(2d)` with explicit unit conversion | aligned to the kinisi fit start |
| native MSD diagnostic | the product's explicit-range OLS diagnostic (regime-aware) | the recorded fit window |
| kinisi posterior | tracer diffusion with a credible interval | the kinisi lag and fit window |
| Nernst–Einstein | `sigma_NE = n (z e)^2 D_tracer / (k_B T)` | derived from the tracer estimate |

The report records `kinisi_to_plain_D_ratio` and raises an
`estimator_disagreement_warning` when the ratio falls below 0.5 or above 2. That
is a prompt to explain the difference (windows, drift, regime), not a pass/fail
threshold. The unit audit is a known-answer check: `1 Å²/ps = 1e-8 m²/s`, so a 3D
MSD slope of `6 Å²/ps` gives `D = 1e-8 m²/s`.

A large disagreement usually means the fitted window is not in the diffusive
regime, so check the local `alpha`: a "demonstration, not converged" protocol can
show `alpha ≈ 2` even though the trajectory is finite and healthy.

## A tested example

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns transport \
  --mobile Li --charge 1 --fit-start-ps 100 \
  --lag-step-ps 1 --lag-stop-ps 200
```

On the repository's 1 ns GRACE LGPS run this returns `D = 1.885e-09 m²/s` with a
95% credible interval of `[1.492e-09, 2.275e-09] m²/s` (fit window 100–200 ps).
Short trajectories return an insufficient-sampling status instead; mliport never
makes up a diffusion coefficient for them.
