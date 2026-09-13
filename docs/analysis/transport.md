# Transport (kinisi)

**Purpose.** Estimate self-diffusion coefficients and ionic conductivity from
an MD trajectory with explicit uncertainties, using the kinisi analyzer, plus
the Nernst-Einstein relation for charged species.

**Minimal command.**

```bash
mliport analyze runs/md transport --mobile Li --charge 1 \
  --fit-start-ps 50 --temperature-K 800 --dimensions 3
```

**Important options.**

| Option | Meaning |
|---|---|
| `--mobile SPECIES` | mobile species (required) |
| `--charge eV` | ionic charge in units of e (required; used for conductivity) |
| `--fit-start-ps` | start of the diffusion fit (required) |
| `--lag-step-ps`, `--lag-stop-ps` | explicit kinisi lag grid (must be given together) |
| `--dimensions {1,2,3}` | dimensionality of the diffusion/conductivity estimate |
| `--temperature-K` | temperature used in the Nernst-Einstein conversion |
| `--drift-reference`, `--drift-mode`, `--drift-indices` | framework-drift correction |
| `--collective-conductivity`, `--collective-system-particles` | charge-collective (Onsager) conductivity |
| `--jump-diffusion` | also run kinisi's jump-diffusion analyzer |
| `--n-samples`, `--n-walkers`, `--n-burn`, `--n-thin`, `--random-seed` | kinisi sampling controls |
| `--parser-memory-limit-gib` | fail before parsing when the trajectory is too large |
| `--allow-reconstructed-fallback` | opt-in fallback when native artifacts are missing (recorded) |

**Inputs.** A production MD trajectory with known **saved-frame** spacing
(`--frame-interval-fs` for external inputs), mobile species, charge and
temperature. Equilibration frames are excluded unless explicitly included.

**Outputs.** `analysis.json` with `D` and its uncertainty, conductivity terms
(self/Nernst-Einstein and, when requested, collective), the lag grid and fit
window, sampling diagnostics, and provenance (model identity, software
commit).

**Failure semantics.** Too-short trajectories, a fit window below the
required sampling, an unknown frame interval, a wrapped-convention conflict
or exceeding `--parser-memory-limit-gib` fail closed with the reason. A
reconstructed fallback is only used when explicitly allowed and is recorded.

**Scientific caveats.**

- Diffusivity from a finite trajectory is only trustworthy when the fit
  window is inside the diffusive regime and the sampling error is reported.
  Check the kinisi posterior/uncertainty, not just the mean `D`.
- Nernst-Einstein conductivity assumes uncorrelated ion motion; it is an
  upper-bound-like estimate unless collective (Onsager) conductivity is also
  computed. Report which one you used.
- **Fixed cell:** simulations at constant volume/cell constrain the
  mechanism; a packed cell can suppress or bias diffusion. State the cell and
  whether it was relaxed.
- Framework drift must be handled explicitly; the drift policy changes `D`.
- For mechanism questions (which sites/jumps), use
  [GEMDAT electrolyte](electrolyte-gemdat.md); transport gives rates, not
  mechanisms.

**Example.**

```bash
mliport analyze runs/md-800K transport --mobile Li --charge 1 \
  --fit-start-ps 50 --lag-step-ps 5 --lag-stop-ps 500 \
  --dimensions 3 --temperature-K 800 --drift-reference nonmobile \
  --drift-mode mass_weighted_com --collective-conductivity --jump-diffusion
```
