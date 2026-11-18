# GEMDAT electrolyte analysis

`electrolyte` uses [GEMDAT](https://gemdat.readthedocs.io/) to map the mobile
ion's positions onto crystallographic sites, count site-to-site jumps, and look
for percolating pathways. It is about **mechanism**: where the ion sits and how
it hops, as opposed to how fast it diffuses overall.

```bash
mliport analyze runs/md electrolyte --mobile Li --sites sites.yaml --temperature-K 800
mliport analyze runs/md electrolyte --mobile Li --discover-sites-from-density
```

!!! tip "Transport first, mechanism second"
    Run [transport](transport.md) first to establish that there is diffusion and
    to measure its rate. GEMDAT tells you *how* the diffusion happens — which
    sites are occupied, which jumps are frequent, and whether a percolating
    pathway exists.

## Physical background

### Site-based mechanism analysis

In a crystalline solid electrolyte, mobile ions do not diffuse through a
featureless medium. They hop between discrete **crystallographic sites** — local
energy minima in the potential-energy landscape. The transport mechanism is
determined by which sites exist, how they are connected, and how often the ions
jump between them.

GEMDAT turns a continuous MD trajectory into a discrete event sequence:

1. **Define the sites** — either from a known crystal structure or by
   discovering them from the time-averaged density of the mobile species.
2. **Map each frame** — assign each mobile ion to its nearest site (within a
   capture radius).
3. **Detect transitions** — when an ion changes its assigned site, that is a
   transition event.
4. **Filter to jumps** — transitions that satisfy a minimum-residence criterion
   are counted as jumps (a very brief visit followed by a return is not a jump).
5. **Analyze connectivity** — build a transition matrix, compute jump rates,
   and look for percolating pathways along chosen crystallographic axes.

### Site discovery from density

When you do not have a reference site geometry, GEMDAT can discover sites from
the trajectory itself:

1. Build a 3-D number-density histogram of the mobile species on a grid with
   spacing `--resolution-A` (default 0.5 Å).
2. Find density peaks above `--background-level` (default 0.1 of the maximum).
3. Segment the density around the peaks to define site volumes.
4. Convert the site volumes to a pymatgen `Structure` for downstream analysis.

This is exploratory — the sites depend on the resolution, the background
level, and the trajectory length. Always state the discovery parameters.

### Free-energy landscape and percolation

From the density histogram, GEMDAT computes a **free-energy map**:

$$
F(\mathbf{r}) = -k_B T \ln \frac{\rho(\mathbf{r})}{\rho_\text{max}}
$$

where ρ(**r**) is the local density. The free-energy barriers between sites
are the occupancy-derived analogue of migration barriers, but they are **not**
the same as NEB potential-energy barriers — they include finite-temperature
entropic contributions and depend on the trajectory length and the grid
resolution.

A **percolating path** is a connected route through the free-energy landscape
that spans the simulation cell along a given crystallographic axis. The
existence of a percolating path along *x*, *y*, and *z* suggests (but does
not prove) 3-D long-range ionic transport.

### Jump diffusivity

GEMDAT can estimate a diffusivity from the jump statistics:

$$
D_\text{jump} = \frac{\langle l^2 \rangle}{2 \, d \, \tau}
$$

where ⟨*l*²⟩ is the mean-squared jump distance, *d* is the dimensionality
(`--jump-dimensions`), and τ is the mean time between jumps. This is a
**diagnostic cross-check**, not a replacement for the kinisi transport
analysis. Jump diffusivity and tracer diffusivity can differ because
of correlated motion, incomplete site assignment, and the difference between
discrete jumps and continuous displacement.

## Options you will use

| Option | Meaning |
|---|---|
| `--mobile SPECIES` | the mobile species (required) |
| `--sites PATH` or `--discover-sites-from-density` | an explicit site file, or density-based discovery (exactly one is required) |
| `--resolution-A` | density grid spacing in Å (default 0.5) |
| `--background-level` | density peak detection threshold (default 0.1) |
| `--site-radius-A` | capture radius around a site |
| `--minimal-residence` | minimum residence, in frames, to count as a site visit (default 0) |
| `--jump-dimensions` | dimensionality of the jump analysis: 1, 2 or 3 (default 3) |
| `--percolation-axes` | axes tested for percolation (default `xyz`) |
| `--temperature-K` | temperature for free-energy and derived quantities |
| `--drift-reference`, `--drift-mode`, `--drift-indices` | framework-drift correction (mode default `arithmetic_mean`) |

### Site source

You must choose exactly one:

- **`--sites PATH`** — an explicit site geometry file (CIF, POSCAR, or any
  format pymatgen can read). The lattice must match the trajectory cell. This
  is the reproducible choice: the sites are known and fixed.
- **`--discover-sites-from-density`** — discover sites from the trajectory's
  own density map. This is exploratory: the sites depend on the trajectory,
  the resolution, and the background level. Always state these parameters.

## What comes out

`results.json` has the site list, visit and jump counts, jump vectors and
residence statistics, and the percolation diagnostics, with the provenance that
every analysis carries.

| Section | Contents |
|---|---|
| `summary` | Site count, jump count, jump diffusivity, temperatures, drift policy, GEMDAT version |
| `arrays` | Density histogram, free energy, transition and jump matrices, peak voxels |
| `tables` | Transition events, residence times, jump table with distances, jump rates |
| `structures` | Site geometry, occupancy-annotated sites, reference frame |
| `paths` | Percolating path coordinates and free energies for each tested axis |

### Status codes

| Status | Meaning |
|---|---|
| `success` | Transitions were computed and jumps are physical |
| `no_events` | The analysis ran and genuinely found no transitions (physical zero) |
| `backend_error` | GEMDAT itself failed — jump counts are **unknown**, not zero |
| `invalid_input` | Site geometry is incompatible with the trajectory |

If GEMDAT fails, that is reported as a **backend failure**, never as "zero
jumps". Zero jumps is reported only when the analysis ran and found none.
mliport includes an independent nearest-site cross-check to distinguish a
true "no transitions" result from a GEMDAT stacking error on empty event
collections.

## Interpreting results

### What to report

1. **Where the sites came from** — explicit file or density-based discovery,
   and if discovered, the resolution and background level.
2. **Jump counts and the residence criterion** — `--minimal-residence` changes
   how many events count as jumps.
3. **Drift policy** — framework drift correction changes the mapping.
4. **Percolation axes** — which dimensions and axes were tested.
5. **Temperature** — affects the free-energy landscape.

### Comparing to transport

The GEMDAT jump diffusivity and the kinisi tracer diffusivity answer
different questions. They should be in the same order of magnitude, but
they will not agree exactly. If they differ by more than a factor of 5,
investigate: the site definition may not capture the mechanism, or the
trajectory may be too short for the jump statistics to converge.

### Cost

Discovering sites on a full 1 ns, 400-atom trajectory is expensive. The
acceptance suite runs it on a 30 ps segment:

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns electrolyte \
  --mobile Li --discover-sites-from-density
```

### Example

A full mechanism analysis with density-based site discovery:

```bash
mliport analyze runs/md-800K electrolyte --mobile Li \
  --discover-sites-from-density --resolution-A 0.2 --minimal-residence 2 \
  --jump-dimensions 3 --percolation-axes xyz --temperature-K 800
```

GEMDAT's partial-occupancy warnings come from the upstream library and do not
change the requested site set.

## Next steps

- To measure the diffusion rate with proper uncertainty: [Transport](transport.md)
- To extract an activation energy from multiple temperatures: [Arrhenius](arrhenius.md)
- To check whether the trajectory is in the diffusive regime: [MSD](msd.md)
