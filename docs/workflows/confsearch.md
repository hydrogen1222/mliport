# Configuration search

`mliport confsearch` searches for **low-energy configurations of one adsorbate
on one side of a periodic slab**. It runs many independent MD chains at
elevated temperature, quenches what they find, deduplicates the minima and
exports a ranked candidate list for DFT re-ranking.

## What it does — and what it does not

Configuration search is a **screening** tool. It reports the lowest energy
minimum it found with evidence of how thoroughly it searched, but it never
claims to have found the global minimum. It does not compute adsorption
energies, free energies or Boltzmann populations, and it never changes the
cell. The deliverable is a ranked list of candidates to hand to DFT.

## Two start modes

**Mode A — Prepared starts:** you supply pre-built slab+adsorbate structures:

```bash
mliport confsearch run starts/*.extxyz \
  --adsorbate-indices 300-341 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --cycles 5 --output runs/lds-on-lpsc
```

All computing subcommands (`confsearch run`, `resume` and `extend`) accept
`--device cpu|cuda|cuda:N`; omit it to use the configured device (default CPU).
For `resume` and `extend`, the recorded model/device identity remains subject
to checkpoint validation; changing it is not a silent restart or CPU fallback.

**Mode B — Automatic placement:** mliport places a molecule on a slab for you:

```bash
mliport confsearch run \
  --slab slab.vasp --molecule conformers.xyz --n-starts 20 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --output runs/placed
```

You can preview the placement without loading a model:

```bash
mliport confsearch place --slab slab.vasp \
  --molecule conformers.xyz --n-starts 20 --output runs/placement-check
```

## Python API

```python
from mliport.api import run_confsearch

results = run_confsearch(
    starts=["starts/start_000.extxyz"],
    adsorbate_indices="300-341",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    cycles=5,
    output_dir="runs/lds-on-lpsc",
    verbose=False,
)
print(results["ranking"]["best_energy_eV"])
```

The function returns the `search_results.json` content as a dictionary.

## How the search works

Every start is quenched first (positions-only FIRE relaxation at fixed cell),
giving a "multi-start relaxation only" baseline. Then each replica runs
`--cycles` heat–hold–cool cycles:

1. **Heat and hold** at `CS_TEMPERATURE_HIGH` with a Langevin thermostat
2. **Sample snapshots** during the hold phase, then cool linearly to
   `CS_TEMPERATURE_LOW`
3. **Quench** every snapshot and the end-of-cycle frame with FIRE at fixed cell
4. **Classify and deduplicate** each minimum
5. **Metropolis test** at `CS_ACCEPT_TEMPERATURE` to accept or reject the
   end-of-cycle minimum as the next starting point

Replicas are independent, and the round-robin schedule means the search
has a usable answer after every cycle.

### Slab mobility

By default the slab is frozen during MD (`CS_MD_MOBILE = adsorbate`):

- `adsorbate`: only the adsorbate moves during MD (default)
- `adsorbate_and_top`: also mobilizes the top `CS_TOP_DEPTH` Å of the surface
- `all_free`: mobilizes everything that is not fixed in the input

Quenches always relax every atom that is not fixed, regardless of
`CS_MD_MOBILE`.

## Pre-search checks

Before any dynamics, mliport validates the model:

- **Model identity** (backend, version, task, head, dtype, inference mode,
  device and any DFT-D3 correction) is recorded in `run_context.json`.
- A model with `direct_forces = true` is refused unless you pass
  `--allow-direct-forces`, and every result carries a warning that the
  ranking is not meaningful.
- A **gradient preflight** (finite-difference energy–force consistency check)
  is run on the first prepared start.
- With hydrogen in the mobile set, timesteps above 1.0 fs are refused. The
  default is 0.5 fs.

## Automatic placement (Mode B)

Each start picks a conformer uniformly at random, centres it, applies a
uniform random rotation, draws in-plane coordinates, and solves the height by
bisection so that the closest adsorbate–slab pair reaches a target ratio of
the covalent-radii sum.

| Option | Default | Meaning |
|---|---|---|
| `CS_PLACE_CONTACT_SCALE` | 1.5 | Contact distance ratio for non-bonded placement |
| `CS_PLACE_ANCHOR` | unset | 0-based anchor atom indices in the molecule file |
| `CS_PLACE_ANCHOR_SCALE` | 1.1 | With an anchor, start already at bonding distance |
| `CS_PLACE_OTHER_MIN_SCALE` | 1.3 | Every other pair must stay above this ratio |
| `CS_PLACE_ANCHOR_TARGETS` | unset | Require the closest slab atom to be one of these elements |

**Anchored placement** is the chemisorption-first mode: the lowest atom of
the rotated molecule is drawn from the anchor set, and the height puts it at
bonding distance (inside the bond criterion used by the classifier).

## Deduplication

Two minima are considered the same when their energies differ by at most
`CS_DEDUP_ENERGY_TOL` **and** either:

- **G (geometric identity):** with fixed atoms anchoring the slab, every
  adsorbate atom is within `CS_DEDUP_GEOM_TOL` of the corresponding atom in
  the other minimum (absorbs quench noise in flat basins).
- **D (fingerprint distance):** the focused Oganov–Valle fingerprint distance
  is at most `CS_DEDUP_FP_TOL` (catches symmetry-equivalent sites and
  permutation-equivalent conformers like rotated methyl groups).

The tolerances are deliberately conservative: a **false merge** silently
deletes a candidate that DFT might have ranked first, while a false split
only costs a duplicate.

## Adsorbate indices

`cs_adsorbate_indices` uses **0-based indexing** (like ASE). CP2K and VMD
count from 1. Wrong indices are the most common mistake — a selection that is
not one connected covalent molecule fails closed instead of searching nonsense.

## Output files

A run writes:

| File | Contents |
|---|---|
| `search_results.json` | Ranking, discovery curve, saturation diagnostics |
| `run_context.json` | Model identity and gradient preflight |
| `resolved_config.json` | Complete resolved configuration |
| `artifacts.json` | File listing with hashes |
| `run.log` | Human-readable log |
| `starts/` | Prepared/placed starting structures |
| `raw/` | Raw MD trajectories |
| `minima/` | Deduplicated minima |
| `diagnostics/discovery.csv` | Per-cycle discovery statistics |
| `top/rank_NNN/` | Exported candidates (see below) |
| `checkpoints/` | Per-replica, per-cycle checkpoints |

Each exported rank contains:

- `POSCAR` — direct coordinates with selective dynamics for fixed atoms
- `structure.extxyz` — ASE-readable structure
- `cp2k_coord.xyz`, `cp2k_cell.inc`, `cp2k_fixed_atoms.inc` — CP2K input
  fragments (1-based atom indices with `a..b` ranges)
- `minimum.json` — metadata for this minimum

## Resuming

```bash
mliport confsearch resume runs/lds-on-lpsc
```

A checkpoint is written after the start-quench stage and after every
(replica, cycle) unit. Resuming refuses to continue if anything in the
identity changed — model, inputs, regions, seed or any option other than
output and verbosity. On CPU, an interrupted-and-resumed search is bitwise
identical to an uninterrupted one. Raw files from interrupted attempts are
moved to `incomplete_attempt_N/`, never deleted.

## Risks and caveats

- **Model quality:** Foundation models trained on inorganic bulk may describe
  organic adsorbates on sulfide surfaces poorly. High temperatures amplify
  this. A large `reacted` fraction is a symptom — re-rank exports with DFT.
- **Dispersion:** Dispersion decides whether alkyl chains lie flat or stand
  up. Use `--d3` when the model's training data lacks it. The D3 correction
  is part of the model identity, so a search with D3 can never be merged with
  one without.
- **Frozen slab bias:** Freezing the slab during MD biases exploration toward
  the initial surface geometry. `adsorbate_and_top` relaxes this at the cost
  of noisier energies; check `far_field_slab_changed` in the diagnostics.
- **Cell quality:** The cell must already be the relaxed lattice of the same
  model; confsearch does not check this and never changes the cell.
- **Periodicity:** Self-image and vacuum distance guards are geometric. They
  do not prove that the model's receptive field is decoupled across periodic
  images.
- **Electrostatics:** Short-range MLIPs have no long-range electrostatics. DFT
  single points on one-sided slabs need a dipole correction.
- **Chemisorption:** Bonds formed on surfaces the model was not trained on are
  the least trustworthy part of the result. Confirm every `chemisorbed`
  candidate with DFT.
