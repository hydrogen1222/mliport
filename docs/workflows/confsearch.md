# Configuration search

`mliport confsearch` searches for low-energy configurations of **one adsorbate
on one side of a periodic slab**. It needs a model on an explicit backend; see
[Backends](../backends/index.md) for the engine environments and
[Installation](../installation.md) for how they are built. It runs many independent MD chains, quenches
what they find, deduplicates the minima and exports a ranked candidate list for
DFT re-ranking.

It does **not** compute adsorption energies, free energies or Boltzmann
populations, it does not change the cell, and it never claims to have found a
global minimum. The deliverable is a screening list: "lowest found, with
evidence about how thoroughly it was searched".

```bash
.venv-mace/bin/mliport confsearch run starts/*.extxyz \
  --adsorbate-indices 300-341 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --cycles 5 --output runs/lds-on-lpsc
```

Mode A takes structures you prepared; mode B places a molecule above a slab for
you:

```bash
.venv-mace/bin/mliport confsearch run \
  --slab slab.vasp --molecule conformers.xyz --n-starts 20 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --output runs/placed
```

`cs_adsorbate_indices` is **0-based, like ASE**; CP2K and VMD count from 1.
Wrong indices are the most common mistake, so a selection that is not one
connected covalent molecule fails closed instead of searching nonsense.

## What the search does

Every start is quenched first, which already gives a "multi-start relaxation
only" baseline. Then each replica runs `--cycles` heat–hold–cool cycles:

- heat and hold at `CS_TEMPERATURE_HIGH` with a Langevin thermostat,
- take frames during the hold phase and cool the chain linearly to
  `CS_TEMPERATURE_LOW`,
- quench every snapshot and the end-of-cycle frame with FIRE at fixed cell,
- classify and deduplicate each minimum,
- accept or reject the end-of-cycle minimum with a Metropolis test at
  `CS_ACCEPT_TEMPERATURE`.

Replicas are independent, and the round-robin schedule means the search has a
usable answer after every cycle. By default the slab is frozen during MD
(`CS_MD_MOBILE = adsorbate`); `adsorbate_and_top` also mobilizes the top
`CS_TOP_DEPTH` Å of the surface, and `all_free` mobilizes everything that is not
fixed. Quenches always relax every atom that is not fixed in the input.

## Before the search starts

The ranking is only meaningful if the model's forces are the gradient of its
energy, so mliport checks the model identity and measures energy–force
consistency on the first prepared start before any dynamics:

- the model identity (backend, version, task, head, dtype, inference mode,
  device and any DFT-D3 correction) is recorded in `run_context.json`;
- a model that reports `direct_forces = true` is refused unless you pass
  `--allow-direct-forces`, and then every result carries the warning that the
  ranking is not meaningful;
- otherwise a finite-difference spot check compares `-F·u` with the model's
  energy along a few directions; it is stored as `gradient_preflight`.

Timestep is checked too: with hydrogen in the mobile set, more than 1.0 fs is
refused. The default is 0.5 fs.

## Automatic placement

In mode B, each start picks a conformer uniformly at random, centres it, applies
a uniform random rotation, draws in-plane coordinates and then solves the height
by bisection so that the closest adsorbate–slab pair reaches a target ratio of
the covalent-radii sum:

| Option | Default | Meaning |
|---|---|---|
| `CS_PLACE_CONTACT_SCALE` | 1.5 | closest-contact start (not bonded) |
| `CS_PLACE_ANCHOR` | unset | 0-based anchor atoms in the molecule file |
| `CS_PLACE_ANCHOR_SCALE` | 1.1 | with an anchor, start already bonded |
| `CS_PLACE_OTHER_MIN_SCALE` | 1.3 | every other pair must stay above this |
| `CS_PLACE_ANCHOR_TARGETS` | unset | require the closest slab atom to be one of these elements |

Anchored placement is the chemisorption-first mode: the lowest atom of the
rotated molecule is drawn from the anchor set, and the height puts it at
`CS_PLACE_ANCHOR_SCALE` times the covalent-radii sum, i.e. already inside the
bond criterion used by the classifier.

## Deduplication

Two minima are the same when their energies differ by at most
`CS_DEDUP_ENERGY_TOL` **and** either

- `G`, direct geometric identity: with fixed atoms anchoring the slab and a
  fixed atom order, every adsorbate atom is within `CS_DEDUP_GEOM_TOL` of the
  same atom in the other minimum (this absorbs quench noise in flat basins), or
- `D`, the focused Oganov–Valle fingerprint distance, which is at most
  `CS_DEDUP_FP_TOL` (this is what catches symmetry-equivalent sites and
  permutation-equivalent conformers such as a rotated methyl group).

The fingerprint is computed with ASE's discretisation so it can be checked
against `OFPComparator` bin by bin, unlike the NEB pre-flight in
[NEB](neb.md), which measures energy-force consistency instead. Its tolerance is calibrated for **zero
false merges**: a false merge silently deletes a candidate that DFT might have
ranked first, while a false split only costs a duplicate, so the defaults are
deliberately conservative and prefer duplicates.

**Calibration limits.** The shipped `CS_DEDUP_FP_TOL` comes from the labelled
sweep in `tests/mliport/mliport/confsearch/test_fingerprint_calibration.py`,
which uses CH3OH inside the LGPS **bulk** cell, builds the "same basin" class
from random perturbations rather than quenches, and takes the
"symmetry-equivalent" class from exact rigid images (whose `D` is ~0). The
tolerance is therefore proven to give zero false merges, but it is untested for
*noisy* equivalent minima: in practice `D` merges little beyond exact images and
the `G` rule plus `CS_DEDUP_ENERGY_TOL` do most of the work. That is the safe
direction (duplicates instead of deleted candidates). Re-measuring the `D`
distribution on a real slab and model is part of the P5 validation campaign
(`validation/science/scripts/confsearch_suite.py`).

## Outputs

A run writes `resolved_config.json`, `run_context.json`, `run.log`,
`search_results.json`, `artifacts.json`, `starts/`, `raw/`,
`minima/`, `diagnostics/discovery.csv`, `top/rank_NNN/` and `checkpoints/`; see
[Output files](../outputs.md) for the shared conventions.

Each exported rank contains `POSCAR` (direct coordinates with selective
dynamics for the fixed atoms), `structure.extxyz`, `cp2k_coord.xyz`,
`cp2k_cell.inc`, `cp2k_fixed_atoms.inc` and `minimum.json`. The CP2K include uses
the `&MOTION/&CONSTRAINT/&FIXED_ATOMS` `LIST` convention: 1-based atom indices
with `a..b` ranges, as documented in the CP2K manual. The exported geometry
wraps the slab into the cell and makes the adsorbate whole under the
minimum-image convention, so it equals the minimum up to lattice translations.

`search_results.json` reports the ranking, the discovery curve, the saturation
diagnostics (hit counts, how many replicas found the best minimum, how late it
was first found) and the baseline comparison with the multi-start relaxation
only. The campaigns that archive this kind of evidence live in
[Validation](../validation.md). Read it as "lowest found": a finite search cannot prove a global minimum.

## Resuming

```bash
.venv-mace/bin/mliport confsearch resume runs/lds-on-lpsc
```

A checkpoint is written after the start-quench stage and after every
(replica, cycle) unit. Resuming refuses to continue if anything in the identity
changed — model, inputs, regions, seed or any option other than output and
verbosity — and re-runs the interrupted unit from its checkpointed random state.
On CPU, an interrupted-and-resumed search is bitwise identical to an
uninterrupted one. Raw files of an interrupted attempt are moved to
`incomplete_attempt_N/`, never deleted.

## Risks and caveats

- Foundation models trained on inorganic bulk may describe an organic adsorbate
  on a sulfide surface poorly, and high temperatures amplify that. A large
  `reacted` fraction is a symptom; re-rank exports with DFT.
- Dispersion decides whether alkyl chains lie flat or stand up; use `--d3` when
  the model's reference lacks it. The correction is part of the model identity,
  so a search with D3 can never be merged with one without.
- Freezing the slab during MD biases exploration towards the initial surface.
  `adsorbate_and_top` relaxes that at the cost of noisier energies; watch
  `far_field_slab_changed` in the diagnostics.
- The cell must already be the relaxed lattice of the same model; confsearch
  does not check that and never changes the cell.
- The self-image and vacuum distances are geometric guards. They do not prove
  that the model's receptive field is decoupled across the periodic images.
- Short-range MLIPs have no long-range electrostatics, so DFT single points or
  relaxations of one-sided slabs need a dipole correction.
- Chemisorption bonds formed on a surface the model was not trained on are the
  least trustworthy part of the result; confirm every `chemisorbed` candidate.

The Python API mirrors the CLI and returns the `search_results.json` content:

```python
from mliport import run_confsearch

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

The interactive terminal app offers the same two start modes on its
configuration screen, including the anchor atom, anchor target elements and the
random seed; `resume`, `extend` and `merge` are available as commands only; the capability matrix records exactly which interface
can express what. `mliport confsearch place` runs the placement alone (no
model) so you can inspect the starts before spending GPU time:

```bash
.venv-mace/bin/mliport confsearch place --slab slab.vasp \
  --molecule conformers.xyz --n-starts 20 --output runs/placement-check
```
