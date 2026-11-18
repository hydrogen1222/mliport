# RDF

`rdf` computes the radial distribution function between a center species and a
neighbor species, together with a coordination number.

```bash
mliport analyze runs/md rdf --center Li --neighbor S --rmax 6.0 --bins 200
```

Both species are required. For LGPS the useful pairs are Li–Li, Li–S and P–S.

## Physical background

### What is the RDF?

The **radial distribution function** (RDF), commonly written *g(r)*, measures
how the local density of one atomic species varies as a function of distance
from atoms of another species, relative to a uniform distribution:

$$
g(r) = \frac{1}{4 \pi r^2 \, \Delta r} \cdot \frac{V}{N_\text{center} \, N_\text{neighbor}} \sum_{i \in \text{center}} \sum_{j \in \text{neighbor}} \delta(r - r_{ij})
$$

In practice, the continuous *δ* is replaced by histogram bins of width Δ*r*.
The result tells you:

- **g(r) → 1** at large *r*: the local structure is indistinguishable from a
  homogeneous fluid. This is the expected long-range limit for any liquid or
  well-mixed system; if your RDF does not approach 1 at large *r*, either the
  cutoff is too short or the cell is too small.
- **g(r) > 1** at some *r*: atoms are found more often at this distance than
  in a random distribution — a **coordination shell**.
- **g(r) ≈ 0** at short *r*: the repulsive core. No atoms overlap.

### Coordination number

The coordination number is the integral of *g(r)* out to a cutoff:

$$
n(r_c) = 4 \pi \, \rho_\text{neighbor} \int_0^{r_c} g(r) \, r^2 \, dr
$$

where ρ is the number density of the neighbor species. The cutoff should be
placed at the first minimum of *g(r)*, which marks the boundary of the first
coordination shell. Reporting a coordination number without stating the cutoff
is meaningless.

### What the RDF tells you

An RDF is a **structural average**, not a bond definition. It answers: "on
average, how many neighbors of species B are within distance *r* of species A?"
A single snapshot RDF is noisy; a properly averaged RDF over many frames gives
the equilibrium pair correlation.

For solid electrolytes, the RDF reveals:

- **Framework integrity** — a crystalline framework shows sharp peaks at the
  lattice-determined distances. Broadening or shifting signals structural
  change; loss of peaks signals melting or amorphization.
- **Mobile-ion solvation** — the first peak of Li–S or Li–O RDF shows the
  average coordination shell geometry.
- **Model quality** — comparing MLIP RDF to DFT–MD reference shows whether
  the potential reproduces the local structure.

## Options you will use

| Option | Meaning |
|---|---|
| `--center SPECIES`, `--neighbor SPECIES` | the pair (both required) |
| `--rmax` | maximum radius in Å |
| `--bins` | histogram bins (default 200) |
| `--cn-cutoff` | the cutoff in Å used for the coordination number |
| `--stride` | use every Nth frame (default 1) |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | the analysis window |
| `--positions-convention`, `--frame-interval-fs` | metadata for external trajectories |

`--start-frame`, `--stop-frame` and `--stride` keep the cost down on long
trajectories:

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns rdf \
  --center Li --neighbor S --start-frame 0 --stop-frame 2000 --stride 5
```

## What comes out

The partial RDF, the coordination number and CSV, PNG and SVG artifacts under
`<run>/analysis/rdf/<analysis-id>/`. An unknown species, or too few valid frames,
fails closed, and an external trajectory needs its coordinate convention stated.

| File | Contents |
|---|---|
| `results.json` | The full RDF, coordination number, metadata and provenance |
| `rdf.csv` | *r* (Å), *g(r)* table |
| `rdf.png`, `rdf.svg` | The RDF plot |

## Interpreting results

### Reading g(r)

1. **First peak position** — the most probable nearest-neighbor distance. For
   Li–S in LGPS, typically around 2.4–2.5 Å.
2. **First peak height** — how structured the first shell is. A taller, sharper
   peak means a more ordered coordination environment.
3. **First minimum position** — this is where you should place `--cn-cutoff`
   for the coordination number.
4. **Long-range behavior** — *g(r)* should converge to 1 at distances well
   below `--rmax`. If it oscillates or trends away from 1, the cell may be too
   small or the trajectory too short.

### Common pitfalls

- **Cell-size artifact**: if `--rmax` exceeds half the shortest cell vector,
  the RDF at large *r* includes periodic images and becomes unreliable. Keep
  `--rmax` below *L*/2.
- **Poor statistics**: a few frames give a noisy RDF. Use thousands of frames
  from the production phase.
- **Wrong convention**: an RDF from wrapped coordinates gives the same result
  as unwrapped, because periodic distances are the same either way. But
  `--positions-convention` still matters for consistency checks.

### Comparing to reference

When validating a model, compare the MLIP RDF to a DFT–MD reference computed
with the same bin width, cutoff, and number of frames. Differences in peak
positions > 0.1 Å or coordination number > 0.5 suggest the model disagrees
with DFT on the local structure.

```bash
mliport analyze runs/md-800K rdf --center Li --neighbor O \
  --rmax 8 --bins 240 --cn-cutoff 3.0 --stride 5
```
