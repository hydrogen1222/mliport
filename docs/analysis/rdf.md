# RDF

**Purpose.** Compute the radial distribution function between a centre and a
neighbour species, with a coordination-number estimate.

**Minimal command.**

```bash
mliport analyze runs/md rdf --center Li --neighbor O --rmax 6.0 --bins 200
```

**Important options.**

| Option | Meaning |
|---|---|
| `--center SPECIES` | centre species (required) |
| `--neighbor SPECIES` | neighbour species (required) |
| `--rmax Å` | maximum radius |
| `--bins` | histogram bins |
| `--cn-cutoff Å` | coordination-number cutoff radius |
| `--stride` | frame stride for the average |
| `--start-frame`, `--stop-frame`, `--include-equilibration` | analysis window |
| `--positions-convention`, `--frame-interval-fs` | external trajectory metadata |

**Inputs / outputs.** A trajectory and the species pair; `analysis.json`
plus RDF/coordination tables and plots under `<run>/analysis/<id>/`.

**Failure semantics.** Unknown species or too few valid frames fails closed.
Wrapped/unwrapped conventions must be stated for external trajectories.

**Scientific caveats.** The RDF is a structural average, not a bond
definition: report the cutoff used for coordination and the analysed window.
A single-snapshot RDF is a diagnostic; a converged RDF needs sufficient
frames and a homogeneous cell.

**Example.**

```bash
mliport analyze runs/md-800K rdf --center Li --neighbor O \
  --rmax 8 --bins 240 --cn-cutoff 3.0 --stride 5
```
