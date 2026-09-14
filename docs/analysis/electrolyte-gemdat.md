# GEMDAT electrolyte analysis

**Purpose.** Map mobile-ion positions onto crystallographic/interstitial
sites, count site-to-site jumps and percolation pathways, and report
mechanism-level diagnostics with GEMDAT.

**Minimal command.**

```bash
mliport analyze runs/md electrolyte --mobile Li --sites sites.yaml \
  --temperature-K 800
```

or discover sites from the mobile-species density:

```bash
mliport analyze runs/md electrolyte --mobile Li --discover-sites-from-density
```

**Important options.**

| Option | Meaning |
|---|---|
| `--mobile SPECIES` | mobile species (required) |
| `--sites PATH` | explicit site definitions (YAML/JSON) |
| `--discover-sites-from-density` | derive sites from the density instead of an explicit file |
| `--resolution-A`, `--smoothing-sigma-A`, `--background-level` | density/site-discovery parameters |
| `--site-radius-A` | capture radius around a site |
| `--minimal-residence` | minimum residence time counted as a site visit |
| `--jump-dimensions {1,2,3}` | dimensionality for jump/percolation analysis |
| `--percolation-axes` | axes considered for percolation |
| `--drift-reference`, `--drift-mode`, `--drift-indices` | framework-drift correction |

**Inputs.** A production trajectory plus either explicit sites or density
discovery. Site definitions must match the structure/atom ordering; a
mismatch is a configuration error, not something to ignore.

**Outputs.** `analysis.json` with the site list, visit/jump counts, jump
vectors/residence statistics and percolation diagnostics; provenance as for
all analyses.

**Failure semantics.** GEMDAT backend errors are reported as
**backend failures** (`backend_error`), never as "zero jumps". Zero jumps is
only reported when the analysis actually ran and found none. Invalid/empty
site definitions or a mobile selection that matches no atoms fail closed.

**Scientific caveats.**

- Site mapping is a model of the transport mechanism: different site
  definitions can change jump counts. State the site source.
- `backend_error` ≠ "no diffusion"; check the error before interpreting a
  mechanism.
- Jump counts depend on `--minimal-residence`, drift policy and frame
  spacing; report them together.
- Percolation along chosen axes is not automatically 3D long-range
  transport; say which dimensions/axes were tested.

**Example.**

```bash
mliport analyze runs/md-800K electrolyte --mobile Li \
  --discover-sites-from-density --resolution-A 0.2 --minimal-residence 2 \
  --jump-dimensions 3 --percolation-axes xyz
```

## Required site input and budget

GEMDAT needs a site definition. Use either an explicit site file or the
automatic density-based discovery:

```bash
# tested example
.venv-grace/bin/mliport analyze results/lgps-1ns electrolyte \
  --mobile Li --discover-sites-from-density
```

Site discovery on a full 1 ns, 400-atom trajectory is expensive; the
acceptance suite runs it on a 30 ps segment with
`--frame-interval-fs 100`. Partial-occupancy warnings from GEMDAT are reported
by the upstream library and do not change the requested site set.
