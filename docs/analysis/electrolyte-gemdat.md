# GEMDAT electrolyte analysis

`electrolyte` uses GEMDAT to map the mobile ion's positions onto sites, count
site-to-site jumps, and look for percolating pathways. It is about *mechanism*:
where the ion sits and how it hops, as opposed to how fast it diffuses overall.

GEMDAT needs a definition of the sites, from exactly one of two sources: your own
site file, or sites discovered from the density of the mobile species.

```bash
mliport analyze runs/md electrolyte --mobile Li --sites sites.yaml --temperature-K 800
mliport analyze runs/md electrolyte --mobile Li --discover-sites-from-density
```

A site definition has to match the structure and its atom order; a mismatch is a
configuration error, not something to ignore.

## Options you will use

| Option | Meaning |
|---|---|
| `--mobile SPECIES` | the mobile species (required) |
| `--sites PATH` or `--discover-sites-from-density` | an explicit site file, or density-based discovery (exactly one is required) |
| `--resolution-A`, `--background-level` | density and site-discovery parameters (defaults 0.5 Å, 0.1) |
| `--site-radius-A` | capture radius around a site |
| `--minimal-residence` | minimum residence, in frames, to count as a site visit (default 0) |
| `--jump-dimensions` | dimensionality of the jump analysis: 1, 2 or 3 (default 3) |
| `--percolation-axes` | axes tested for percolation (default `xyz`) |
| `--temperature-K` | temperature for derived quantities |
| `--drift-reference`, `--drift-mode`, `--drift-indices` | framework-drift correction (mode default `arithmetic_mean`) |

## What comes out

`results.json` has the site list, visit and jump counts, jump vectors and
residence statistics, and the percolation diagnostics, with the provenance that
every analysis carries.

If GEMDAT itself fails, that is reported as a **backend failure**
(`backend_error`), never as "zero jumps". Zero jumps is reported only when the
analysis ran and found none. Empty or invalid site definitions, or a mobile
selection that matches no atoms, fail closed.

## Reading the result

- A site mapping is a model of the transport mechanism. Different site
  definitions can change the jump counts, so state where your sites came from.
- `backend_error` does not mean "no diffusion". Read the error before you
  interpret any mechanism.
- Jump counts depend on `--minimal-residence`, the drift policy and the frame
  spacing; report them together.
- Percolation along chosen axes is not automatically 3-D long-range transport.
  Say which dimensions and axes were tested.

```bash
mliport analyze runs/md-800K electrolyte --mobile Li \
  --discover-sites-from-density --resolution-A 0.2 --minimal-residence 2 \
  --jump-dimensions 3 --percolation-axes xyz
```

## Cost

Discovering sites on a full 1 ns, 400-atom trajectory is expensive. The
acceptance suite runs it on a 30 ps segment with `--frame-interval-fs 100`:

```bash
.venv-grace/bin/mliport analyze results/lgps-1ns electrolyte \
  --mobile Li --discover-sites-from-density
```

GEMDAT's partial-occupancy warnings come from the upstream library and do not
change the requested site set.
