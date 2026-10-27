# Test systems

Before you trust a model on your own problem, run it on a system whose answer is
known. This page lists a few classic cases, what each one tests, and the
reference value to compare with. They are suggestions, not part of mliport's
validation record: the numbers come from the cited literature or from the
repository's own [validation campaign](validation.md), and your result depends
on the checkpoint you use.

A foundation model reproduces its *reference functional* (mostly PBE / PBE+U,
see [Backends](backends/index.md)), not experiment. Compare with DFT at the same
level first; compare with experiment only to learn how far that level is from
reality.

## 1. Lattice constants (Si, Cu)

The cheapest sanity check, and the first step of every other recipe: a
variable-cell relaxation of a bulk crystal.

```bash
.venv-mace/bin/mliport opt Si.cif --model model.model --model-type mace \
  --task bulk --cell-opt --fmax 0.01 --device cuda --output runs/si-opt
```

Compare the relaxed lattice constant in `CONTCAR` with the PBE value of the same
structure from the [Materials Project](https://next-gen.materialsproject.org/)
(diamond Si is `mp-149`, fcc Cu is `mp-30`). A deviation of a few hundredths of
an Ångström is typical; a large one means the model is the wrong tool for this
chemistry. Use the model's own lattice constant for every later supercell, so
that you do not start from a strained cell.

## 2. A vacancy hop in Cu (NEB)

A vacancy in fcc Cu and one nearest neighbour jumping into it. The path is short
and symmetric, so it tests the NEB machinery more than the model.

```python
import numpy as np
from ase.build import bulk
from ase.geometry import get_distances
from ase.io import write

a = 3.63  # replace with the lattice constant from recipe 1
cell = bulk("Cu", "fcc", a=a, cubic=True).repeat((3, 3, 3))  # 108 atoms
site = cell.positions[0].copy()
initial = cell.copy()
del initial[0]  # vacancy at the origin
_, distances = get_distances(site, initial.positions, cell=initial.cell, pbc=True)
neighbour = int(np.argmin(distances[0]))  # a nearest neighbour of the vacancy
final = initial.copy()
final.positions[neighbour] = site  # it jumps into the vacant site
write("initial.vasp", initial)
write("final.vasp", final)
```

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --images 5 --climb --model model.model --model-type mace --task bulk \
  --device cuda --output runs/cu-vac-neb
```

The forward barrier should equal the reverse one (the path is symmetric). The
validation campaign found 0.735–0.762 eV with the four example checkpoints
([validation report](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md)).

## 3. Li diffusion in Li10GeP2S12 (MD, transport, Arrhenius)

The superionic conductor whose structure ships with the repository
(`examples/structures/li10gep2s12_primitive.vasp`). It exercises long MD runs and
the whole transport chain. Use a supercell; the 50-atom cell is too small for
diffusion.

```python
from ase.io import read, write

atoms = read("examples/structures/li10gep2s12_primitive.vasp")
write("LGPS_221.vasp", atoms.repeat((2, 2, 1)), direct=True)  # 200 atoms
```

```bash
for T in 600 700 800 900; do
  .venv-mace/bin/mliport md LGPS_221.vasp --model model.model --model-type mace \
    --task bulk --ensemble NVT --temp $T --timestep 1.0 \
    --equilibration-steps 10000 --steps 100000 --save-interval 100 \
    --device cuda --output runs/lgps-$T
  .venv-mace/bin/mliport analyze runs/lgps-$T transport --mobile Li --charge 1
done
```

Then fit the four diffusivities with `analyze ... arrhenius` (see
[Arrhenius](analysis/arrhenius.md)). For reference, ab initio MD (PBE, 600–1500 K,
40 ps) gave an activation energy of 0.21 eV and an extrapolated room-temperature
conductivity of about 9 mS/cm (range 2–40), against 0.24 eV and 12 mS/cm in
experiment ([Mo, Ong & Ceder, Chem. Mater. 2012](https://doi.org/10.1021/cm203303y)).
An MLIP can run much longer and larger cells than that study, so expect a
smaller error bar, not necessarily the same number.

## 4. CO on Pt(111): a model inherits its functional

At low coverage, CO adsorbs on top sites of Pt(111) in experiment, but converged
GGA calculations prefer the hollow site
([Feibelman et al., J. Phys. Chem. B 2001](https://doi.org/10.1021/jp002302t)). A
model trained on GGA data should reproduce the GGA answer. Relax CO in the top and
in the fcc-hollow site on a four-layer Pt(111) slab with the bottom two layers
fixed, and compare the two energies.

```bash
.venv/bin/mliport opt CO_top.vasp --model uma-s-1p2.pt --model-type uma \
  --task oc20 --fmax 0.02 --device cuda --output runs/co-top
.venv/bin/mliport opt CO_fcc.vasp --model uma-s-1p2.pt --model-type uma \
  --task oc20 --fmax 0.02 --device cuda --output runs/co-fcc
```

If the hollow site comes out lower, the model is doing its job: it is reproducing
GGA, error included. This is the clearest reminder that an MLIP cannot be more
right than its training data.

## 5. A small molecule on a slab (configuration search)

Methanol on a slab cut from Li10GeP2S12 is the system the
[configuration search](workflows/confsearch.md) validation script uses
(`validation/science/scripts/confsearch_suite.py`). Run it once before a large
molecule: it compares random placements relaxed without MD against the full
search with the same starts, and reports whether the MD cycles found anything
lower and how the deduplication behaved on real minima.

```bash
.venv-mace/bin/mliport confsearch run --slab slab.vasp --molecule methanol.xyz \
  --n-starts 10 --cycles 3 --dry-run \
  --model model.model --model-type mace --task bulk --output runs/meoh-dry
```

Drop `--dry-run` once the inputs and the budget look right.

## What to record

For every test, keep the model file and its SHA-256 (mliport records both), the
backend version, the task or head, the precision and the device. Two numbers are
only comparable when all of these match; see [Models](models.md).
