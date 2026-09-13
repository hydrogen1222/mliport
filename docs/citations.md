# Citations and upstream credits

mliport is a **backend-neutral MLIP inference and workflow framework**. It
does not define or train potentials: it orchestrates upstream MLIP runtimes
(UMA/fairchem, MACE, DPA/DeePMD-kit, GRACE/tensorpotential), the ASE
ecosystem and analysis libraries (kinisi, GEMDAT) behind one CLI/TUI/Python
workflow. When you publish work done with mliport, cite the upstream
software and the specific model checkpoint you actually used.

## Citing mliport

mliport has **no DOI of its own yet**. Until one is registered:

- cite the repository `https://github.com/hydrogen1222/mliport` and the
  version you used (`2.0.0b1` at the time of writing), for example with the
  metadata in [`CITATION.cff`](../CITATION.cff);
- do **not** present an upstream DOI (for example the FAIRChem Zenodo DOI or
  a model paper DOI) as mliport's DOI;
- if a workflow publishes a DOI for mliport later, this file and
  `CITATION.cff` will be updated together.

## Upstream software and model citations

### ASE — Atomic Simulation Environment

mliport uses ASE for structures, calculators, optimizers, MD integrators and
file formats. Cite:

> A. H. Larsen et al., *The atomic simulation environment — a Python library
> for working with atoms*, J. Phys.: Condens. Matter **29**, 273002 (2017),
> doi:10.1088/1361-648X/aa680e.
> <https://wiki.fysik.dtu.dk/ase/>

### FAIRChem / UMA — `uma` backend

The `uma` backend loads UMA checkpoints through `fairchem-core` (Meta FAIR).
Cite the UMA model publication listed by the upstream project for the exact
checkpoint you used, and the fairchem software:

> <https://github.com/facebookresearch/fairchem> · <https://fair-chem.github.io/>

### MACE — `mace` backend

The `mace` backend loads MACE checkpoints through `mace-torch`. Cite the
MACE paper and the checkpoint's upstream release:

> I. Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
> Networks for Fast and Accurate Force Fields*, NeurIPS 2022,
> arXiv:2206.07697.
> <https://github.com/ACEsuit/mace>

### DeePMD-kit / DPA — `dpa` backend

The `dpa` backend loads DPA checkpoints through `deepmd-kit`. Cite:

> H. Wang, L. Zhang, J. Han, W. E, *DeePMD-kit: A deep learning package for
> many-body potential energy representation and molecular dynamics*,
> Comput. Phys. Commun. **228**, 178–184 (2018),
> doi:10.1016/j.cpc.2018.03.016.
> <https://github.com/deepmodeling/deepmd-kit>

For DPA-2/DPA-3 model checkpoints, cite the technical report referenced by
the upstream DeePMD project for that checkpoint.

### GRACE — `grace` backend

The `grace` backend loads GRACE exported SavedModels through
`tensorpotential`. Cite the GRACE publication referenced by the upstream
project for the checkpoint you used:

> <https://github.com/ICAMS/grace-tensorpotential>

### kinisi — transport analysis

`mliport analyze transport` uses kinisi for covariance-aware Bayesian
tracer/collective transport. Cite the kinisi paper referenced by the
upstream project:

> <https://github.com/bjmorgan/kinisi> · <https://kinisi.rtfd.io>

### GEMDAT — mechanism analysis

`mliport analyze electrolyte` uses GEMDAT for site occupancy and jump
analysis. Cite the GEMDAT paper referenced by the upstream project:

> <https://github.com/GEMDAT-repos/GEMDAT> · <https://gemdat.readthedocs.io>

## Data and benchmark provenance

- The OMat24 validation subset used by the T3 accuracy workload is derived
  from the public OMat24 release; its pinned provenance is recorded in
  `validation/science/data_manifest.json`.
- Model artifacts are referenced by upstream model identity plus SHA-256 in
  `validation/science/model_manifest.json`; model weights are not
  redistributed by mliport.

## Reporting missing or stale citations

If an upstream project, model card or license notice is missing or out of
date here, please open an issue. Upstream copyright and license notices are
never removed from this repository.
