# Citations and credits

mliport is a workflow layer. It does not define or train potentials; it drives
other people's software (UMA/fairchem, MACE, DPA/DeePMD-kit, GRACE/tensorpotential,
ASE, kinisi, GEMDAT) behind one CLI, TUI and Python API. So when you publish
something computed with it, the people to credit are mostly upstream: the software
and, above all, the exact model checkpoint you used.

## Citing mliport

mliport has **no DOI of its own yet**. Until it does:

- cite the repository, `https://github.com/hydrogen1222/mliport`, and the version
  you ran (`mliport --version`). The metadata in
  [`CITATION.cff`](https://github.com/hydrogen1222/mliport/blob/main/CITATION.cff)
  is ready for reference managers;
- do not present an upstream DOI (the FAIRChem Zenodo DOI, a model paper's DOI) as
  mliport's;
- if mliport gets a DOI later, this page and `CITATION.cff` will change together.

## What to cite upstream

### ASE

mliport uses ASE for structures, calculators, optimizers, MD integrators and file
formats.

> A. H. Larsen et al., *The atomic simulation environment — a Python library for
> working with atoms*, J. Phys.: Condens. Matter **29**, 273002 (2017),
> doi:10.1088/1361-648X/aa680e. <https://wiki.fysik.dtu.dk/ase/>

### FAIRChem and UMA (`uma` backend)

The `uma` backend loads UMA checkpoints through `fairchem-core` from Meta FAIR. Cite
the UMA publication that the upstream project lists for the checkpoint you used,
together with the fairchem software.

> <https://github.com/facebookresearch/fairchem> · <https://fair-chem.github.io/>

### MACE (`mace` backend)

The `mace` backend loads checkpoints through `mace-torch`. Cite the MACE paper and
the release of the checkpoint you used.

> I. Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
> Networks for Fast and Accurate Force Fields*, NeurIPS 2022, arXiv:2206.07697.
> <https://github.com/ACEsuit/mace>

### DeePMD-kit and DPA (`dpa` backend)

The `dpa` backend loads checkpoints through `deepmd-kit`.

> H. Wang, L. Zhang, J. Han, W. E, *DeePMD-kit: A deep learning package for
> many-body potential energy representation and molecular dynamics*, Comput.
> Phys. Commun. **228**, 178–184 (2018), doi:10.1016/j.cpc.2018.03.016.
> <https://github.com/deepmodeling/deepmd-kit>

For a DPA-2 or DPA-3 checkpoint, also cite the technical report the DeePMD project
references for it.

### GRACE (`grace` backend)

The `grace` backend loads exported SavedModels through `tensorpotential`. Cite the
GRACE publication the upstream project gives for your checkpoint.

> <https://github.com/ICAMS/grace-tensorpotential>

### kinisi

`mliport analyze transport` uses kinisi for covariance-aware Bayesian tracer and
collective transport. Cite the kinisi paper the upstream project references.

> <https://github.com/bjmorgan/kinisi> · <https://kinisi.rtfd.io>

### GEMDAT

`mliport analyze electrolyte` uses GEMDAT for site occupancy and jump analysis.
Cite the GEMDAT paper the upstream project references.

> <https://github.com/GEMDAT-repos/GEMDAT> · <https://gemdat.readthedocs.io>

## Data and model provenance

- The OMat24 subset used by the accuracy workload comes from the public OMat24
  release; its pinned provenance is in `validation/science/data_manifest.json`.
- Models are identified by their upstream identity and SHA-256 in
  `validation/science/model_manifest.json`. mliport does not redistribute weights.

## If something is missing

If an upstream project, model card or licence notice is absent or out of date here,
please open an issue. Upstream copyright and licence notices are never removed from
the repository.
