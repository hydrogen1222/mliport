# mliport

mliport runs machine-learned interatomic potentials — MACE, DPA, GRACE and
UMA — from one command line, one terminal UI and one Python API. You hand it a
structure and a checkpoint; it relaxes the structure, runs molecular dynamics,
finds a reaction barrier with NEB, searches low-energy configurations of an
adsorbate on a slab, or analyzes a trajectory. Results come back
in files anyone who has used VASP can open: `OUTCAR`, `OSZICAR`, `CONTCAR` and
`XDATCAR`, next to JSON records of exactly what was run.

It is a workflow layer. It is not a model, not a trainer and not a DFT code:
the files are VASP-shaped, but there is no SCF, no k-point mesh and no POTCAR,
and an absolute energy only means something inside the one model that produced
it. Whether the potential is *right* for your chemistry is up to the
checkpoint you pick and the checks you run on it.

## What it does for you

- **One workflow, four backends.** Single point, relaxation (fixed or variable
  cell), NVE/NVT molecular dynamics, NEB and CI-NEB, configuration search,
  batch sweeps and a job queue behave the same whichever model sits underneath.
- **No silent guessing.** There is no default backend. A misspelled option, or
  one that belongs to another backend, stops the run instead of being ignored.
  A DPA checkpoint with several heads asks which one you mean.
- **A paper trail.** Every run records the model file and its hash, the
  resolved configuration (and where each value came from), the device that was
  really used and the mliport version.
- **Analysis that admits when the data is too short.** MSD, transport, RDF,
  VACF, GEMDAT electrolyte analysis and Arrhenius fits report an explicit
  "insufficient sampling" instead of a number you should not trust.

## Try it

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto

.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model --model-type mace --task bulk
```

The [quickstart](quickstart.md) walks through this line by line, including
where to get the checkpoint.

## Where to go next

| If you want to… | Read |
|---|---|
| install it and run a first calculation | [Quickstart](quickstart.md) |
| understand environments, Python versions and GPUs | [Installation](installation.md) |
| pick a model, or see which ones were validated | [Models](models.md), [Backends](backends/index.md) |
| set options through the CLI, an INCAR file or `settings.ini` | [Configuration](configuration.md) |
| run a relaxation, MD, NEB, a configuration search or a batch | [Workflows](workflows/index.md) |
| analyze a trajectory | [Analysis](analysis/index.md) |
| know what each output file contains | [Output files](outputs.md) |
| see what has actually been tested | [Validation](validation.md) |
| fix an error | [Troubleshooting](troubleshooting.md) |
| move over from the old `mlipx` name | [Migration](migration-from-mlipx.md) |
| cite mliport and the models it runs | [Citations](citations.md) |
| work on mliport itself | [Development](development.md) |
