# Molecular dynamics

`mliport md` integrates the classical equations of motion to generate a
trajectory — a time series of atomic positions and velocities that samples a
thermodynamic ensemble. The trajectory is the raw material for all of
mliport's [analysis](../analysis/index.md) tools: diffusion coefficients,
radial distribution functions, vibrational spectra and more.

## The physics

Molecular dynamics solves Newton's equations of motion:

$$m_i \ddot{\mathbf{r}}_i = \mathbf{F}_i = -\nabla_i E$$

where the forces $\mathbf{F}_i$ come from the MLIP model at each timestep.
The trajectory is integrated with velocity Verlet, and a thermostat can be
coupled to control the temperature.

### Ensembles

- **NVE** (microcanonical): constant number of atoms, volume and total
  energy. No thermostat. Used for benchmarking energy conservation or when
  you need unperturbed dynamics (e.g. for vibrational spectra via VACF).
- **NVT** (canonical): constant temperature via a thermostat. The most common
  choice for equilibrium sampling and transport-coefficient extraction.
- **NPT** (isothermal–isobaric): constant temperature and pressure via a
  coupled thermostat and barostat. Used for equilibrating the cell volume at a
  target pressure before switching to NVT for production.

## Command line

```bash
mliport md structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --ensemble NVT --temp 300 --timestep 1.0 --steps 1000 \
  --equilibration-steps 500 --save-interval 10 \
  --output runs/md
```

### A production run at 800 K with NHC thermostat

```bash
mliport md structure.cif --model uma.pt --model-type uma --task omat \
  --ensemble NVT --temp 800 --timestep 1.0 \
  --equilibration-steps 5000 --steps 50000 --save-interval 10 \
  --thermostat NHC \
  --output runs/md-800K
```

### NPT cell equilibration

```bash
mliport md structure.cif --model model.pt --model-type mace \
  --ensemble NPT --temp 800 --pressure-gpa 0 \
  --npt-tdamp 100 --npt-pdamp 1000 \
  --timestep 0.5 --steps 50000 --save-interval 10 \
  --no-pre-relax --output runs/npt-800K
```

NPT uses ASE's **isotropic Martyna–Tobias–Klein (MTK)** thermostat/barostat
for the UMA, MACE, GRACE and DPA backends. NEP/GPUMD is not supported for
NPT. See the [NPT details](#npt-details) section below.

## Python API

```python
from mliport.api import run_md

# NVT production run
results = run_md(
    structure="structure.cif",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    ensemble="NVT",
    temperature=800,
    timestep=1.0,
    steps=50000,
    thermostat="NHC",
    save_interval=10,
    output_dir="runs/md-800K",
)
print(f"Final temperature: {results['temperature']:.1f} K")
```

The function accepts an ASE `Atoms` object or a file path. It returns a
dictionary with temperature, energy, and trajectory metadata.

!!! note "NPT arguments are keyword-only"
    When calling `run_md` for NPT, pass NPT parameters as keyword arguments:
    `pressure_gpa=0.0`, `npt_tdamp=100.0`, etc. They cannot be passed
    positionally. This prevents the historical positional-argument ambiguity
    with `MDRunner` (Issue #102).

## Options

### Dynamics

| Option | Default | Meaning |
|---|---|---|
| `--ensemble` | `NVT` | `NVE`, `NVT`, or `NPT` |
| `--temp` | 300 K | Target temperature |
| `--timestep` | 1.0 fs | Integration timestep |
| `--steps` | 1000 | Production steps |
| `--equilibration-steps` | 0 | Equilibration steps (same ensemble, before production) |
| `--save-interval` | 10 | Write a trajectory frame every N steps |
| `--seed` | random | Random seed; generated and recorded when omitted |

### Thermostats (NVT)

| Option | Default | Meaning |
|---|---|---|
| `--thermostat` | `LANGEVIN` | `LANGEVIN`, `BUSSI` or `NHC` |
| `--friction` | — | Langevin friction coefficient (1/fs) |
| `--bussi-tau` | — | Bussi/CSVR coupling time (fs) |
| `--nhc-tdamp` | — | Nosé–Hoover chain damping time (fs) |
| `--nhc-tchain` | — | NHC chain length |
| `--nhc-tloop` | — | NHC integration substeps |

### NPT parameters

| Option | Default | Meaning |
|---|---|---|
| `--pressure-gpa` | 0 | External hydrostatic pressure in GPa (positive = compressive) |
| `--npt-tdamp` | — | MTK thermostat damping time (fs) |
| `--npt-pdamp` | — | MTK barostat damping time (fs) |
| `--npt-tchain` / `--npt-pchain` | 3 | MTK chain lengths |
| `--npt-tloop` / `--npt-ploop` | 1 | MTK integration substeps |

### Pre-relaxation

| Option | Default | Meaning |
|---|---|---|
| `--pre-relax` | on for NVT, off for NVE/NPT | Positions-only relaxation before MD |
| `--pre-relax-steps` | 50 | Maximum pre-relaxation steps |
| `--pre-relax-fmax` | 0.1 eV/Å | Pre-relaxation force threshold |

### Safety and output

| Option | Default | Meaning |
|---|---|---|
| `--fmax-abort` | 20 eV/Å | Abort when a force exceeds this |
| `--velocity-policy` | — | How initial velocities are handled |
| `--com-policy` | `auto` | Center-of-mass motion handling |
| `--write-outcar` | true | Write VASP-style OUTCAR |
| `--write-xdatcar` | true | Write VASP-style XDATCAR (NVT/NVE only) |
| `--write-trajectory` | true | Write ASE `.traj` trajectory |

## Pre-relaxation: a gate, not a suggestion

If the pre-relaxation does not reach `--pre-relax-fmax` within
`--pre-relax-steps`, the run **stops before production MD begins**. Starting
dynamics from a structure with large forces is rarely what anyone wants, and a
silent "good enough" would hide it. The error tells you what to do: increase
the step budget, loosen the threshold if you can justify it scientifically, or
turn pre-relaxation off with `--no-pre-relax`. The outcome of a successful
pre-relaxation is recorded in `md_provenance.pre_relax`.

## Thermostats: which one and why it matters

The thermostat controls temperature in NVT simulations. The choice affects
the dynamics and therefore the transport coefficients you extract.

- **Langevin** (default): stochastic thermostat that adds friction and random
  forces. Straightforward and stable. The friction parameter couples how
  strongly the thermostat acts. *Caveat*: Langevin dynamics are not
  Hamiltonian, so velocity autocorrelation functions and vibrational spectra
  computed from Langevin trajectories carry thermostat artifacts. For spectra,
  use NVE or NHC.
- **Bussi/CSVR**: deterministic thermostat that rescales velocities to match
  the canonical distribution (Bussi, Donadio and Parrinello, J. Chem. Phys.
  **126**, 014101, 2007). A good middle ground: minimal perturbation to the
  dynamics while enforcing the correct temperature distribution.
- **NHC** (Nosé–Hoover chain): deterministic thermostat based on extended
  system dynamics (Martyna et al., J. Chem. Phys. **97**, 2635, 1992).
  Produces true canonical sampling with unperturbed dynamics. Best choice for
  vibrational spectra and diffusion coefficients. **Cannot coexist with
  FixCom or other ASE constraints.** mliport checks for constraints *before*
  loading the model — if the input structure carries any constraint, the run
  fails immediately with an actionable error rather than loading the model and
  then crashing.

**Always state the thermostat when reporting a diffusion coefficient.** Two
simulations with different thermostats at the same temperature are different
dynamical systems.

### NHC and center-of-mass motion

NHC cannot run with a `FixCom` constraint. With the default `--com-policy
auto` (same as `initialize_only`), mliport zeroes the center-of-mass momentum
once after initializing velocities, and NHC then conserves it. The run
therefore has $3N-3$ kinetic degrees of freedom, and the thermostat set point
is scaled by $(3N-3)/(3N)$ (recorded as `nhc_set_temperature_K`). With
`--com-policy none`, the random COM momentum stays, the cell drifts
ballistically, and that drift dominates the MSD unless the analysis removes it.

## Timestep, save interval and what analysis sees

`--timestep` is the integration step size; `--save-interval` is how often a
frame is saved. With 1 fs timestep and `--save-interval 10`, you get a frame
every 10 fs. It is this **saved spacing** — not the integration step — that
transport analysis needs to know.

Keep equilibration out of your statistics: use `--equilibration-steps` and
analyze only the production segment. Keep ensembles apart — NVE and NVT are
different dynamics and their trajectories should never be averaged together.

## NPT details

NPT runs an isotropic Martyna–Tobias–Klein (MTK) thermostat/barostat. It is
available on UMA, MACE, GRACE and DPA backends (not NEP/GPUMD).

**Requirements:**

- 3D periodic boundary conditions
- A finite positive-volume cell
- A model that provides **stress**
- No atomic/COM constraints, no partial PBC

**Pressure convention:** positive pressure is compressive (GPa). The
conversion to ASE's internal eV/Å³ is recorded.

**XDATCAR and analysis:** Standard XDATCAR represents fixed-cell MD only, so
NPT disables XDATCAR (with a logged reason). The trajectory is preserved in
`raw/trajectory.traj` with each frame's cell and momenta. The `md.csv` file
records instantaneous total pressure (including kinetic term) and volume per
saved frame.

**Transport analysis caveat:** NPT trajectories are marked
variable-cell / transport-ineligible. Do **not** calculate conventional
fixed-cell MSD, Einstein–Helfand conductivity or diffusion from NPT data.
For solid electrolytes, the recommended workflow is:

1. Equilibrate volume/cell at target T and pressure with NPT
2. Extract the equilibrated-window mean cell with an independent statistical
   check (the last NPT frame is **not** automatically an average cell)
3. Start a **separate** fixed-cell NVT or NVE production run on the
   equilibrated cell

**Frame restarts:** NPT frame restarts do not preserve hidden MTK
chain/barostat state and are not exact trajectory continuations.

## Continuing a run

MD has no `--resume` flag. To continue a trajectory, read a saved frame that
carries momenta and start a **new** run:

```bash
mliport md last_frame.traj --model mace-omat-0-medium.model \
  --model-type mace --velocity-policy preserve --no-pre-relax \
  --output runs/md-part2
```

The new run keeps the supplied positions and velocities without drawing a
fresh Maxwell–Boltzmann distribution. It is a separate run directory with its
own record, and its first frame equals the last frame of the previous leg.

## What can stop a run

- A force above `--fmax-abort` or a NaN aborts immediately rather than writing
  corrupt frames.
- A failure while writing output aborts instead of silently dropping frames.
- A missing model, structure, or unsupported option combination fails before
  the dynamics start.

## Output files

| File | Location | Contents |
|---|---|---|
| `trajectory.traj` | `raw/` | Full ASE trajectory with positions, velocities, cell |
| `md.csv` | `raw/` | Per-frame energy, temperature, pressure, volume |
| `OUTCAR`, `OSZICAR` | `vasp/` | VASP-compatible energy and force records |
| `XDATCAR` | `vasp/` | VASP-compatible trajectory (NVT/NVE only, fixed cell) |
| `mliport_results.json` | `.` | Summary: temperature, energy, timing |
| `resolved_config.json` | `.` | Complete resolved configuration |
| `run_context.json` | `.` | Model identity and provenance |
| `artifacts.json` | `.` | File listing with units and hashes |
| `run.log` | `.` | Human-readable log |

See [Output files](../outputs.md) for the conventions shared by all workflows.
