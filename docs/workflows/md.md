# Molecular dynamics

`mliport md` runs NVE, NVT or NPT dynamics with the backend you pick and records a
trajectory that the [analysis](../analysis/index.md) tools can read back.

```bash
.venv-mace/bin/mliport md structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --ensemble NVT --temp 300 --timestep 1.0 --steps 1000 \
  --equilibration-steps 500 --save-interval 10 \
  --output runs/md
```

You need a periodic structure, a model and an explicit backend. MD on a
molecule needs a molecular task (`omol` / `molecule`) and sensible charge and
spin.

## Isotropic NPT cell equilibration (ASE-only)

For UMA/MACE/GRACE/DPA, `--ensemble NPT` uses ASE's **isotropic Martyna–Tobias–Klein (MTK)** thermostat/barostat, not Langevin/Bussi/NHC. Example:

```bash
mliport md structure.cif --model model.pt --model-type mace \
  --ensemble NPT --temp 800 --pressure-gpa 0 \
  --npt-tdamp 100 --npt-pdamp 1000 \
  --timestep 0.5 --steps 50000 --save-interval 10 \
  --no-pre-relax --output runs/npt-800K
```

The parameters `--npt-tchain`/`--npt-pchain` default to 3 and `--npt-tloop`/`--npt-ploop` to 1. Times are in fs. **Positive pressure is compressive (GPa)**; the conversion to ASE eV/Å³ is recorded. Thermostat and barostat damping are not equilibration time guarantees. The NPT run uses 3D PBC, a finite positive-volume cell, and model **stress**. It rejects all atomic/COM constraints, partial PBC, zero T, and unsupported backends (including NEP/GPUMD). Automatic COM handling zeroes initial COM momentum once; the recorded MTK thermostat set point accounts for the 3 missing kinetic degrees of freedom.

Standard XDATCAR represents fixed-cell MD only: NPT disables XDATCAR with a logged reason, while `raw/trajectory.traj` preserves each frame's cell and momenta. The `md.csv` file records instantaneous **total** pressure including the kinetic term, and volume per saved frame. NPT trajectory metadata is marked variable-cell / transport-ineligible: do **not** calculate conventional fixed-cell MSD, Einstein–Helfand conductivity or diffusion from it. For solid electrolytes, equilibrate volume/cell at the target T and pressure, extract the *equilibrated-window mean cell* with an independent statistical check, then start a **separate** fixed-cell NVT/NVE run. The last NPT frame is not automatically an average cell. NPT frame restarts do not preserve hidden MTK chain/barostat state and are not exact trajectory continuation. Near a phase transition or an anisotropic crystal, an isotropic cell-only barostat may be insufficient.

## Options you will use

| Option | Meaning |
|---|---|
| `--ensemble` | `NVT` (default) or `NVE` |
| `--temp` | target temperature in K (default 300) |
| `--timestep` | integration step in fs (default 1.0) |
| `--steps` | production steps (default 1000) |
| `--equilibration-steps` | equilibration in the same ensemble before production (default 0) |
| `--thermostat` | `LANGEVIN` (default), `BUSSI` or `NHC` |
| `--friction`, `--bussi-tau`, `--nhc-tdamp`, `--nhc-tchain`, `--nhc-tloop` | thermostat parameters |
| `--save-interval` | write a frame every N steps (default 10) |
| `--seed` | random seed; generated and recorded when omitted |
| `--velocity-policy`, `--com-policy` | how velocities and the center-of-mass motion are handled |
| `--pre-relax`, `--pre-relax-steps`, `--pre-relax-fmax` | a short positions-only relaxation before MD (default on for NVT, off for NVE; 50 steps, 0.1 eV/Å) |
| `--fmax-abort` | abort when a force exceeds this many eV/Å (default 20) |
| `--write-outcar`, `--write-xdatcar`, `--write-trajectory` | which outputs to write |

### Pre-relaxation is a gate, not a suggestion

If the pre-relaxation does not reach `--pre-relax-fmax` within
`--pre-relax-steps`, the run **stops before production MD**. Starting dynamics
from a structure that is still far from a minimum is rarely what anyone wants,
and a silent "good enough" would hide it. The error tells you what to do: raise
the step budget, loosen the threshold if you can justify it scientifically, or
turn pre-relaxation off explicitly. The outcome of a pre-relaxation that did
finish is recorded in `md_provenance.pre_relax`.

## Timestep, save interval and what analysis sees

`--timestep` is how the equations of motion are integrated; `--save-interval`
is how often a frame is written. With a 1 fs timestep and `--save-interval 10`
you get a frame every 10 fs, and it is this saved spacing, not the integration
step, that transport analysis needs.

Keep equilibration out of your statistics: use `--equilibration-steps` and
analyze only the production segment. Keep the ensembles apart, too. NVE and NVT
are different dynamics and their trajectories should never be averaged together.

## Thermostats and center-of-mass motion

`--thermostat` chooses how NVT temperature is controlled. Langevin is
stochastic; Bussi and NHC are deterministic. Transport coefficients depend on the
choice, so state it when you report a diffusion coefficient.

NHC deserves a note. It cannot run together with a FixCom (or any other)
constraint. With the default `--com-policy auto` (the same as `initialize_only`)
mliport zeroes the center-of-mass momentum once after initializing velocities,
which NHC then conserves. The run therefore has `3N-3` kinetic degrees of
freedom, and the thermostat set point is scaled by `(3N-3)/3N` (recorded as
`nhc_set_temperature_K`). With `--com-policy none` the random center-of-mass
momentum stays, the whole cell drifts ballistically, and that drift dominates the
MSD of the mobile ions unless the analysis removes it.

Constraints and fixed atoms are applied through the calculator and runners;
state them when you report diffusion.

## Continuing a run

`md` has no `--resume` flag. To continue, read a saved frame that carries
momenta and start a **new** run from it:

```bash
.venv-mace/bin/mliport md last_frame.traj --model mace-omat-0-medium.model \
  --model-type mace --velocity-policy preserve --no-pre-relax --output runs/md-part2
```

The new run keeps the supplied positions and velocities and does not draw a
fresh Maxwell–Boltzmann distribution. It is a separate run directory with its own
record, and its first frame equals the last frame of the previous leg (the
acceptance suite checks exactly that).

## What can stop a run

A force above `--fmax-abort` or a NaN aborts the run rather than writing corrupt
frames. A failure while writing output aborts it too, instead of silently
dropping frames. A missing model or structure, or an option combination that is
not supported, fails before the dynamics start.

The acceptance matrix runs 5-step NVE and 5-step Langevin, Bussi and NHC smoke
tests in all four backend environments. Those runs check the workflow and the
thermostat plumbing, not that the temperature has equilibrated.

## A production-style example

```bash
.venv/bin/mliport md structure.cif --model uma.pt --model-type uma --task omat \
  --ensemble NVT --temp 800 --timestep 1.0 \
  --equilibration-steps 5000 --steps 50000 --save-interval 10 \
  --thermostat NHC \
  --output runs/md-800K
```

The files this produces are described in [Output files](../outputs.md): the
canonical trajectory and `md.csv` under `raw/`, the VASP-style files under
`vasp/`, and `artifacts.json` listing them with units and hashes.
