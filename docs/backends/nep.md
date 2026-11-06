# NEP (Calorine / native GPUMD)

NEP (neuroevolution potential) is a potential *family*, not one model. You
supply a `nep.txt` trained for your elements and conditions; mliport does not
ship, download or train one, and it cannot tell you whether a given `nep.txt`
is accurate for your material. Everything below assumes you have such a file.

Unlike the other four backends, NEP has **two execution runtimes**. You must
choose one:

| `NEP_RUNTIME` | Device | Implementation | Numeric precision | One GPU call = |
|---|---|---|---|---|
| `ase` | `cpu` | `calorine.calculators.CPUNEP` | float64 | one in-process evaluation |
| `ase` | `cuda[:N]` | `calorine.calculators.GPUNEP` | float32 | **one spawned `gpumd` process** |
| `gpumd` | `cuda[:N]` | native `gpumd` process | float32 | the whole run |

mliport never infers the runtime, never derives it from the device, and never
falls back from one to the other. They differ in precision, in cost model and
in which workflows they can perform, so a silent switch would make two runs
that look identically configured non-comparable.

`NEP_RUNTIME=gpumd` with `DEVICE=cpu` is rejected: there is no CPU native path
in this implementation, and downgrading the request to `CPUNEP` would change
the precision behind your back.

## Installation

<!-- BEGIN GENERATED: environment -->
| Environment | Python | Runtime stack |
|---|---|---|
| `.venv-nep` | 3.10, 3.11, 3.12, 3.13 | `calorine==4.1` (numpy 12.9) |

```bash
./scripts/install_mliport.sh --engines nep --device auto
```
<!-- END GENERATED -->

## Pinned profiles

<!-- BEGIN GENERATED: profiles -->
| Profile | Identity | dtype | task | head | model SHA-256 |
|---|---|---|---|---|---|
<!-- END GENERATED -->

NEP lives in its own environment because Calorine is a compiled extension:

```bash
uv venv .venv-nep --python 3.12
uv pip install --python .venv-nep/bin/python calorine ase
```

The native runtime additionally needs the `gpumd` executable, which is *not*
installed by the mliport installer (it needs a CUDA toolkit and a matching
`-arch=sm_XX` for your GPU):

```bash
git clone https://github.com/brucefan1983/GPUMD.git
cd GPUMD/src
source /path/to/cuda/env.sh          # provides nvcc + libcublas/libcusolver/libcufft
make CUDA_ARCH="-arch=sm_70" -j16    # sm_70 = V100; pick your own architecture
```

Then point mliport at it with `NEP_GPUMD_COMMAND` / `--gpumd-command`, or put it
on `PATH`. Verify both runtimes at once:

```bash
.venv-nep/bin/mliport doctor --engine nep --device cuda:0
```

`doctor --engine nep` reports the Calorine/ASE versions, the resolved `gpumd`
path and SHA-256, and the first lines of its version banner. It does **not**
load a model: a NEP run additionally needs your `nep.txt`.

## Configuration

```bash
.venv-nep/bin/mliport sp structure.vasp \
    --model-type nep \
    --model /path/to/nep.txt \
    --nep-runtime ase \
    --device cpu
```

```ini
MODEL_TYPE   = NEP
MODEL_PATH   = /path/to/nep.txt
NEP_RUNTIME  = gpumd          # ase | gpumd  (required)
NEP_GPUMD_COMMAND = /opt/gpumd/src/gpumd
DEVICE       = cuda:0
TASK         = bulk           # PBC semantic only; NEP has no task families
```

The same two keys reach every entry point: `--nep-runtime` / `--gpumd-command`
on the CLI, `nep_runtime=` / `nep_gpumd_command=` in the Python API,
`NEP_RUNTIME` / `NEP_GPUMD_COMMAND` in the INCAR and `settings.ini`, the
runtime selector in the TUI, and the same flags in queue task files. All of
them resolve to one normalised configuration object that reports the value
**and its source**; they cannot drift.

`MODEL_PATH` must be a single file. A directory is rejected: a folder of models
has no single identity.

## What each runtime can do

| Workflow | `NEP_RUNTIME=ase` | `NEP_RUNTIME=gpumd` |
|---|---|---|
| single point (`sp`) | yes | yes |
| fixed-cell optimisation (`opt`) | yes | yes, `minimize fire\|sd` |
| cell/box optimisation | only through the existing ASE contract | `minimize fire ... 1` |
| MD `NVE` | yes (ASE integrator) | yes (native, recommended) |
| MD `NVT` | yes (ASE integrators) | yes, thermostat-mapped (see below) |
| `NEB` / CI-NEB | yes | **rejected** |
| `batch`, `confsearch` | yes | **rejected** |

The native runtime refuses `neb`, `batch` and `confsearch` with an error that
names `NEP_RUNTIME=ase`. It does not reroute them. A NEB band driven by the
native process would need its own scheduler, and quietly switching runtimes
would mix two potentials' absolute energies inside one ranking.

DFT-D3 is rejected for NEP in both runtimes. A NEP model may already carry a
van der Waals term (`nep4-vdw`), and mliport has no evidence that adding D3 on
top avoids double-counting dispersion. The option is refused, not silently
dropped.

## Precision and what "the same numbers" means

`CPUNEP` computes in float64; `GPUNEP` and native `gpumd` compute in float32.
The `precision double` option in a GPUMD input file only widens the **text**
output — it does not make the inference double precision. Comparing the two
runtimes on one structure therefore needs a measured float32 noise floor, not a
bit-for-bit expectation.

Nothing in mliport's output claims otherwise: `model.manifest.json` /
`resolved_config.json` record `numeric_precision` (`float64` or `float32`),
`per_call_subprocess`, the `gpumd` path/SHA-256/version and the actual GPU UUID.

## Cost model — do not mix these up

`NEP_RUNTIME=ase` on a CUDA device runs Calorine `GPUNEP`, which writes inputs,
launches `gpumd` and reads outputs **on every ASE evaluation**. Every MD step or
optimiser step pays process startup and file I/O. mliport warns about this at
calculator construction and records `per_call_subprocess: true`.

Native `NEP_RUNTIME=gpumd` launches one process for the entire run. Any
steps-per-second comparison between the two is a statement about the adapter,
not about NEP. Never quote a `GPUNEP` ASE-MD throughput as native GPUMD
performance.

## Native MD: thermostats, velocities, and what is not ASE

GPUMD's `ensemble nvt_*` coupling constant is `T_coup = tau_T / dt`, a
dimensionless number, and mliport converts your ASE-style time parameters
exactly:

| mliport `THERMOSTAT` | GPUMD ensemble | `tau_T` used |
|---|---|---|
| `LANGEVIN` | `nvt_lan` | `1 / friction` (friction in 1/fs) |
| `BUSSI` | `nvt_bdp` | `bussi_tau` (fs) |
| `NHC` | `nvt_nhc` | `nhc_tdamp` (fs) |

`T_coup = round(tau_T / dt)` and must be `>= 1`; anything smaller is rejected
because GPUMD would silently reinterpret it. The chosen mapping, `tau_T` and
`T_coup` are recorded in the run manifest.

`velocity <T> seed <N>` is **not** a Maxwell–Boltzmann draw and it zeroes the
total linear *and* angular momentum. Do not treat a GPUMD-initialised NVT run as
equivalent to an ASE-Maxwell initialised one; the manifest records the method
and the seed so the difference stays traceable.

Velocities written into `model.xyz` are in Å/fs. ASE `momenta` are in
amu·Å/(ASE internal time unit), so the conversion is
`momentum / mass * ase.units.fs`; a literal `p/m` would be off by ~10x.

## Native output layout

```
<out>/native/model.xyz       extended XYZ: species, positions, lattice, pbc, velocities
<out>/native/run.in         the exact input file that was executed
<out>/native/thermo.out     raw GPUMD thermo (append semantics preserved)
<out>/native/traj.xyz       raw dump_xyz output, untouched
<out>/native/stdout.log
<out>/native/stderr.log
<out>/native/restart.xyz    only when dump_restart was requested
<out>/native/gpumd_provenance.json
<out>/trajectory.extxyz     normalised frames (step, time, cell, pbc, unwrapped)
<out>/model.manifest.json   terminal status + identity + frame accounting
```

Raw GPUMD files are written before any normalisation and never deleted, on
success or failure. A `gpumd` exit status of 0 with a missing or unparseable
required output is reported as `failed`, never `completed`.

### Units and signs

GPUMD writes `virial` (eV, nine row-major components) and a `stress` field that
is `virial / volume` in eV/Å³. ASE's stress convention is the opposite sign:

```text
ase_stress_voigt = -gpumd_stress_voigt        # xx yy zz yz xz xy, eV/Å³
GPa              = -ase_stress * 160.2176634
```

`thermo.out` columns are `T KE PE sxx syy szz syz sxz sxy ax..cz`, with the
stress components in **GPa**. Note the difference between the two files: the
`dump_xyz` `virial`/`stress` fields are the **potential** (virial) part only,
while `thermo.out` `sxx..sxy` is the **total** stress including the kinetic
contribution, so the two do not agree at finite temperature even after unit
conversion. mliport reports the potential-only stress (the quantity that is the
derivative of the potential energy, and the one comparable with the ASE
calculators of the other backends) and keeps `thermo.out` raw for inspection.

Both conventions are checked by a finite-difference known-answer test against
`CPUNEP`, not by trusting the header.

### Unwrapped positions and transport analysis

`dump_xyz ... unwrapped_position` gives the unfolded trajectory, which is what
tracer MSD, collective charge displacement and Haven ratios need. When that
column is unavailable, the manifest sets `msd_eligible: false` with a reason and
mliport refuses to derive bulk transport from sparse wrapped positions. Variable
cell runs have no single displacement reference and are not used for transport
in this release.

`restart.xyz` is an overwrite-only file with no RNG state; a single existing
file is not a reproducible MD checkpoint, and it is recorded as such.

## Known limits of this release

- No `qNEP` / `TNEP` / `nep4-vdw` support: those model types change the meaning
  of the energy (free energy, dipoles) and must not be used as a plain
  potential. They are outside this release's verified scope.
- Partial PBC (`T T F`) is rejected for the native runtime instead of being
  promoted to `T T T`.
- Variable-cell optimisation is exposed only through the ASE runtime contract.
- Native checkpoint/restart equivalence is not claimed.
- Accuracy of any particular `nep.txt` for a particular material is a model
  question, not a mliport question. `runtime_supported` is not
  `physically_validated_on_target`.
