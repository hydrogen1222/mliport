# INCAR key reference

Every key mliport accepts in an `INCAR.mliport` file, what it means, its default
and the values it accepts. The tables are generated from the option table in
`mliport/config/schema.py` - the same table the templates and `mliport config
schema` come from - so this page cannot drift from the code.

Three things apply everywhere on this page. A key is written as
`KEY = VALUE`, one assignment per line, `#` starts a comment, and the file name
is `INCAR.mliport` (mliport never reads a file called plain `INCAR`, which may
be a real VASP input). `REQUIRED` is a placeholder: the template cannot invent
a model checkpoint or a start structure, so you fill it in. Precedence is CLI,
then a profile, then a model alias, then the INCAR, then `settings.ini`, then the
built-in default; see [Configuration](configuration.md) for the layering and the
task-folder usage.

Units are the ones named in each row: femtoseconds (fs), picoseconds (ps),
Angstrom, electronvolt (eV), Kelvin and gigapascal (GPa). Energies are MLIP
total energies and are only comparable within one model identity.

## Keys common to every calculation type

<!-- BEGIN GENERATED: incar-common -->
**Calculation**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**Model**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Path to the model checkpoint/file. | - | - |
| `TASK` | str | - | - | UMA task (omat/omol/...) or bulk/molecule for other engines. | - | - |
| `INFERENCE_MODE` | str | - | `default`, `turbo` | UMA inference mode. Non-default modes are rejected for other engines. | - | - |
| `DEFAULT_DTYPE` | str | - | `float32`, `float64` | MACE model dtype (accuracy-first default: float64). | - | - |
| `HEAD` | str | - | - | MACE or DeepMD model head/branch (multi-task models). | - | - |
| `ACTIVATION_CHECKPOINTING` | bool | `-` | - | GPU memory saving (UMA, overrides inference_mode preset). | - | - |
| `GPU_MEMORY_GROWTH` | bool | - | - | GRACE TensorFlow allocator grows on demand instead of claiming the whole visible GPU. | - | - |
| `GPU_MEMORY_LIMIT_MB` | int | - | `1..` | Hard GRACE TensorFlow logical-device memory limit in MiB. | - | - |
| `NEIGHBOR_CACHE` | bool | - | - | GRACE verlet-style neighbor-list cache (extended cutoff + exact re-filter). The complete periodic-image multiset matches a fresh search; pair ordering may differ. Default: True. | - | - |
| `NEIGHBOR_SKIN` | float | - | `1e-12..` | GRACE neighbor-cache skin in Å: the cached neighbor list is rebuilt when any atom moves more than skin/2. Larger values rebuild less often but filter longer lists. | - | - |
| `D3` | bool | - | - | Add Grimme DFT-D3 dispersion (torch-dftd) on top of the model. Only for models trained on dispersion-free reference data. | - | - |
| `D3_XC` | str | - | - | Reference functional of the model's training data, selecting the D3 parameters (default: pbe). | - | - |
| `D3_DAMPING` | str | - | `zero`, `bj`, `zerom`, `bjm` | DFT-D3 damping function (default: bj, i.e. D3(BJ)). | - | - |
| `DPA_RUNTIME` | str | - | `current`, `legacy` | DPA runtime environment: 'current' (pinned DPA-4-capable deepmd-kit) or 'legacy' (.venv-dpa-legacy, deepmd-kit 3.0.0b3, for DPA-2 checkpoints such as DPA-SSE). No automatic fallback. | - | - |
| `NEP_RUNTIME` | str | - | `ase`, `gpumd` | NEP execution runtime: 'ase' (Calorine CPUNEP/GPUNEP calculator) or 'gpumd' (native gpumd process). Required for MODEL_TYPE=nep; mliport never infers it or falls back between the two. | - | - |
| `NEP_GPUMD_COMMAND` | str | - | - | Path to the gpumd executable for NEP_RUNTIME=gpumd (and for the GPUNEP ASE adapter). Launched as an argv list, never through a shell; its real path, version and SHA256 are recorded. | - | - |

**Device and resources**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**Molecular dynamics**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**Output**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `WRITE_RESOLVED_CONFIG` | bool | `.TRUE.` | - | Write resolved_config.json before a calculation. | - | - |
| `WRITE_FORCES` | bool | `.TRUE.` | - | Write forces to OUTCAR. | - | - |
| `WRITE_OUTCAR` | bool | - | - | Write the VASP-like OUTCAR interoperability view. | - | - |
| `WRITE_XDATCAR` | bool | - | - | Write the VASP XDATCAR interoperability trajectory. | - | - |
| `WRITE_STRESS` | bool | `.TRUE.` | - | Write stress to OUTCAR. | - | - |
| `WRITE_TRAJECTORY` | bool | - | - | Write ASE trajectory. | - | - |
| `WRITE_JSON` | bool | - | - | Write JSON results. | - | - |
| `OUTPUT_FORMAT` | str | `VASP` | - | Output format (VASP). | - | - |
| `OUTPUT_DIR` | str | - | - | Output directory for the calculation. Used when the command line does not pass --output; the resolved config is written into the final job directory. Relative values resolve against the file that declares them (the INCAR folder for an INCAR), and `mliport run` defaults to <INCAR folder>/results when neither --output nor OUTPUT_DIR is given. | - | - |
| `JOB_NAME` | str | - | - | Optional job name. | - | - |

**General**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## Single point

`CALC_TYPE = SP` evaluates energy, forces and stress of one structure. Output
control keys apply here as they do for optimization and MD.

<!-- BEGIN GENERATED: incar-sp -->
**Calculation**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**Model**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Path to the model checkpoint/file. | - | - |
| `TASK` | str | - | - | UMA task (omat/omol/...) or bulk/molecule for other engines. | - | - |
| `INFERENCE_MODE` | str | `default` | `default`, `turbo` | UMA inference mode. Non-default modes are rejected for other engines. | - | - |
| `DEFAULT_DTYPE` | str | - | `float32`, `float64` | MACE model dtype (accuracy-first default: float64). | - | - |
| `HEAD` | str | - | - | MACE or DeepMD model head/branch (multi-task models). | - | - |
| `ACTIVATION_CHECKPOINTING` | bool | `-` | - | GPU memory saving (UMA, overrides inference_mode preset). | - | - |
| `GPU_MEMORY_GROWTH` | bool | - | - | GRACE TensorFlow allocator grows on demand instead of claiming the whole visible GPU. | - | - |
| `GPU_MEMORY_LIMIT_MB` | int | - | `1..` | Hard GRACE TensorFlow logical-device memory limit in MiB. | - | - |
| `NEIGHBOR_CACHE` | bool | - | - | GRACE verlet-style neighbor-list cache (extended cutoff + exact re-filter). The complete periodic-image multiset matches a fresh search; pair ordering may differ. Default: True. | - | - |
| `NEIGHBOR_SKIN` | float | - | `1e-12..` | GRACE neighbor-cache skin in Å: the cached neighbor list is rebuilt when any atom moves more than skin/2. Larger values rebuild less often but filter longer lists. | - | - |
| `D3` | bool | - | - | Add Grimme DFT-D3 dispersion (torch-dftd) on top of the model. Only for models trained on dispersion-free reference data. | - | - |
| `D3_XC` | str | - | - | Reference functional of the model's training data, selecting the D3 parameters (default: pbe). | - | - |
| `D3_DAMPING` | str | - | `zero`, `bj`, `zerom`, `bjm` | DFT-D3 damping function (default: bj, i.e. D3(BJ)). | - | - |
| `DPA_RUNTIME` | str | - | `current`, `legacy` | DPA runtime environment: 'current' (pinned DPA-4-capable deepmd-kit) or 'legacy' (.venv-dpa-legacy, deepmd-kit 3.0.0b3, for DPA-2 checkpoints such as DPA-SSE). No automatic fallback. | - | - |
| `NEP_RUNTIME` | str | - | `ase`, `gpumd` | NEP execution runtime: 'ase' (Calorine CPUNEP/GPUNEP calculator) or 'gpumd' (native gpumd process). Required for MODEL_TYPE=nep; mliport never infers it or falls back between the two. | - | - |
| `NEP_GPUMD_COMMAND` | str | - | - | Path to the gpumd executable for NEP_RUNTIME=gpumd (and for the GPUNEP ASE adapter). Launched as an argv list, never through a shell; its real path, version and SHA256 are recorded. | - | - |
| `CHARGE` | int | - | `-100..100` | Total molecular charge stored in atoms.info. | - | - |
| `SPIN` | int | - | `0..100` | Molecular spin metadata; UMA omol uses spin multiplicity (2S+1). | - | `SPIN_MULTIPLICITY` |

**Device and resources**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**Molecular dynamics**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**Output**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `WRITE_RESOLVED_CONFIG` | bool | `.TRUE.` | - | Write resolved_config.json before a calculation. | - | - |
| `WRITE_FORCES` | bool | `.TRUE.` | - | Write forces to OUTCAR. | - | - |
| `WRITE_OUTCAR` | bool | - | - | Write the VASP-like OUTCAR interoperability view. | - | - |
| `WRITE_XDATCAR` | bool | - | - | Write the VASP XDATCAR interoperability trajectory. | - | - |
| `WRITE_STRESS` | bool | `.TRUE.` | - | Write stress to OUTCAR. | - | - |
| `WRITE_TRAJECTORY` | bool | - | - | Write ASE trajectory. | - | - |
| `WRITE_JSON` | bool | - | - | Write JSON results. | - | - |
| `OUTPUT_FORMAT` | str | `VASP` | - | Output format (VASP). | - | - |
| `OUTPUT_DIR` | str | - | - | Output directory for the calculation. Used when the command line does not pass --output; the resolved config is written into the final job directory. Relative values resolve against the file that declares them (the INCAR folder for an INCAR), and `mliport run` defaults to <INCAR folder>/results when neither --output nor OUTPUT_DIR is given. | - | - |
| `JOB_NAME` | str | - | - | Optional job name. | - | - |

**General**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## Structure optimization

`CALC_TYPE = OPT` relaxes the structure, with the cell optionally free.

<!-- BEGIN GENERATED: incar-opt -->
**Calculation**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**Model**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Path to the model checkpoint/file. | - | - |
| `TASK` | str | - | - | UMA task (omat/omol/...) or bulk/molecule for other engines. | - | - |
| `INFERENCE_MODE` | str | `default` | `default`, `turbo` | UMA inference mode. Non-default modes are rejected for other engines. | - | - |
| `DEFAULT_DTYPE` | str | - | `float32`, `float64` | MACE model dtype (accuracy-first default: float64). | - | - |
| `HEAD` | str | - | - | MACE or DeepMD model head/branch (multi-task models). | - | - |
| `ACTIVATION_CHECKPOINTING` | bool | `-` | - | GPU memory saving (UMA, overrides inference_mode preset). | - | - |
| `GPU_MEMORY_GROWTH` | bool | - | - | GRACE TensorFlow allocator grows on demand instead of claiming the whole visible GPU. | - | - |
| `GPU_MEMORY_LIMIT_MB` | int | - | `1..` | Hard GRACE TensorFlow logical-device memory limit in MiB. | - | - |
| `NEIGHBOR_CACHE` | bool | - | - | GRACE verlet-style neighbor-list cache (extended cutoff + exact re-filter). The complete periodic-image multiset matches a fresh search; pair ordering may differ. Default: True. | - | - |
| `NEIGHBOR_SKIN` | float | - | `1e-12..` | GRACE neighbor-cache skin in Å: the cached neighbor list is rebuilt when any atom moves more than skin/2. Larger values rebuild less often but filter longer lists. | - | - |
| `D3` | bool | - | - | Add Grimme DFT-D3 dispersion (torch-dftd) on top of the model. Only for models trained on dispersion-free reference data. | - | - |
| `D3_XC` | str | - | - | Reference functional of the model's training data, selecting the D3 parameters (default: pbe). | - | - |
| `D3_DAMPING` | str | - | `zero`, `bj`, `zerom`, `bjm` | DFT-D3 damping function (default: bj, i.e. D3(BJ)). | - | - |
| `DPA_RUNTIME` | str | - | `current`, `legacy` | DPA runtime environment: 'current' (pinned DPA-4-capable deepmd-kit) or 'legacy' (.venv-dpa-legacy, deepmd-kit 3.0.0b3, for DPA-2 checkpoints such as DPA-SSE). No automatic fallback. | - | - |
| `NEP_RUNTIME` | str | - | `ase`, `gpumd` | NEP execution runtime: 'ase' (Calorine CPUNEP/GPUNEP calculator) or 'gpumd' (native gpumd process). Required for MODEL_TYPE=nep; mliport never infers it or falls back between the two. | - | - |
| `NEP_GPUMD_COMMAND` | str | - | - | Path to the gpumd executable for NEP_RUNTIME=gpumd (and for the GPUNEP ASE adapter). Launched as an argv list, never through a shell; its real path, version and SHA256 are recorded. | - | - |
| `CHARGE` | int | - | `-100..100` | Total molecular charge stored in atoms.info. | - | - |
| `SPIN` | int | - | `0..100` | Molecular spin metadata; UMA omol uses spin multiplicity (2S+1). | - | `SPIN_MULTIPLICITY` |

**Device and resources**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**Optimization**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `OPT_ALGO` | str | `FIRE` | `FIRE`, `BFGS`, `LBFGS` | Geometry optimization algorithm. | - | - |
| `FMAX` | float | `0.05` | `0.0..` | Force convergence threshold in eV/Angstrom. | - | - |
| `MAX_STEPS` | int | `500` | `0..` | Maximum optimization steps. | - | - |
| `CELL_OPT` | bool | `.FALSE.` | - | Optimize cell parameters (requires stress support). | - | - |
| `FIX_SYMMETRY` | bool | `.FALSE.` | - | Preserve crystal symmetry during optimization. | - | - |

**Molecular dynamics**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**Output**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `WRITE_RESOLVED_CONFIG` | bool | `.TRUE.` | - | Write resolved_config.json before a calculation. | - | - |
| `WRITE_FORCES` | bool | `.TRUE.` | - | Write forces to OUTCAR. | - | - |
| `WRITE_OUTCAR` | bool | - | - | Write the VASP-like OUTCAR interoperability view. | - | - |
| `WRITE_XDATCAR` | bool | - | - | Write the VASP XDATCAR interoperability trajectory. | - | - |
| `WRITE_STRESS` | bool | `.TRUE.` | - | Write stress to OUTCAR. | - | - |
| `WRITE_TRAJECTORY` | bool | - | - | Write ASE trajectory. | - | - |
| `WRITE_JSON` | bool | - | - | Write JSON results. | - | - |
| `OUTPUT_FORMAT` | str | `VASP` | - | Output format (VASP). | - | - |
| `OUTPUT_DIR` | str | - | - | Output directory for the calculation. Used when the command line does not pass --output; the resolved config is written into the final job directory. Relative values resolve against the file that declares them (the INCAR folder for an INCAR), and `mliport run` defaults to <INCAR folder>/results when neither --output nor OUTPUT_DIR is given. | - | - |
| `JOB_NAME` | str | - | - | Optional job name. | - | - |

**General**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## Molecular dynamics

`CALC_TYPE = MD` runs NVT or NPT MD. Thermostat-specific keys only apply to the
thermostat named in the "Applies when" column; the NPT keys only apply to
`ENSEMBLE = NPT`.

<!-- BEGIN GENERATED: incar-md -->
**Calculation**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**Model**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Path to the model checkpoint/file. | - | - |
| `TASK` | str | - | - | UMA task (omat/omol/...) or bulk/molecule for other engines. | - | - |
| `INFERENCE_MODE` | str | `default` | `default`, `turbo` | UMA inference mode. Non-default modes are rejected for other engines. | - | - |
| `DEFAULT_DTYPE` | str | - | `float32`, `float64` | MACE model dtype (accuracy-first default: float64). | - | - |
| `HEAD` | str | - | - | MACE or DeepMD model head/branch (multi-task models). | - | - |
| `ACTIVATION_CHECKPOINTING` | bool | `-` | - | GPU memory saving (UMA, overrides inference_mode preset). | - | - |
| `GPU_MEMORY_GROWTH` | bool | - | - | GRACE TensorFlow allocator grows on demand instead of claiming the whole visible GPU. | - | - |
| `GPU_MEMORY_LIMIT_MB` | int | - | `1..` | Hard GRACE TensorFlow logical-device memory limit in MiB. | - | - |
| `NEIGHBOR_CACHE` | bool | - | - | GRACE verlet-style neighbor-list cache (extended cutoff + exact re-filter). The complete periodic-image multiset matches a fresh search; pair ordering may differ. Default: True. | - | - |
| `NEIGHBOR_SKIN` | float | - | `1e-12..` | GRACE neighbor-cache skin in Å: the cached neighbor list is rebuilt when any atom moves more than skin/2. Larger values rebuild less often but filter longer lists. | - | - |
| `D3` | bool | - | - | Add Grimme DFT-D3 dispersion (torch-dftd) on top of the model. Only for models trained on dispersion-free reference data. | - | - |
| `D3_XC` | str | - | - | Reference functional of the model's training data, selecting the D3 parameters (default: pbe). | - | - |
| `D3_DAMPING` | str | - | `zero`, `bj`, `zerom`, `bjm` | DFT-D3 damping function (default: bj, i.e. D3(BJ)). | - | - |
| `DPA_RUNTIME` | str | - | `current`, `legacy` | DPA runtime environment: 'current' (pinned DPA-4-capable deepmd-kit) or 'legacy' (.venv-dpa-legacy, deepmd-kit 3.0.0b3, for DPA-2 checkpoints such as DPA-SSE). No automatic fallback. | - | - |
| `NEP_RUNTIME` | str | - | `ase`, `gpumd` | NEP execution runtime: 'ase' (Calorine CPUNEP/GPUNEP calculator) or 'gpumd' (native gpumd process). Required for MODEL_TYPE=nep; mliport never infers it or falls back between the two. | - | - |
| `NEP_GPUMD_COMMAND` | str | - | - | Path to the gpumd executable for NEP_RUNTIME=gpumd (and for the GPUNEP ASE adapter). Launched as an argv list, never through a shell; its real path, version and SHA256 are recorded. | - | - |
| `CHARGE` | int | - | `-100..100` | Total molecular charge stored in atoms.info. | - | - |
| `SPIN` | int | - | `0..100` | Molecular spin metadata; UMA omol uses spin multiplicity (2S+1). | - | `SPIN_MULTIPLICITY` |

**Device and resources**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**Molecular dynamics**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `MD_ENSEMBLE` | str | `NVT` | `NVT`, `NVE`, `NPT` | MD ensemble. | - | - |
| `TEMPERATURE` | float | `300.0` | `0.0..` | Temperature in Kelvin. | - | - |
| `PRESSURE_GPA` | float | `0.0` | - | NPT hydrostatic target pressure in GPa (positive is compressive). | `ENSEMBLE = NPT` | `PRESSURE` |
| `NPT_TDAMP` | float | `100.0` | `0.0..` | MTK thermostat damping time in fs (strictly positive). | `ENSEMBLE = NPT` | - |
| `NPT_PDAMP` | float | `1000.0` | `0.0..` | MTK barostat damping time in fs (strictly positive). | `ENSEMBLE = NPT` | - |
| `NPT_TCHAIN` | int | `3` | `1..` | MTK NPT only: number of Nose-Hoover chain variables in the thermostat; >= 1, default 3 | `ENSEMBLE = NPT` | - |
| `NPT_PCHAIN` | int | `3` | `1..` | MTK NPT only: number of chain variables in the barostat; >= 1, default 3 | `ENSEMBLE = NPT` | - |
| `NPT_TLOOP` | int | `1` | `1..` | MTK NPT only: number of Suzuki-Yoshida sub-steps per thermostat integration; >= 1, default 1 | `ENSEMBLE = NPT` | - |
| `NPT_PLOOP` | int | `1` | `1..` | MTK NPT only: number of Suzuki-Yoshida sub-steps per barostat integration; >= 1, default 1 | `ENSEMBLE = NPT` | - |
| `TIMESTEP` | float | `1.0` | `0.0..` | Time step in femtoseconds. | - | - |
| `STEPS` | int | `1000` | `0..` | Number of production MD steps. | - | `PRODUCTION_STEPS` |
| `EQUILIBRATION_STEPS` | int | `0` | `0..` | Number of same-ensemble equilibration steps before production. | - | `EQUIL_STEPS` |
| `THERMOSTAT` | str | `LANGEVIN` | `LANGEVIN`, `BUSSI`, `NHC` | NVT thermostat (Langevin, Bussi/CSVR, or Nose-Hoover chain). | - | - |
| `FRICTION` | float | `0.001` | `0.0..` | Langevin friction coefficient (1/fs). | `THERMOSTAT = LANGEVIN` | - |
| `BUSSI_TAU` | float | `1000.0` | `0.0..` | Bussi/CSVR coupling time in femtoseconds. | `THERMOSTAT = BUSSI` | - |
| `NHC_TDAMP` | float | `100.0` | `0.0..` | Nose-Hoover-chain damping time in femtoseconds. | `THERMOSTAT = NHC` | - |
| `NHC_TCHAIN` | int | `3` | `1..` | Number of Nose-Hoover thermostat variables. | `THERMOSTAT = NHC` | - |
| `NHC_TLOOP` | int | `1` | `1..` | Nose-Hoover thermostat integration substeps. | `THERMOSTAT = NHC` | - |
| `SAVE_INTERVAL` | int | `10` | `1..` | Interval for saving trajectory frames. | - | `TRAJECTORY_INTERVAL` |
| `PRE_RELAX` | bool | - | - | Pre-relax the structure before MD (legacy). | - | - |
| `PRE_RELAX_STEPS` | int | `50` | `0..` | Maximum pre-relaxation steps (legacy). | - | - |
| `PRE_RELAX_FMAX` | float | `0.1` | `0.0..` | Pre-relaxation force threshold (legacy). | - | - |
| `PRE_RELAX_MODE` | str | `none` | `none`, `positions`, `cell` | Pre-relaxation mode (plan section 13.2; Phase 3). | - | - |
| `VELOCITY_POLICY` | str | `auto` | `auto`, `initialize`, `preserve` | Velocity initialisation policy (plan section 13.5; Phase 3). | - | - |
| `COM_POLICY` | str | `auto` | `auto`, `none`, `initialize_only`, `constraint` | Explicit center-of-mass policy for MD. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**Output**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `WRITE_RESOLVED_CONFIG` | bool | `.TRUE.` | - | Write resolved_config.json before a calculation. | - | - |
| `WRITE_FORCES` | bool | `.TRUE.` | - | Write forces to OUTCAR. | - | - |
| `WRITE_OUTCAR` | bool | - | - | Write the VASP-like OUTCAR interoperability view. | - | - |
| `WRITE_XDATCAR` | bool | - | - | Write the VASP XDATCAR interoperability trajectory. | - | - |
| `WRITE_STRESS` | bool | `.TRUE.` | - | Write stress to OUTCAR. | - | - |
| `WRITE_TRAJECTORY` | bool | - | - | Write ASE trajectory. | - | - |
| `WRITE_JSON` | bool | - | - | Write JSON results. | - | - |
| `OUTPUT_FORMAT` | str | `VASP` | - | Output format (VASP). | - | - |
| `OUTPUT_DIR` | str | - | - | Output directory for the calculation. Used when the command line does not pass --output; the resolved config is written into the final job directory. Relative values resolve against the file that declares them (the INCAR folder for an INCAR), and `mliport run` defaults to <INCAR folder>/results when neither --output nor OUTPUT_DIR is given. | - | - |
| `JOB_NAME` | str | - | - | Optional job name. | - | - |

**General**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## NEB

`CALCULATION = NEB` (`CALC_TYPE = NEB` is the same key) runs a fixed-cell
nudged elastic band between two endpoint structures. NEB refuses the output
switches that would silently drop a mandatory record.

<!-- BEGIN GENERATED: incar-neb -->
**Calculation**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**Model**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Path to the model checkpoint/file. | - | - |
| `TASK` | str | - | - | UMA task (omat/omol/...) or bulk/molecule for other engines. | - | - |
| `INFERENCE_MODE` | str | `default` | `default`, `turbo` | UMA inference mode. Non-default modes are rejected for other engines. | - | - |
| `DEFAULT_DTYPE` | str | - | `float32`, `float64` | MACE model dtype (accuracy-first default: float64). | - | - |
| `HEAD` | str | - | - | MACE or DeepMD model head/branch (multi-task models). | - | - |
| `ACTIVATION_CHECKPOINTING` | bool | `-` | - | GPU memory saving (UMA, overrides inference_mode preset). | - | - |
| `GPU_MEMORY_GROWTH` | bool | - | - | GRACE TensorFlow allocator grows on demand instead of claiming the whole visible GPU. | - | - |
| `GPU_MEMORY_LIMIT_MB` | int | - | `1..` | Hard GRACE TensorFlow logical-device memory limit in MiB. | - | - |
| `NEIGHBOR_CACHE` | bool | - | - | GRACE verlet-style neighbor-list cache (extended cutoff + exact re-filter). The complete periodic-image multiset matches a fresh search; pair ordering may differ. Default: True. | - | - |
| `NEIGHBOR_SKIN` | float | - | `1e-12..` | GRACE neighbor-cache skin in Å: the cached neighbor list is rebuilt when any atom moves more than skin/2. Larger values rebuild less often but filter longer lists. | - | - |
| `D3` | bool | - | - | Add Grimme DFT-D3 dispersion (torch-dftd) on top of the model. Only for models trained on dispersion-free reference data. | - | - |
| `D3_XC` | str | - | - | Reference functional of the model's training data, selecting the D3 parameters (default: pbe). | - | - |
| `D3_DAMPING` | str | - | `zero`, `bj`, `zerom`, `bjm` | DFT-D3 damping function (default: bj, i.e. D3(BJ)). | - | - |
| `DPA_RUNTIME` | str | - | `current`, `legacy` | DPA runtime environment: 'current' (pinned DPA-4-capable deepmd-kit) or 'legacy' (.venv-dpa-legacy, deepmd-kit 3.0.0b3, for DPA-2 checkpoints such as DPA-SSE). No automatic fallback. | - | - |
| `NEP_RUNTIME` | str | - | `ase`, `gpumd` | NEP execution runtime: 'ase' (Calorine CPUNEP/GPUNEP calculator) or 'gpumd' (native gpumd process). Required for MODEL_TYPE=nep; mliport never infers it or falls back between the two. | - | - |
| `NEP_GPUMD_COMMAND` | str | - | - | Path to the gpumd executable for NEP_RUNTIME=gpumd (and for the GPUNEP ASE adapter). Launched as an argv list, never through a shell; its real path, version and SHA256 are recorded. | - | - |
| `CHARGE` | int | - | `-100..100` | Total molecular charge stored in atoms.info. | - | - |
| `SPIN` | int | - | `0..100` | Molecular spin metadata; UMA omol uses spin multiplicity (2S+1). | - | `SPIN_MULTIPLICITY` |

**Device and resources**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**Optimization**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `FMAX` | float | `0.03` | `0.0..` | Force convergence threshold in eV/Angstrom. | - | - |
| `MAX_STEPS` | int | `1000` | `0..` | Maximum optimization steps. | - | - |

**Molecular dynamics**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**NEB**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `NEB_INITIAL` | str | required (`initial.vasp`) | - | Initial endpoint structure for NEB. | - | - |
| `NEB_FINAL` | str | required (`final.vasp`) | - | Final endpoint structure for NEB. | - | - |
| `NEB_IMAGES` | int | `7` | `1..` | Number of intermediate images; endpoints are additional. | - | `IMAGES` |
| `CLIMB` | bool | `.FALSE.` | - | Run a second climbing-image NEB stage. | - | `NEB_CLIMB` |
| `NEB_METHOD` | str | `improvedtangent` | `improvedtangent` | ASE NEB tangent method. | - | - |
| `NEB_INTERPOLATION` | str | `linear` | `linear`, `idpp` | Initial band interpolation method. | - | - |
| `NEB_PATH_CONVENTION` | str | `mic` | `mic`, `unwrapped` | Periodic path convention. | - | - |
| `NEB_SPRING` | float | `0.1` | `1e-12..` | NEB spring constant in eV/Angstrom^2. | - | - |
| `NEB_PRE_FMAX` | float | `0.1` | `1e-12..` | Ordinary pre-stage force criterion in eV/Angstrom. | - | - |
| `NEB_PRE_MAX_STEPS` | int | `300` | `0..` | Maximum ordinary pre-stage optimizer steps. | - | - |
| `NEB_MAXSTEP` | float | `0.1` | `1e-12..` | Maximum FIRE displacement per step in Angstrom. | - | - |
| `NEB_ENDPOINT_POLICY` | str | `validate` | `validate`, `relax` | Validate or relax endpoint geometries before NEB. | - | - |
| `NEB_ENDPOINT_FMAX` | float | `0.02` | `1e-12..` | Endpoint convergence threshold in eV/Angstrom. | - | - |
| `NEB_ENDPOINT_STEPS` | int | `500` | `1..` | Maximum endpoint relaxation steps. | - | - |
| `IDPP_FMAX` | float | `0.1` | `1e-12..` | IDPP interpolation force criterion. | - | - |
| `IDPP_STEPS` | int | `200` | `1..` | Maximum IDPP interpolation steps. | - | - |
| `IDPP_MIC` | bool | `.TRUE.` | - | Use minimum-image distances in IDPP. | - | - |
| `NEB_MIN_DISTANCE` | float | `0.5` | `1e-12..` | Minimum allowed interatomic distance in Angstrom. | - | - |
| `NEB_CHECKPOINT_INTERVAL` | int | `10` | `1..` | Trusted optimizer-state interval between full-band checkpoints. | - | - |

**Output**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `WRITE_RESOLVED_CONFIG` | bool | `.TRUE.` | - | Write resolved_config.json before a calculation. | - | - |
| `WRITE_FORCES` | bool | `.TRUE.` | - | Write forces to OUTCAR. | - | - |
| `WRITE_OUTCAR` | bool | - | - | Write the VASP-like OUTCAR interoperability view. | - | - |
| `WRITE_XDATCAR` | bool | - | - | Write the VASP XDATCAR interoperability trajectory. | - | - |
| `WRITE_STRESS` | bool | `.TRUE.` | - | Write stress to OUTCAR. | - | - |
| `WRITE_TRAJECTORY` | bool | - | - | Write ASE trajectory. | - | - |
| `WRITE_JSON` | bool | - | - | Write JSON results. | - | - |
| `OUTPUT_FORMAT` | str | `VASP` | - | Output format (VASP). | - | - |
| `OUTPUT_DIR` | str | - | - | Output directory for the calculation. Used when the command line does not pass --output; the resolved config is written into the final job directory. Relative values resolve against the file that declares them (the INCAR folder for an INCAR), and `mliport run` defaults to <INCAR folder>/results when neither --output nor OUTPUT_DIR is given. | - | - |
| `JOB_NAME` | str | - | - | Optional job name. | - | - |

**General**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## Configuration search

`CALC_TYPE = CONFSEARCH` runs the MD-based configuration search. `mliport run`
executes it through the same code path as `mliport confsearch run`.

<!-- BEGIN GENERATED: incar-confsearch -->
**Calculation**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**Model**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Path to the model checkpoint/file. | - | - |
| `TASK` | str | - | - | UMA task (omat/omol/...) or bulk/molecule for other engines. | - | - |
| `INFERENCE_MODE` | str | - | `default`, `turbo` | UMA inference mode. Non-default modes are rejected for other engines. | - | - |
| `DEFAULT_DTYPE` | str | - | `float32`, `float64` | MACE model dtype (accuracy-first default: float64). | - | - |
| `HEAD` | str | - | - | MACE or DeepMD model head/branch (multi-task models). | - | - |
| `ACTIVATION_CHECKPOINTING` | bool | `-` | - | GPU memory saving (UMA, overrides inference_mode preset). | - | - |
| `GPU_MEMORY_GROWTH` | bool | - | - | GRACE TensorFlow allocator grows on demand instead of claiming the whole visible GPU. | - | - |
| `GPU_MEMORY_LIMIT_MB` | int | - | `1..` | Hard GRACE TensorFlow logical-device memory limit in MiB. | - | - |
| `NEIGHBOR_CACHE` | bool | - | - | GRACE verlet-style neighbor-list cache (extended cutoff + exact re-filter). The complete periodic-image multiset matches a fresh search; pair ordering may differ. Default: True. | - | - |
| `NEIGHBOR_SKIN` | float | - | `1e-12..` | GRACE neighbor-cache skin in Å: the cached neighbor list is rebuilt when any atom moves more than skin/2. Larger values rebuild less often but filter longer lists. | - | - |
| `D3` | bool | - | - | Add Grimme DFT-D3 dispersion (torch-dftd) on top of the model. Only for models trained on dispersion-free reference data. | - | - |
| `D3_XC` | str | - | - | Reference functional of the model's training data, selecting the D3 parameters (default: pbe). | - | - |
| `D3_DAMPING` | str | - | `zero`, `bj`, `zerom`, `bjm` | DFT-D3 damping function (default: bj, i.e. D3(BJ)). | - | - |
| `DPA_RUNTIME` | str | - | `current`, `legacy` | DPA runtime environment: 'current' (pinned DPA-4-capable deepmd-kit) or 'legacy' (.venv-dpa-legacy, deepmd-kit 3.0.0b3, for DPA-2 checkpoints such as DPA-SSE). No automatic fallback. | - | - |
| `NEP_RUNTIME` | str | - | `ase`, `gpumd` | NEP execution runtime: 'ase' (Calorine CPUNEP/GPUNEP calculator) or 'gpumd' (native gpumd process). Required for MODEL_TYPE=nep; mliport never infers it or falls back between the two. | - | - |
| `NEP_GPUMD_COMMAND` | str | - | - | Path to the gpumd executable for NEP_RUNTIME=gpumd (and for the GPUNEP ASE adapter). Launched as an argv list, never through a shell; its real path, version and SHA256 are recorded. | - | - |
| `CHARGE` | int | - | `-100..100` | Total molecular charge stored in atoms.info. | - | - |
| `SPIN` | int | - | `0..100` | Molecular spin metadata; UMA omol uses spin multiplicity (2S+1). | - | `SPIN_MULTIPLICITY` |

**Device and resources**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**Molecular dynamics**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `TIMESTEP` | float | `0.5` | `0.0..` | Time step in femtoseconds. | - | - |
| `FRICTION` | float | `0.01` | `0.0..` | Langevin friction coefficient (1/fs). | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**Configuration search**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `CS_MODE` | str | `adsorbate` | `adsorbate` | Search mode; v1 accepts only 'adsorbate' (bulk_mobile, melt_quench, interface, multi_adsorbate are rejected) | - | - |
| `CS_STARTS` | str | required (`REQUIRED`) | - | Mode A: start structures (files or directories), one independent replica each; needs cs_adsorbate_indices | - | - |
| `CS_ADSORBATE_INDICES` | str | required (`REQUIRED`) | - | Mode A: 0-based adsorbate atom indices such as 300-341 or 0,5,7-9; must be one connected covalent molecule | - | - |
| `CS_SLAB` | str | required (`slab.vasp`) | - | Mode B: slab structure file, exactly one frame; conformers from cs_molecule are placed above it | `CS_MODE = MODE_B` | - |
| `CS_MOLECULE` | str | required (`molecule.xyz`) | - | Mode B: molecule conformer file; one conformer is drawn at random for each placed start | `CS_MODE = MODE_B` | - |
| `CS_N_STARTS` | int | `10` | `1..` | Mode B: number of placed start structures to build, each becoming one independent replica; minimum 1 | `CS_MODE = MODE_B` | - |
| `CS_FIX_BELOW` | float | - | - | Extend the input FixAtoms to slab atoms within this height in Angstrom of the lowest slab atom; unset = off | - | - |
| `CS_MD_MOBILE` | str | `adsorbate` | `adsorbate`, `adsorbate_and_top`, `all_free` | Which atoms are mobile in MD: adsorbate (default), adsorbate_and_top or all_free (every atom not fixed) | - | - |
| `CS_TOP_DEPTH` | float | `2.0` | - | Depth in Angstrom below the slab top of the extra mobile atoms in cs_md_mobile=adsorbate_and_top; default 2.0 | - | - |
| `CS_TEMPERATURE_HIGH` | float | `500.0` | - | Hold temperature in K of every heat-hold-cool cycle and of the initial velocities; default 500 | - | - |
| `CS_TEMPERATURE_LOW` | float | `300.0` | - | End temperature in K of the linear cooling ramp in each cycle, must be <= cs_temperature_high; default 300 | - | - |
| `CS_HOLD_PS` | float | `2.0` | - | Duration in ps of the hold phase at cs_temperature_high in each cycle; must be a whole number of timesteps | - | - |
| `CS_COOL_PS` | float | `2.0` | - | Duration in ps of the linear cooling ramp from cs_temperature_high to cs_temperature_low in each cycle | - | - |
| `CS_SNAPSHOT_INTERVAL_PS` | float | `0.5` | - | Spacing in ps between hold-phase snapshots that get quenched; must be < cs_hold_ps, 0 disables snapshots | - | - |
| `CS_RAMP_INTERVAL` | int | `10` | `1..` | Number of MD steps between thermostat set-point updates during the linear cooling ramp; default 10 | - | - |
| `CS_CYCLES` | int | `5` | `0..` | Number of heat-hold-cool cycles run per replica after its start quench; 0 leaves only the start quenches | - | - |
| `CS_SAVE_INTERVAL` | int | `100` | `1..` | Write one raw MD frame every this many MD steps (default 100); quench snapshots and cycle ends are always kept | - | - |
| `CS_ACCEPT_TEMPERATURE` | float | `500.0` | - | Temperature in K of the Metropolis test that accepts or rejects a replica's end-of-cycle minimum; default 500 | - | - |
| `CS_QUENCH_FMAX` | float | `0.05` | - | FIRE force convergence threshold in eV/A for every snapshot and cycle-end quench; default 0.05 | - | - |
| `CS_QUENCH_STEPS` | int | `2000` | `1..` | Maximum FIRE steps per quench; exceeding it marks the minimum quench_not_converged and unranked; default 2000 | - | - |
| `CS_BOND_SCALE` | float | `1.2` | - | Bond test in the covalent graphs: bonded when closer than this multiple of the covalent-radii sum; default 1.2 | - | - |
| `CS_BOND_BREAK_SCALE` | float | `1.5` | - | MD aborts if an adsorbate bond exceeds this multiple of the covalent-radii sum; must be > cs_bond_scale (1.5) | - | - |
| `CS_IONIC_SPECIES` | str | `Li,Na,K,Rb,Cs,Mg,Ca` | - | Element symbols excluded from every covalent graph (ionic hopping is no reaction); default Li,Na,K,Rb,Cs,Mg,Ca | - | - |
| `CS_REACTION_POLICY` | str | `exclude` | `exclude`, `include` | Whether minima whose covalent bonds changed are ranked and exported: exclude (default) or include | - | - |
| `CS_DESORPTION_DISTANCE` | float | `5.0` | - | Adsorbate-slab distance in Angstrom above which a minimum is labelled desorbed and dropped; default 5.0 | - | - |
| `CS_MIN_IMAGE_DISTANCE` | float | `4.0` | - | Adsorbate in-plane self-image distance in Angstrom below which a minimum is image_contact; default 4.0 | - | - |
| `CS_MIN_VACUUM_GAP` | float | `8.0` | - | Gap in Angstrom from the adsorbate to the slab image one cell along c; smaller marks image_contact (8.0) | - | - |
| `CS_PLACE_CONTACT_SCALE` | float | `1.5` | - | Mode B: target ratio of the closest adsorbate-slab pair to the covalent-radii sum (unbonded); default 1.5 | - | - |
| `CS_PLACE_ANCHOR` | str | - | - | Mode B: 0-based anchor atom indices in the molecule file; the lowest atom is drawn from them and placed bonded | - | - |
| `CS_PLACE_ANCHOR_SCALE` | float | `1.1` | - | Mode B with cs_place_anchor: anchored pair ratio to the covalent-radii sum, i.e. already bonded; default 1.1 | - | - |
| `CS_PLACE_OTHER_MIN_SCALE` | float | `1.3` | - | Mode B with cs_place_anchor: lower bound for every other adsorbate-slab pair in radii-sum units; default 1.3 | - | - |
| `CS_PLACE_ANCHOR_TARGETS` | str | - | - | Mode B with cs_place_anchor: slab elements the closest slab atom must belong to (e.g. an ionic species) | - | - |
| `CS_PLACE_MAX_ATTEMPTS` | int | `1000` | `1..` | Mode B: random placement attempts tried per start before the run fails; default 1000 | - | - |
| `CS_DEDUP_ENERGY_TOL` | float | `0.01` | - | Energy window in eV within which two minima may be merged as the same minimum; default 0.010 | - | - |
| `CS_DEDUP_GEOM_TOL` | float | `0.2` | - | Per-atom displacement in Angstrom below which two minima count as the same geometry (rule G); default 0.20 | - | - |
| `CS_DEDUP_FP_TOL` | float | `0.0004` | - | Fingerprint distance below which two minima merge (rule D, catches symmetry equivalents); default 0.0004 | - | - |
| `CS_FP_RCUT` | float | `8.0` | - | Cutoff radius in Angstrom of the adsorbate Oganov-Valle fingerprint used for dedup; default 8.0 | - | - |
| `CS_FP_BINWIDTH` | float | `0.02` | - | Fingerprint histogram bin width in Angstrom, at most half of cs_fp_sigma; default 0.02 | - | - |
| `CS_FP_SIGMA` | float | `0.2` | - | Gaussian smearing width in Angstrom applied to the fingerprint histogram bins; default 0.20 | - | - |
| `CS_TOP_K` | int | `10` | `1..` | Maximum number of ranked minima exported as candidates for DFT re-ranking; default 10 | - | - |
| `CS_EXPORT_WINDOW` | float | `0.5` | - | Only minima within this energy window in eV above the best one are refined and exported; default 0.5 | - | - |
| `CS_REFINE` | bool | `.TRUE.` | - | Re-refine the exported candidates with a tighter force threshold before ranking and export; default true | - | - |
| `CS_REFINE_FMAX` | float | `0.02` | - | Force threshold in eV/A for the final refinement, must be <= cs_quench_fmax; default 0.02 | - | - |
| `CS_REFINE_STEPS` | int | `3000` | `1..` | Maximum number of FIRE steps in the final refinement of each exported minimum; default 3000 | - | - |
| `CS_ALLOW_DIRECT_FORCES` | bool | `.FALSE.` | - | Allow a model that reports direct_forces=true, in which case the ranking is not meaningful; default false | - | - |

**Output**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `WRITE_RESOLVED_CONFIG` | bool | `.TRUE.` | - | Write resolved_config.json before a calculation. | - | - |
| `WRITE_FORCES` | bool | `.TRUE.` | - | Write forces to OUTCAR. | - | - |
| `WRITE_OUTCAR` | bool | - | - | Write the VASP-like OUTCAR interoperability view. | - | - |
| `WRITE_XDATCAR` | bool | - | - | Write the VASP XDATCAR interoperability trajectory. | - | - |
| `WRITE_STRESS` | bool | `.TRUE.` | - | Write stress to OUTCAR. | - | - |
| `WRITE_TRAJECTORY` | bool | - | - | Write ASE trajectory. | - | - |
| `WRITE_JSON` | bool | - | - | Write JSON results. | - | - |
| `OUTPUT_FORMAT` | str | `VASP` | - | Output format (VASP). | - | - |
| `OUTPUT_DIR` | str | - | - | Output directory for the calculation. Used when the command line does not pass --output; the resolved config is written into the final job directory. Relative values resolve against the file that declares them (the INCAR folder for an INCAR), and `mliport run` defaults to <INCAR folder>/results when neither --output nor OUTPUT_DIR is given. | - | - |
| `JOB_NAME` | str | - | - | Optional job name. | - | - |

**General**

| Key | Type | Default | Allowed values | Meaning | Applies when | Aliases |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->
