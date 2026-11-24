# INCAR 键参考

mliport 在 `INCAR.mliport` 里接受的全部键、含义、默认值和允许的取值。下面的表格由
`mliport/config/schema.py` 里的选项表生成，模板和 `mliport config schema` 用的是同一份
表，所以这一页不会和代码脱节。

有三条适用于全页。键写成 `KEY = VALUE`，一行一个，`#` 开始注释，文件名是
`INCAR.mliport`（mliport 不会去读名字就叫 `INCAR` 的文件，那可能是真正的 VASP 输入）。
`REQUIRED` 是占位符：模板没法替你编一个模型权重或起始结构，需要你填。优先级依次是命令行、
profile、模型别名、INCAR、`settings.ini`、内置默认值；分层规则和任务文件夹用法见
[配置](configuration.md)。

表格里的“含义”一列直接来自代码里的选项表（英文），所以保持英文原文，避免与实现脱节；分类标题已翻译。
单位就是每行里写的：飞秒（fs）、皮秒（ps）、埃、电子伏（eV）、开尔文和吉帕（GPa）。能量是
MLIP 的总能量，只有在同一个模型身份下才能相互比较。

## 所有计算类型通用的键

<!-- BEGIN GENERATED: incar-common -->
**计算类型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**模型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Model checkpoint file or directory. A value without any path separator is a file name: it is resolved next to the declaring file first, then in <MLIPORT_ROOT>/models (one level of subdirectories allowed). | - | - |
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

**设备与资源**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**分子动力学**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**输出控制**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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

**通用**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## 单点

`CALC_TYPE = SP` 计算一个结构的能量、力和应力。输出控制键在结构优化和 MD 里同样适用。

<!-- BEGIN GENERATED: incar-sp -->
**计算类型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**模型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Model checkpoint file or directory. A value without any path separator is a file name: it is resolved next to the declaring file first, then in <MLIPORT_ROOT>/models (one level of subdirectories allowed). | - | - |
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

**设备与资源**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**分子动力学**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**输出控制**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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

**通用**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## 结构优化

`CALC_TYPE = OPT` 做结构优化，晶胞可以一起放开。

<!-- BEGIN GENERATED: incar-opt -->
**计算类型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**模型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Model checkpoint file or directory. A value without any path separator is a file name: it is resolved next to the declaring file first, then in <MLIPORT_ROOT>/models (one level of subdirectories allowed). | - | - |
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

**设备与资源**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**结构优化**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `OPT_ALGO` | str | `FIRE` | `FIRE`, `BFGS`, `LBFGS` | Geometry optimization algorithm. | - | - |
| `FMAX` | float | `0.05` | `0.0..` | Force convergence threshold in eV/Angstrom. | - | - |
| `MAX_STEPS` | int | `500` | `0..` | Maximum optimization steps. | - | - |
| `CELL_OPT` | bool | `.FALSE.` | - | Optimize cell parameters (requires stress support). | - | - |
| `FIX_SYMMETRY` | bool | `.FALSE.` | - | Preserve crystal symmetry during optimization. | - | - |

**分子动力学**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**输出控制**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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

**通用**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## 分子动力学

`CALC_TYPE = MD` 跑 NVT 或 NPT 的 MD。恒温器专属的键只对“适用条件”里写的那个恒温器生效；
NPT 的键只在 `ENSEMBLE = NPT` 时生效。

<!-- BEGIN GENERATED: incar-md -->
**计算类型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**模型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Model checkpoint file or directory. A value without any path separator is a file name: it is resolved next to the declaring file first, then in <MLIPORT_ROOT>/models (one level of subdirectories allowed). | - | - |
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

**设备与资源**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**分子动力学**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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
| `RAW_FORCES_STORAGE` | str | `json` | `json`, `binary` | How raw (unprojected) MD forces are stored in raw/trajectory.traj: 'json' (info field, default) or 'binary' (exact float64 ULM field, about 35% smaller; read with mliport.trajectory_forces). | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**输出控制**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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

**通用**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## NEB

`CALCULATION = NEB`（`CALC_TYPE = NEB` 是同一个键）在两个端点结构之间做固定晶胞的
nudged elastic band。会让必需记录悄悄消失的输出开关，NEB 会直接拒绝。

<!-- BEGIN GENERATED: incar-neb -->
**计算类型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**模型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Model checkpoint file or directory. A value without any path separator is a file name: it is resolved next to the declaring file first, then in <MLIPORT_ROOT>/models (one level of subdirectories allowed). | - | - |
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

**设备与资源**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**结构优化**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `FMAX` | float | `0.03` | `0.0..` | Force convergence threshold in eV/Angstrom. | - | - |
| `MAX_STEPS` | int | `1000` | `0..` | Maximum optimization steps. | - | - |

**分子动力学**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**NEB**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `NEB_INITIAL` | str | required (`REQUIRED`) | - | Initial endpoint structure for NEB. | - | - |
| `NEB_FINAL` | str | required (`REQUIRED`) | - | Final endpoint structure for NEB. | - | - |
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

**输出控制**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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

**通用**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## 构型搜索

`CALC_TYPE = CONFSEARCH` 跑基于 MD 的构型搜索。`mliport run` 走的是和
`mliport confsearch run` 完全相同的代码路径。

<!-- BEGIN GENERATED: incar-confsearch -->
**计算类型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**模型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Model checkpoint file or directory. A value without any path separator is a file name: it is resolved next to the declaring file first, then in <MLIPORT_ROOT>/models (one level of subdirectories allowed). | - | - |
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

**设备与资源**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**分子动力学**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `TIMESTEP` | float | `0.5` | `0.0..` | Time step in femtoseconds. | - | - |
| `FRICTION` | float | `0.01` | `0.0..` | Langevin friction coefficient (1/fs). | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**构型搜索**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CS_MODE` | str | `adsorbate` | `adsorbate` | Search mode; v1 accepts only 'adsorbate' (bulk_mobile, melt_quench, interface, multi_adsorbate are rejected) | - | - |
| `CS_STARTS` | str | required (`REQUIRED`) | - | Mode A: start structures (files or directories), one independent replica each; needs cs_adsorbate_indices | - | - |
| `CS_ADSORBATE_INDICES` | str | required (`REQUIRED`) | - | Mode A: 0-based adsorbate atom indices such as 300-341 or 0,5,7-9; must be one connected covalent molecule | - | - |
| `CS_SLAB` | str | `slab.vasp` (example) | - | Mode B: slab structure file, exactly one frame; conformers from cs_molecule are placed above it | - | - |
| `CS_MOLECULE` | str | `molecule.xyz` (example) | - | Mode B: molecule conformer file; one conformer is drawn at random for each placed start | - | - |
| `CS_N_STARTS` | int | `10` | `1..` | Mode B: number of placed start structures to build, each becoming one independent replica; minimum 1 | - | - |
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

**输出控制**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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

**通用**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->

## 批量计算

`mliport batch` 扫一个目录里的结构，而不是跑一份 INCAR，所以这些键来自 `settings.ini` 或
命令行；`CALC_TYPE = BATCH` 的 `INCAR.mliport` 会被 `mliport run` 拒绝，并提示改用批量
子命令。

<!-- BEGIN GENERATED: incar-batch -->
**计算类型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `CALC_TYPE` | str | - | `sp`, `opt`, `md`, `neb`, `batch`, `confsearch` | Top-level calculation type (selects the runner). | - | `CALCULATION` |

**模型**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `MODEL_TYPE` | str | - | `uma`, `fairchem`, `mace`, `dpa`, `grace`, `nep` | MLIP engine. | - | - |
| `MODEL_PATH` | str | required (`REQUIRED`) | - | Model checkpoint file or directory. A value without any path separator is a file name: it is resolved next to the declaring file first, then in <MLIPORT_ROOT>/models (one level of subdirectories allowed). | - | - |
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

**设备与资源**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `DEVICE` | str | `cpu` | - | Compute device: cpu, cuda, gpu or cuda:N. | - | - |
| `TORCH_NUM_THREADS` | int | `-` | `1..` | Backend CPU intra-op threads (PyTorch for UMA/MACE/DPA .pt; TensorFlow for GRACE; legacy canonical field name). | - | `CPU_THREADS` |

**结构优化**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `OPT_ALGO` | str | - | `FIRE`, `BFGS`, `LBFGS` | Geometry optimization algorithm. | - | - |
| `FMAX` | float | - | `0.0..` | Force convergence threshold in eV/Angstrom. | - | - |
| `MAX_STEPS` | int | - | `0..` | Maximum optimization steps. | - | - |
| `CELL_OPT` | bool | - | - | Optimize cell parameters (requires stress support). | - | - |
| `FIX_SYMMETRY` | bool | - | - | Preserve crystal symmetry during optimization. | - | - |

**分子动力学**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `FMAX_ABORT` | float | - | `0.0..` | MD force-safety abort threshold in eV/Angstrom. | - | - |
| `SEED` | int | - | `0..` | Random seed for reproducible MD. | - | `DEFAULT_SEED` |

**输出控制**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
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

**批量计算**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `SUB_CALC_TYPE` | str | - | `sp`, `opt` | Sub-calculation type for a sweep batch (sp/opt). | - | - |
| `PATTERN` | str | - | - | Glob pattern for batch structure discovery. | - | - |

**通用**

| 键 | 类型 | 默认值 | 取值范围 | 含义 | 适用条件 | 别名 |
|---|---|---|---|---|---|---|
| `STRICT_CONFIG` | bool | `.TRUE.` | - | Treat unknown configuration keys as errors. | - | - |

<!-- END GENERATED -->
