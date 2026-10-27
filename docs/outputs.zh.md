# 输出文件

每次运行会生成两类文件。一类沿用 VASP 的文件格式，方便用你熟悉的工具查看；另一类是 JSON，
记录这次计算实际用了什么、算出了什么。两者有出入时，以 JSON 为准。

这些文件只是格式和 VASP 相同，数值上和 VASP 计算没有对应关系：没有 SCF、k 点、ENCUT 和
POTCAR，绝对能量只对产生它的模型有意义。`OUTCAR` 主要给人看，也方便读取 VASP 输出的工具使用；
程序处理请读 JSON。

## 单点与结构优化

直接写在输出目录里：

```text
OUTCAR                  类似 VASP 的能量、力、应力报告
OSZICAR                 每一步的能量与力（仅结构优化）
CONTCAR                 最终结构（单点时是输入结构的副本）
mliport_results.json    结果，以及模型的身份
resolved_config.json    配置，以及每个值的来源
run.log                 运行的实时日志
```

`mliport_results.json` 里，物理量在 `calculation.results` 下（能量、每原子能量、力及其统计、
应力和压强），结构在 `calculation.system` 下，结构优化的收敛情况在 `calculation.optimization`
下（`converged`、`nsteps`、`fmax_threshold`）。

运行期间输出目录会被锁定。如果目录里已经有结果（之前的计算，或者 VASP 自己的 `OUTCAR`、
`CONTCAR`），程序会拒绝写入，确定要覆盖时加 `--overwrite`。

## 分子动力学

MD 的完整原始数据和 VASP 格式的导出文件分开存放：

```text
raw/trajectory.traj     标准 ASE 轨迹
raw/md.csv              逐帧标量：能量、温度等
raw/mliport_results.json
vasp/OUTCAR, vasp/CONTCAR, vasp/XDATCAR
artifacts.json          清单：文件、单位、可观测量定义、哈希
resolved_config.json
run.log
```

`artifacts.json` 写明了单位和每个物理量的定义，例如 `forces_raw_eV_A`（原始的力）和
`forces_applied_eV_A`（施加约束之后的力），以及构型应力和总应力。任何 ASE 轨迹都可以用
`mliport convert-xdatcar` 转成标准的 XDATCAR。

## NEB

```text
neb_results.json        能量、力、收敛状态、势垒数值
run_context.json        运行信息，包括 gradient_preflight
resolved_config.json
artifacts.json
checkpoints/            完整能带检查点（band.traj + checkpoint.json）
vasp_path/              每个 image 一个 POSCAR，可用 VASP 的 NEB 工具处理
```

## 分析

`mliport analyze` 的结果写在 `<run>/analysis/<task>/<analysis-id>/` 下，包括 JSON、CSV，有图
的分析还会输出 PNG 和 SVG，同时记录这次分析的设置。详见[轨迹分析](analysis/index.md)。

## 单位

| 物理量 | 单位 |
|---|---|
| 时间 | fs |
| 长度 | Å |
| 能量 | eV |
| 力 | eV/Å |
| 应力 | eV/Å³ |
| 压强 | GPa |
| 温度 | K |

## 计算记录

`resolved_config.json` 记录每个参数的最终值和来源（内置默认值、设置文件、别名、profile、
INCAR 或命令行），据此可以准确复现或核查一次计算。

结果文件和 `run_context.json` 里还记录了：

- **模型信息**：`model_type`、路径、task、head、精度、后端和框架版本，能获取时还有
  `model_sha256`；
- **设备**：请求的和实际使用的，能获取时还有 GPU 的 UUID；
- **软件版本**：运行这次计算的 mliport commit；
- **哈希值**：输入结构、模型和输出文件的哈希，分析和验证记录会用到它们。

### 请求的设备与实际设备

请求 `cuda:0` 的运行，记录的实际设备可能是另一个编号或者一个 UUID。写论文方法部分时，以
记录的实际设备为准。如果写的是 `unknown`，就不要声称这次计算用了某块具体的 GPU。
