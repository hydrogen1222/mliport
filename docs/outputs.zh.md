# 输出文件

每次运行会留下两类文件。第一类长得像 VASP 的，所以你已有的习惯和工具可以沿用。第二
类是 JSON，记录实际发生了什么；两者不一致时，以它为准。

mliport 不是 VASP，它的文件在数值上也不与 VASP 等价：这里没有 SCF，没有 k 点网格，没
有 ENCUT 或 POTCAR，绝对能量属于产生它的模型，而不是某个 DFT 计算。`OUTCAR` 是给人和
读取 VASP 输出的工具看的报告；机器可读的事实在 JSON 里。

## 单点与弛豫

直接写在输出目录里：

```text
OUTCAR                  类似 VASP 的能量、力、应力报告
OSZICAR                 逐步的能量与力变化（仅弛豫）
CONTCAR                 最终结构（单点时是输入结构的副本）
mliport_results.json    结果，以及模型的身份
resolved_config.json    配置，以及每个值的来源
run.log                 运行的实时日志
```

`mliport_results.json` 把物理量放在 `calculation.results` 下（能量、每原子能量、力及其
统计、应力与压强），结构放在 `calculation.system` 下，弛豫的收敛记录放在
`calculation.optimization` 下（`converged`、`nsteps`、`fmax_threshold`）。

运行期间会锁住输出目录；如果目录里已有结果（早先的运行，或 VASP 自己的 `OUTCAR` /
`CONTCAR`），除非加 `--overwrite`，否则会拒绝使用。

## 分子动力学

MD 把无损数据和 VASP 风格的导出文件分开放：

```text
raw/trajectory.traj     标准 ASE 轨迹
raw/md.csv              逐帧标量：能量、温度等
raw/mliport_results.json
vasp/OUTCAR, vasp/CONTCAR, vasp/XDATCAR
artifacts.json          清单：文件、单位、可观测量定义、哈希
resolved_config.json
run.log
```

`artifacts.json` 明确写出单位和每个可观测量的含义，包括 `forces_raw_eV_A` 与
`forces_applied_eV_A`（施加约束之后），以及构型应力与总应力。任何一次运行的 ASE 轨迹
都可以用 `mliport convert-xdatcar` 转成标准 XDATCAR。

## NEB

```text
neb_results.json        能量、力、收敛状态、势垒数值
run_context.json        运行身份，包括 gradient_preflight
resolved_config.json
artifacts.json
checkpoints/            完整能带检查点（band.traj + checkpoint.json）
vasp_path/              每个像一个 POSCAR，供 VASP 风格的 NEB 工具使用
```

## 分析

`mliport analyze` 写在 `<run>/analysis/<task>/<analysis-id>/` 之下：JSON、CSV，以及在合适
时的 PNG 和 SVG 图，同时还有分析本身的溯源信息。详见[轨迹分析](analysis/index.md)。

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

## 溯源信息

`resolved_config.json` 对每个解析出来的键记录它的值和它来自哪一层（内置默认值、设置文
件、别名、profile、INCAR 或命令行）。凭它不用猜测就能复现或审查一次运行。

结果和上下文记录里还有：

- **模型身份**：`model_type`、路径、task、head、dtype、后端和框架版本，以及可用时的
  `model_sha256`；
- **设备**，包括请求的和实际用到的，已知时还有物理 GPU 的 UUID；
- **软件 commit**，即产生这次运行的 mliport 版本；
- **哈希**：输入结构、模型和产物的哈希，分析与验证证据依赖它们。

### 请求的设备与实际设备

请求 `cuda:0` 的运行，实际设备可能报告成另一个可见编号或一个 UUID。写方法部分时，请使
用记录下来的实际设备。如果它写的是 `unknown`，就不要为那条记录声称用了某块具体的
GPU。
