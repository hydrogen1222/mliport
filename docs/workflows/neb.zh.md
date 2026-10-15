# NEB 与 CI-NEB

`mliport neb` 用固定晶胞的弹性带（NEB）在两个端点结构之间寻找最小能量路径，可选地
再用爬升像（climbing image）精修，并报告它找到的势垒。

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace --images 7 \
  --climb --output runs/neb
```

你需要提供两个原子顺序一致的结构（顺序不同时用明确的 `--atom-map`），一个在明确后
端上的模型，以及能容纳 `images + 2` 份计算器的内存。

## 能带动起来之前发生了什么

NEB 只有在模型的力恰好是能量的梯度时才有意义。否则，能带优化的依据就是一些并不描
述这个势能面的力，算出来的势垒毫无意义，却看不出任何异常。所以 mliport 会先检查。

在即将优化的能带上（两个端点和插值出来的像），它把沿几个方向（力的方向和随机方向）
的力投影 `-F·u`，与模型能量的中心有限差分作比较。两种步长用 Richardson 外推合
并，容差里还包含了模型 dtype 的舍入噪声。以下情况会拒绝运行：

- 力不是能量的梯度，
- 出现了非有限的值，
- 能量分辨率不足以下结论（比如较大的 float32 晶胞），
- 模型是直接预测力而不是对能量求导（UMA 就是这样，这类模型一律被拒绝）。

测量结果保存为 `run_context.json` 和 `neb_results.json` 中的 `gradient_preflight`。
它是在被采样的结构和方向上做的一致性抽查，不是对整个能量面的证明，也不说明模型有
多准。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--initial`、`--final` | 两个端点结构（除非是续算，否则必需） |
| `--images` | 中间像的个数，不含端点（默认 7） |
| `--climb` | 用爬升像精修鞍点（默认关闭） |
| `--fmax`、`--max-steps` | 收敛阈值 eV/Å（默认 0.03）和步数上限（默认 1000） |
| `--pre-fmax`、`--pre-max-steps` | 配合 `--climb`，爬升像启用前较宽松的第一阶段（默认 0.1 eV/Å，300 步） |
| `--spring`、`--maxstep` | 弹簧常数（默认 0.1 eV/Å²）和优化器最大步长（默认 0.1 Å） |
| `--interpolation` | 初始路径用 `linear`（默认）或 `idpp`；用 `--idpp-fmax`、`--idpp-steps`、`--idpp-mic` 调节 IDPP |
| `--path-convention` | `mic`（默认）或 `unwrapped` |
| `--endpoint-policy` | `validate`（默认）或 `relax`，见下文 |
| `--endpoint-fmax`、`--endpoint-steps` | 端点力阈值（默认 0.02 eV/Å）和弛豫步数预算（默认 500） |
| `--min-distance` | 原子间距小于该值的像会被拒绝（默认 0.5 Å） |
| `--atom-map` | 以初始原子顺序给出的 0 起始末态索引，逗号分隔 |
| `--image-shifts` | 每个原子的整数晶格平移，例如 `0,0,0;1,0,0` |
| `--checkpoint-interval` | 每 N 步保存一次整条能带（默认 10） |
| `--resume` | 从检查点或 NEB 运行目录继续 |

### 端点：校验还是弛豫

默认的 `--endpoint-policy validate` 要求端点本身已经弛豫好：任何一端的力超过
`--endpoint-fmax`，mliport 就会停下，并告诉你是哪一端。`relax` 则先弛豫两个端点，
再构建能带。无论哪种，端点的处理方式都会改变你的参考，请记录下来。`relax` 不能改写
一条显式提供的能带。

## 会得到什么

`neb_results.json` 里有能量、力、收敛状态和势垒数据（`barrier_forward_sampled_eV`、
`barrier_reverse_sampled_eV`、`reaction_energy_eV`，以及说明是否找到内部峰值的
`barrier_status`）。旁边是 `run_context.json`、`resolved_config.json`、
`artifacts.json`，装着完整能带检查点的 `checkpoints/` 目录，以及每个像一个 `POSCAR`
的 `vasp_path/` 目录，供期待 VASP NEB 目录布局的工具使用。

这个势垒是*采样*得到的：能带上最高的像减去端点。mliport 不拟合鞍点，拟合字段留空。

## 失败或没收敛时

没有收敛的能带会如实报告，附带目前最好的能带和原因，它不是一个收敛的势垒。原子映
射不对或端点不一致，会在构建能带之前就失败。续算会锁定原来的运行目录，所以不会不
小心用不同的设置重新开始。

## 势垒的价值

- 收敛的 NEB 鞍点候选并不是被验证过的过渡态。用 Hessian（频率）计算去核实它是一项
  独立的分析，能带收敛并不意味着已经做了这一步。
- 势垒只在同一模型与 head 身份、同一 NEB 协议（像数、插值、端点策略、弹簧）之内
  才可比。
- 只有当端点确实是你想要的状态、原子映射在化学上正确时，它才有意义。

## 哪个入口能表达什么

不是每个 NEB 选项都能从每个界面设置。下表由
`validation/science/capability_matrix.json` 生成。

<!-- BEGIN GENERATED: neb-capability -->
| 功能 | API | 直接 CLI | INCAR | TUI |
|---|---|---|---|---|
| endpoints | yes | yes | yes | yes |
| atom map | yes | yes | no | yes |
| image shifts | yes | yes | no | yes |
| resume | yes | yes | no | yes |
| path convention | yes | yes | yes | yes |
| interpolation | yes | yes | yes | yes |
| method climb | yes | yes | yes | yes |
| pre relaxation | yes | yes | yes | yes |
| min distance guard | yes | yes | yes | yes |
<!-- END GENERATED -->

`yes` 表示该入口能表达这个功能，它不代表等价性或验证声明。`no` 或 `partial` 在那个
文件里带有说明，而且不会有任何静默转换：例如原子映射和像平移，只能通过 API、直接
命令行和 TUI 使用，不能通过 INCAR 文件。

## 续算

```bash
.venv-mace/bin/mliport neb --resume runs/neb --output runs/neb
```

续算在原来的运行目录里继续，并检查记录下来的身份：模型、task、head 与 dtype，晶胞与
PBC，约束，能带设置和各选项。在这个 beta 版本中，步数预算是身份的一部分，所以改
了 `--max-steps` 的续算会被以 `Resume fingerprint is incompatible` 拒绝。发生这种情
况时，检查点保持原样。
