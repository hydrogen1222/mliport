# 构型搜索

`mliport confsearch` 搜索**一个吸附分子在周期性 slab 一侧**的低能构型。它通过在升高温度下运行多条独立的 MD 链，对发现的结构做淬火处理、去重，然后导出排序后的候选列表供 DFT 复核。

## 它做什么——以及不做什么

构型搜索是一个**筛选**工具。它报告自己找到的最低能量极小值，并给出搜索充分程度的统计证据，但它从不声称找到了全局最小值。它不计算吸附能、自由能或 Boltzmann 分布，也从不改变晶胞。最终产出是一份排好序的候选列表，交给 DFT 做进一步排序。

## 两种启动模式

**模式 A — 预先准备好的起点：** 你提供已构建好的 slab+吸附分子结构：

```bash
mliport confsearch run starts/*.extxyz \
  --adsorbate-indices 300-341 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --cycles 5 --output runs/lds-on-lpsc
```

计算子命令 `confsearch run`、`resume` 和 `extend` 均支持 `--device cpu|cuda|cuda:N`。
省略时采用已有配置（默认 CPU）。续算或扩展时仍会验证已记录的模型和设备身份，
不允许静默切换 GPU 或退回 CPU。

上面是模式 A，用你自己准备的结构作起点。模式 B 让程序把分子随机摆到 slab 上方：

```bash
mliport confsearch run \
  --slab slab.vasp --molecule conformers.xyz --n-starts 20 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --output runs/placed
```

你可以在不加载模型的情况下预览放置结果：

```bash
mliport confsearch place --slab slab.vasp \
  --molecule conformers.xyz --n-starts 20 --output runs/placement-check
```

## Python API

```python
from mliport.api import run_confsearch

results = run_confsearch(
    starts=["starts/start_000.extxyz"],
    adsorbate_indices="300-341",
    model_path="mace-omat-0-medium.model",
    model_type="mace",
    task="bulk",
    cycles=5,
    output_dir="runs/lds-on-lpsc",
    verbose=False,
)
print(results["ranking"]["best_energy_eV"])
```

该函数返回 `search_results.json` 的内容作为字典。

## 搜索过程

每个起点首先做一次淬火（固定晶胞下的仅原子位置 FIRE 优化），得到"仅多起点优化"的基线。然后每条 replica 运行 `--cycles` 轮加热–保温–冷却循环：

1. 在 `CS_TEMPERATURE_HIGH` 下用 Langevin 恒温器**加热并保温**
2. 在保温阶段**采样快照**，然后线性降温到 `CS_TEMPERATURE_LOW`
3. 对每个快照和本轮末帧在固定晶胞下用 FIRE 做**淬火**
4. 对每个极小值做**分类和去重**
5. 在 `CS_ACCEPT_TEMPERATURE` 下做 **Metropolis 判断**，决定是否接受本轮末尾的极小值作为下一轮的起点

各 replica 独立运行，轮转调度意味着搜索在每轮结束后都有可用的结果。

### slab 原子的活动性

默认情况下 MD 期间 slab 被冻结（`CS_MD_MOBILE = adsorbate`）：

- `adsorbate`：仅吸附分子在 MD 中运动（默认）
- `adsorbate_and_top`：同时解冻表面以下 `CS_TOP_DEPTH` Å 范围内的原子
- `all_free`：解冻输入中所有未被固定的原子

淬火步骤中，无论 `CS_MD_MOBILE` 如何设置，所有未被固定的原子都参与优化。

## 搜索前检查

在任何动力学开始之前，mliport 会验证模型：

- **模型身份**（后端、版本、task、head、dtype、推理模式、设备以及任何 DFT-D3 校正）记录在 `run_context.json` 中。
- 声明 `direct_forces = true` 的模型会被拒绝，除非你传入 `--allow-direct-forces`；此时每个结果都附带排序不可靠的警告。
- 在第一个预备起点上运行**梯度预检**（能量-力一致性的有限差分检查）。
- 如果可活动原子中有氢，超过 1.0 fs 的步长会被拒绝。默认值为 0.5 fs。

## 自动放置（模式 B）

每个起点随机选择一个构象异构体，将其居中，施加均匀随机旋转，抽取面内坐标，然后通过二分法求解高度，使最近的吸附分子–slab 原子对达到共价半径之和的目标比例。

| 选项 | 默认值 | 含义 |
|---|---|---|
| `CS_PLACE_CONTACT_SCALE` | 1.5 | 非键合放置的接触距离比例 |
| `CS_PLACE_ANCHOR` | 未设置 | 分子文件中的锚点原子索引（0-based） |
| `CS_PLACE_ANCHOR_SCALE` | 1.1 | 有锚点时，直接以成键距离放置 |
| `CS_PLACE_OTHER_MIN_SCALE` | 1.3 | 其他所有原子对必须超过此比例 |
| `CS_PLACE_ANCHOR_TARGETS` | 未设置 | 要求最近的 slab 原子为这些元素之一 |

**锚点放置**是化学吸附优先模式：旋转后分子最低位置的原子从锚点集合中选取，高度设置为成键距离（在分类器使用的成键判据范围内）。

## 去重

两个极小值在以下条件同时满足时被视为同一结构：能量差不超过 `CS_DEDUP_ENERGY_TOL`，且满足下列任一条件：

- **G（几何同一性）：** 固定原子为 slab 提供锚定，吸附分子中每个非氢原子在两个极小值中的距离不超过 `CS_DEDUP_GEOM_TOL`（吸收平坦势能盆中淬火噪声）。氢原子不参与比较，因为甲基、亚甲基的扭转势能面几乎是平的，淬火在力达到阈值时就会停下，H 的转角是任意的。吸附分子如果只有氢原子，则比较全部原子。
- **D（指纹距离）：** 聚焦的 Oganov–Valle 指纹距离不超过 `CS_DEDUP_FP_TOL`（捕捉对称等价位点和排列等价的构象，如旋转的甲基）。

容差被有意设为保守值：**错误合并**会默默删掉一个 DFT 复核后可能排第一的候选，而错误分裂只多付出一次重复的代价。

## 吸附分子索引

`cs_adsorbate_indices` 使用 **0-based 索引**（与 ASE 一致）。CP2K 和 VMD 从 1 开始计数。索引填错是最常见的错误——选中的原子如果不能构成一个连通的共价分子，程序会直接报错而不是继续搜索。

## 输出文件

一次运行会写出：

| 文件 | 内容 |
|---|---|
| `search_results.json` | 排名、发现曲线、饱和度诊断 |
| `run_context.json` | 模型身份和梯度预检 |
| `resolved_config.json` | 完整的已解析配置 |
| `artifacts.json` | 文件列表及哈希 |
| `run.log` | 人类可读的日志 |
| `starts/` | 预备/放置的起始结构 |
| `raw/` | 原始 MD 轨迹 |
| `minima/` | 去重后的极小值 |
| `diagnostics/discovery.csv` | 每轮发现统计 |
| `top/rank_NNN/` | 导出的候选（见下文） |
| `checkpoints/` | 每条 replica 每轮的检查点 |

每个导出的排名目录包含：

- `POSCAR` — 分数坐标，固定原子带 selective dynamics
- `structure.extxyz` — ASE 可读的结构
- `cp2k_coord.xyz`、`cp2k_cell.inc`、`cp2k_fixed_atoms.inc` — CP2K 输入片段（1-based 原子索引，连续编号用 `a..b` 范围表示）
- `minimum.json` — 该极小值的元信息

## 续算

```bash
mliport confsearch resume runs/lds-on-lpsc
```

在起点淬火阶段完成后以及每个 (replica, cycle) 单元完成后都会写入检查点。续算时会检查身份是否发生变化——模型、输入、区域划分、种子以及除输出和详细程度外的所有选项。在 CPU 上，中断后续算的结果与一次不间断运行按位一致。被中断尝试的原始文件被移动到 `incomplete_attempt_N/`，不会被删除。

## 风险和注意事项

- **模型质量：** 以无机 bulk 数据训练的基础模型可能对硫化物表面上的有机吸附分子描述不佳。高温会放大这个问题。`reacted`（分子断键）比例过高是症状——请用 DFT 复核导出的结构。
- **色散：** 色散作用决定了烷基链是平躺还是竖立。当模型的训练数据缺乏色散时请使用 `--d3`。D3 校正是模型身份的一部分，带 D3 的搜索不能与不带 D3 的合并。
- **冻结 slab 偏差：** MD 期间冻结 slab 会使搜索偏向初始表面几何结构。`adsorbate_and_top` 可以缓解这一点，但代价是更嘈杂的能量；请检查诊断中的 `far_field_slab_changed`。
- **晶胞质量：** 晶胞必须已经是同一模型的优化结果；confsearch 不检查这一点，也从不改变晶胞。
- **周期性：** 自映像和真空距离的保护是几何性的。它们不能证明模型的感受野在周期像之间完全解耦。
- **静电：** 短程 MLIP 没有长程静电。对单侧 slab 做 DFT 单点计算时需要偶极校正。
- **化学吸附：** 在模型未经训练的表面上形成的化学键是结果中最不可靠的部分。请用 DFT 确认每个标记为 `chemisorbed` 的候选。
