# 构型搜索

`mliport confsearch` 在**周期性 slab 单侧的单个吸附物**上搜索低能构型。它需要明确后端上
的模型；引擎环境见[后端](../backends/index.md)，如何搭建见[安装](../installation.md)。它会跑
多条彼此独立的 MD 链，对找到的帧做淬火，对极小值去重，并导出可直接交给 DFT 复核的候选
清单。

它**不**计算吸附能、自由能或 Boltzmann 布居，不改变晶胞，也从不声称找到了全局最小。交
付物是一份筛选清单：**"目前找到的最低"，并附上搜索得有多充分的证据**。

```bash
.venv-mace/bin/mliport confsearch run starts/*.extxyz \
  --adsorbate-indices 300-341 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --cycles 5 --output runs/lds-on-lpsc
```

模式 A 使用你准备好的结构；模式 B 由程序把分子摆到 slab 上方：

```bash
.venv-mace/bin/mliport confsearch run \
  --slab slab.vasp --molecule conformers.xyz --n-starts 20 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --output runs/placed
```

`cs_adsorbate_indices` **和 ASE 一样从 0 开始**；CP2K 与 VMD 从 1 开始计数。选错索引是最
常见的错误，所以如果选中的不是**一个连通的共价分子**，程序会直接失败关闭，而不是去搜一
堆没有意义的东西。

## 搜索做了什么

每个起点先淬火，这已经给出"只做多起点弛豫"的基线。之后每条 replica 跑 `--cycles` 个
升温—保温—降温循环：

- 在 `CS_TEMPERATURE_HIGH` 下用 Langevin 热浴升温并保温，
- 在保温阶段取帧，并把链条按线性降到 `CS_TEMPERATURE_LOW`，
- 对每个快照和循环末帧做固定晶胞的 FIRE 淬火，
- 对每个极小值分类并去重，
- 用 `CS_ACCEPT_TEMPERATURE` 对循环末极小值做 Metropolis 接受/拒绝。

replica 之间彼此独立，round-robin 调度意味着每个循环结束后都有一份可用的答案。默认情况
下 MD 期间 slab 冻结（`CS_MD_MOBILE = adsorbate`）；`adsorbate_and_top` 还会把表面顶部
`CS_TOP_DEPTH` Å 内的原子放开，`all_free` 放开所有未被固定的原子。淬火永远弛豫所有未被
输入固定的原子。

## 搜索开始之前

只有当模型的力是能量的梯度时，排名才有意义。因此 mliport 在任何动力学之前先核对模型身
份，并在第一个准备好的起点上测量能量—力一致性：

- 模型身份（后端、版本、task、head、dtype、推理模式、设备以及任何 DFT-D3 修正）记录在
  `run_context.json`；
- 报告 `direct_forces = true` 的模型会被拒绝，除非你显式传
  `--allow-direct-forces`，此时每条结果都会带上"排名不可靠"的警告；
- 否则用有限差分抽查沿若干方向的 `-F·u` 与模型能量的比较，结果保存为
  `gradient_preflight`。

时间步也会检查：如果可动集合里有氢，超过 1.0 fs 会被拒绝。默认 0.5 fs。

## 自动摆放

模式 B 中，每个起点均匀随机挑一个构象，把它平移到质心，施加均匀随机旋转，抽取面内坐标，
然后用二分法求解高度，使最近的吸附物—slab 原子对的共价半径和比值达到目标：

| 选项 | 默认值 | 含义 |
|---|---|---|
| `CS_PLACE_CONTACT_SCALE` | 1.5 | 最近接触式起点（尚未成键） |
| `CS_PLACE_ANCHOR` | 未设置 | 分子文件中的 0 基锚点原子 |
| `CS_PLACE_ANCHOR_SCALE` | 1.1 | 给定锚点时，起点已经成键 |
| `CS_PLACE_OTHER_MIN_SCALE` | 1.3 | 其他所有原子对必须高于该比值 |
| `CS_PLACE_ANCHOR_TARGETS` | 未设置 | 要求最近的 slab 原子属于这些元素 |

锚点摆放就是"化学吸附优先"模式：旋转后分子的最低原子从锚点集合中抽取，高度把它放在
`CS_PLACE_ANCHOR_SCALE` 倍的共价半径和处，也就是已经落在分类器使用的成键判据之内。

## 去重

当两个极小值能量差不超过 `CS_DEDUP_ENERGY_TOL`，并且满足下面任一条时，它们被视为同一个：

- `G`，直接几何同一性：在固定原子锚定 slab、原子顺序固定的前提下，每个吸附物原子都与另
  一个极小值中的同一原子相距不超过 `CS_DEDUP_GEOM_TOL`（这吸收了平坦盆地里的淬火噪声）；
- `D`，聚焦的 Oganov–Valle 指纹距离不超过 `CS_DEDUP_FP_TOL`（用来捕捉对称等价位点，以及
  诸如甲基旋转这类等价原子置换的构象）。

指纹按 ASE 的离散化计算，因此可以逐 bin 与 `OFPComparator` 对照；这与
[NEB](neb.md) 里测量能量—力一致性的预检不是一回事。它的容差按**零误合并**标
定：误合并会悄悄删掉一个 DFT 可能排在第一的候选，而误拆分只多花一个 DFT 任务，所以默认
值刻意保守、宁多不误删。

## 输出

一次运行会写出 `resolved_config.json`、`run_context.json`、`run.log`、
`search_results.json`、`artifacts.json`、`starts/`、`raw/`、`minima/`、
`diagnostics/discovery.csv`、`top/rank_NNN/` 和 `checkpoints/`；通用约定见
[输出文件](../outputs.md)。

每个导出的 rank 包含 `POSCAR`（分数坐标，固定原子带 selective dynamics）、
`structure.extxyz`、`cp2k_coord.xyz`、`cp2k_cell.inc`、`cp2k_fixed_atoms.inc` 和
`minimum.json`。CP2K include 使用 `&MOTION/&CONSTRAINT/&FIXED_ATOMS` 的 `LIST` 约定：
1 基原子索引、`a..b` 区间，与 CP2K 手册一致。导出的几何把 slab wrap 进晶胞，并在最小
镜像约定下把吸附物补成整分子，因此它与极小值只差格矢平移。

`search_results.json` 报告排名、发现曲线、饱和诊断（命中次数、最佳极小值被多少条 replica
找到、首次发现的早晚）以及与"只做多起点弛豫"基线的对比。归档这类证据的 campaign 见
[验证](../validation.md)。请按"目前找到的最低"来读：有限搜索无法证明全局最小。

## 续跑

```bash
.venv-mace/bin/mliport confsearch resume runs/lds-on-lpsc
```

起点淬火阶段之后、以及每个（replica, cycle）单元之后都会写检查点。如果身份中任何东西变了
——模型、输入、区域、随机种子，或除输出/详细程度之外的任何选项——续跑会拒绝继续，并从中
断单元之前记录的随机状态重跑该单元。在 CPU 上，中断后续跑与不中断的运行逐位一致。中断尝
试留下的原始文件会被移到 `incomplete_attempt_N/`，绝不删除。

## 风险与注意事项

- 在无机体相上训练的基础模型可能对硫化物表面上的有机吸附物描述很差，高温会放大这一点。
  `reacted` 比例高就是症状；导出的候选请务必用 DFT 复核。
- 色散决定烷基链是平躺还是直立；当模型的参考数据不含色散时请用 `--d3`。该修正属于模型身
  份的一部分，因此带 D3 的搜索永远不会与不带 D3 的合并。
- MD 期间冻结 slab 会把探索偏向初始表面结构。`adsorbate_and_top` 缓解这一点，代价是能量
  更嘈杂；留意诊断里的 `far_field_slab_changed`。
- 晶胞必须已经是同一模型下弛豫好的晶格；confsearch 不检查这一点，也从不改变晶胞。
- 自像距离与真空距离是几何护栏，它们不能证明模型感受野在周期像之间已经解耦。
- 短程 MLIP 没有长程静电，因此单侧 slab 的 DFT 单点或弛豫需要偶极修正。
- 在模型没训练过的表面上形成的化学吸附键是结果里最不可信的部分；每个 `chemisorbed` 候选
  都要确认。

Python API 与 CLI 一一对应，返回 `search_results.json` 的内容：

```python
from mliport import run_confsearch

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

`extend` 与 `merge` 已列入计划但尚未实现，能力矩阵会如实记录。`mliport confsearch place`
只做摆放（不加载模型），方便你在花 GPU 时间之前先检查起点：

```bash
.venv-mace/bin/mliport confsearch place --slab slab.vasp \
  --molecule conformers.xyz --n-starts 20 --output runs/placement-check
```
