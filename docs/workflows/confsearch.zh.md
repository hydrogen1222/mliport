# 构型搜索

`mliport confsearch` 用来找**周期性 slab 一侧单个吸附分子**的低能构型。需要先装好对应后端
并准备模型，见[后端](../backends/index.md)和[安装](../installation.md)。程序会从多个起点各跑
一条 MD 链，把途中的结构做局部优化，去掉重复的极小值，最后导出一份候选结构，留给 DFT
复核。

它只做构型筛选：吸附能、自由能和 Boltzmann 分布都不在它的计算范围内，晶胞也保持不变。
结果里给出的是"这次搜索找到的能量最低的几个结构"，同时附上判断搜索是否充分的统计数据。
有限次数的搜索没法证明找到的就是全局最小。

```bash
.venv-mace/bin/mliport confsearch run starts/*.extxyz \
  --adsorbate-indices 300-341 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --cycles 5 --output runs/lds-on-lpsc
```

上面是模式 A，用你自己准备的结构作起点。模式 B 让程序把分子随机摆到 slab 上方：

```bash
.venv-mace/bin/mliport confsearch run \
  --slab slab.vasp --molecule conformers.xyz --n-starts 20 \
  --model mace-omat-0-medium.model --model-type mace --task bulk \
  --output runs/placed
```

`cs_adsorbate_indices` **和 ASE 一样从 0 开始编号**，CP2K 和 VMD 是从 1 开始的。编号填错是最
常见的问题，所以程序会检查：选中的原子必须能连成**一个完整的共价分子**，否则直接报错退出。

## 搜索过程

每个起点先做一次结构优化，这一步的结果就是"只做多起点优化"的对照组。然后每条 replica
重复 `--cycles` 轮"升温、保温、降温"：

- 用 Langevin 热浴升到 `CS_TEMPERATURE_HIGH` 并保温；
- 保温期间定时取几帧，然后线性降温到 `CS_TEMPERATURE_LOW`；
- 每个取出的帧和本轮最后一帧都在固定晶胞下用 FIRE 做局部优化；
- 对得到的每个极小值做分类和去重；
- 用 `CS_ACCEPT_TEMPERATURE` 下的 Metropolis 判据，决定是否把本轮最后的极小值作为下一轮的
  起点。

各条 replica 互不影响。程序轮流推进它们（先每条跑第 1 轮，再每条跑第 2 轮），所以任何时候
停下来都有一份可用的结果。默认 MD 期间 slab 原子全部固定（`CS_MD_MOBILE = adsorbate`）。
设成 `adsorbate_and_top` 会把表面以下 `CS_TOP_DEPTH` Å 内的原子也放开，设成 `all_free` 则放开
所有没被固定的原子。局部优化时，输入里没有固定的原子都会参与优化。

## 开始前的检查

只有当力等于能量的负梯度时，按能量排序才有意义。所以在跑任何 MD 之前，程序会先记录模型
信息，并在第一个起点上检查能量和力是否一致：

- 模型信息（后端、版本、task、head、精度、推理模式、设备、是否加了 DFT-D3）写进
  `run_context.json`；
- 如果模型声明自己的力是直接预测的（`direct_forces = true`），程序拒绝运行。确实要用的话加
  `--allow-direct-forces`，这时每条结果都会附上"排序不可靠"的警告；
- 其余模型用有限差分抽查：沿几个方向比较 `-F·u` 和能量的数值导数，结果存为
  `gradient_preflight`。

步长也会检查：可动原子里有氢时，步长不能超过 1.0 fs，默认是 0.5 fs。

## 自动摆放

模式 B 下，每个起点的生成步骤是：随机选一个构象，以几何中心为原点做一次随机旋转，在面内
随机选一个位置，再用二分法调整高度，直到分子和 slab 之间最近的一对原子距离达到设定值。距离
用"两原子共价半径之和"的倍数来表示：

| 选项 | 默认值 | 含义 |
|---|---|---|
| `CS_PLACE_CONTACT_SCALE` | 1.5 | 普通摆放：最近原子对贴近但不成键 |
| `CS_PLACE_ANCHOR` | 不设置 | 分子文件里的锚点原子编号（从 0 开始） |
| `CS_PLACE_ANCHOR_SCALE` | 1.1 | 设了锚点时，锚点和表面原子直接以成键距离摆放 |
| `CS_PLACE_OTHER_MIN_SCALE` | 1.3 | 锚点以外的原子离表面都要大于这个倍数 |
| `CS_PLACE_ANCHOR_TARGETS` | 不设置 | 限定锚点只和这几种表面元素成键 |

锚点摆放用于优先考虑化学吸附。程序会旋转分子，直到锚点原子处在最下方，再把它放到
`CS_PLACE_ANCHOR_SCALE` 倍共价半径和的位置。这个距离已经满足分类时的成键判据，所以起点
本身就是成键的。默认情况下锚点只和非离子原子成键，Li 这类离子不算。

## 去重

两个极小值的能量差不超过 `CS_DEDUP_ENERGY_TOL`，并且满足下面任意一条，就算作同一个结构：

- `G`，几何直接比对：slab 有固定原子作锚定，原子顺序也不变，所以可以逐个原子比较吸附物的
  位置，每个原子的位移都不超过 `CS_DEDUP_GEOM_TOL` 就算相同。平缓势能面上优化留下的微小
  差别由这一条处理；
- `D`，指纹比对：吸附物周围环境的 Oganov–Valle 指纹距离不超过 `CS_DEDUP_FP_TOL`。这一条用来
  识别对称等价的吸附位点，以及甲基转了一下这种本质相同的构象。

指纹的离散方式和 ASE 完全一致，可以逐个 bin 和 `OFPComparator` 对照。（它和
[NEB](neb.md) 里检查能量和力是否一致的预检是两回事。）容差按照"宁可多留一个重复，也不错删
一个结构"来设定：错把两个不同结构合并，可能正好删掉 DFT 复核后排第一的那个；多留一个重复，
代价只是多算一次 DFT。

**标定的局限。** 默认的 `CS_DEDUP_FP_TOL` 来自
`tests/mliport/mliport/confsearch/test_fingerprint_calibration.py` 里的标定。标定用的是
LGPS **体相**晶胞里的甲醇；"同一个极小值"那一类样本是随机扰动出来的，没有真正做优化；
"对称等价"那一类是严格的刚性对称拷贝，`D` 几乎为 0。所以这个容差能保证不会错误合并，但对
"带一点噪声的等价结构"还没有实测数据。实际使用中，`D` 基本只会合并完全对称的拷贝，同一
个极小值的合并主要靠 `G` 和 `CS_DEDUP_ENERGY_TOL`。这样做的后果是可能多出一些重复结构，不会
丢掉候选。在真实 slab 和真实模型上重新标定，是验证脚本
`validation/science/scripts/confsearch_suite.py` 要做的事。

## 输出

一次运行会写出 `resolved_config.json`、`run_context.json`、`run.log`、
`search_results.json`、`artifacts.json`、`starts/`、`raw/`、`minima/`、
`diagnostics/discovery.csv`、`top/rank_NNN/` 和 `checkpoints/`。各文件的通用约定见
[输出文件](../outputs.md)。

每个导出的结构目录里有 `POSCAR`（分数坐标，固定原子带 selective dynamics）、
`structure.extxyz`、`cp2k_coord.xyz`、`cp2k_cell.inc`、`cp2k_fixed_atoms.inc` 和
`minimum.json`。CP2K 的固定原子文件按 `&MOTION/&CONSTRAINT/&FIXED_ATOMS` 的 `LIST` 格式写，
原子编号从 1 开始，连续编号写成 `a..b`。导出时 slab 原子会放回晶胞内，吸附分子按最近镜像
拼成完整的分子，和原始极小值只差一个晶格平移。

`search_results.json` 里有排序结果、发现曲线、饱和程度的统计（最低结构被找到几次、被几条
replica 找到、第一次出现在搜索的什么阶段），以及和"只做多起点优化"对照组的比较。相关验证
记录见[验证](../validation.md)。

## 续算

```bash
.venv-mace/bin/mliport confsearch resume runs/lds-on-lpsc
```

起点优化完成后、以及每条 replica 每跑完一轮，都会写一次检查点。续算时程序会核对模型、输入
结构、原子分区、随机种子和各项参数（输出相关的设置除外），有任何一项变了就拒绝续算。核对
通过后，程序从中断前保存的随机数状态重跑被打断的那一轮。在 CPU 上，中断后续算的结果和一口
气跑完的结果逐位相同。被打断那一轮留下的原始文件会移到 `incomplete_attempt_N/`，不会删除。

## 风险和注意事项

- 基础模型大多只用体相无机材料训练，描述硫化物表面上的有机分子可能很差，温度越高问题越
  明显。`reacted`（分子骨架断键）的比例偏高就是信号。导出的候选一定要用 DFT 复核。
- 烷基链是平躺还是竖立，取决于色散作用。如果模型的训练数据不含色散，请加 `--d3`。D3 会记
  入模型信息，加 D3 和不加 D3 的搜索不能合并。
- MD 期间固定 slab，会让搜索偏向初始的表面结构。`adsorbate_and_top` 可以缓解，但能量会更
  嘈杂，注意诊断里的 `far_field_slab_changed`。
- 晶胞必须已经用同一个模型优化过。confsearch 不检查这一点，也不会改变晶胞。
- 分子与自身周期像的距离、真空层厚度的检查只是几何上的保护，不能说明模型在周期像之间完全
  没有相互作用。
- 短程 MLIP 不含长程静电。对单侧吸附的 slab 做 DFT 单点或优化时，记得加偶极修正。
- 模型没见过的表面上形成的化学吸附键，是整个结果里最不可靠的部分。每个标为 `chemisorbed`
  的候选都要用 DFT 确认。

Python API 和命令行一一对应，返回值就是 `search_results.json` 的内容：

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

终端界面（TUI）的配置页也支持这两种起点方式；`extend` 和 `merge` 只能在命令行里用。各个入口
分别支持哪些功能，记录在能力矩阵里。`mliport confsearch place` 只做摆放、不加载模型，可以在
花 GPU 时间之前先看看起点是否合理：

```bash
.venv-mace/bin/mliport confsearch place --slab slab.vasp \
  --molecule conformers.xyz --n-starts 20 --output runs/placement-check
```
