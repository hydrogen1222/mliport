# 测试体系

在你自己的问题上信任一个模型之前，先在答案已知的体系上跑一下。本页列出几个经典算例、各自
检验什么，以及该和什么参考值比较。它们是建议，不属于 mliport 的验证记录：数字来自所引文献
或本仓库自己的[验证](validation.md)，你的结果取决于你用的检查点。

基础模型复现的是它的*参考泛函*（大多是 PBE / PBE+U，见[后端](backends/index.md)），而不是
实验。先和同一水平的 DFT 比；和实验比，只是为了知道这个水平离真实有多远。

## 1. 晶格常数（Si、Cu）

最便宜的健全性检查，也是后面每个算例的第一步：对体相晶体做变晶胞弛豫。

```bash
.venv-mace/bin/mliport opt Si.cif --model model.model --model-type mace \
  --task bulk --cell-opt --fmax 0.01 --device cuda --output runs/si-opt
```

把 `CONTCAR` 里弛豫后的晶格常数，和 [Materials Project](https://next-gen.materialsproject.org/)
上同一结构的 PBE 值比较（金刚石 Si 是 `mp-149`，fcc Cu 是 `mp-30`）。偏差在百分之几埃量级
是常见的；偏差很大说明这个模型不适合这种化学体系。后面所有超胞都用模型自己的晶格常数，
以免从一个受应变的晶胞出发。

## 2. Cu 中的空位跃迁（NEB）

fcc Cu 中的一个空位，以及一个最近邻原子跳进去。路径短而对称，所以它检验的更多是 NEB 流程
本身，而不是模型。

```python
import numpy as np
from ase.build import bulk
from ase.geometry import get_distances
from ase.io import write

a = 3.63  # replace with the lattice constant from recipe 1
cell = bulk("Cu", "fcc", a=a, cubic=True).repeat((3, 3, 3))  # 108 atoms
site = cell.positions[0].copy()
initial = cell.copy()
del initial[0]  # vacancy at the origin
_, distances = get_distances(site, initial.positions, cell=initial.cell, pbc=True)
neighbour = int(np.argmin(distances[0]))  # a nearest neighbour of the vacancy
final = initial.copy()
final.positions[neighbour] = site  # it jumps into the vacant site
write("initial.vasp", initial)
write("final.vasp", final)
```

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --images 5 --climb --model model.model --model-type mace --task bulk \
  --device cuda --output runs/cu-vac-neb
```

正向势垒应当等于逆向势垒（路径对称）。验证中四个示例检查点给出 0.735–0.762 eV
（[验证报告](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md)）。

## 3. Li10GeP2S12 中的 Li 扩散（MD、输运、Arrhenius）

这个超离子导体的结构随仓库提供（`examples/structures/li10gep2s12_primitive.vasp`），它检验
长时间 MD 和完整的输运分析链条。请用超胞；50 个原子的晶胞对扩散来说太小。

```python
from ase.io import read, write

atoms = read("examples/structures/li10gep2s12_primitive.vasp")
write("LGPS_221.vasp", atoms.repeat((2, 2, 1)), direct=True)  # 200 atoms
```

```bash
for T in 600 700 800 900; do
  .venv-mace/bin/mliport md LGPS_221.vasp --model model.model --model-type mace \
    --task bulk --ensemble NVT --temp $T --timestep 1.0 \
    --equilibration-steps 10000 --steps 100000 --save-interval 100 \
    --device cuda --output runs/lgps-$T
  .venv-mace/bin/mliport analyze runs/lgps-$T transport --mobile Li --charge 1
done
```

然后用 `analyze ... arrhenius` 拟合这四个扩散系数（见 [Arrhenius](analysis/arrhenius.md)）。
作为参考，第一性原理 MD（PBE，600–1500 K，40 ps）给出活化能 0.21 eV、外推室温电导率约
9 mS/cm（范围 2–40），实验值为 0.24 eV 和 12 mS/cm
（[Mo、Ong 与 Ceder，Chem. Mater. 2012](https://doi.org/10.1021/cm203303y)）。MLIP 能跑比那项
研究长得多、大得多的模拟，所以应当期待更小的误差棒，而不一定是同一个数。

## 4. Pt(111) 上的 CO：模型继承它的泛函

低覆盖度下，实验中 CO 吸附在 Pt(111) 的顶位，而收敛的 GGA 计算更偏好空位
（[Feibelman 等，J. Phys. Chem. B 2001](https://doi.org/10.1021/jp002302t)）。用 GGA 数据训
练的模型应当复现 GGA 的答案。在四层、底部两层固定的 Pt(111) slab 上分别弛豫顶位和 fcc 空位
的 CO，比较两个能量。

```bash
.venv/bin/mliport opt CO_top.vasp --model uma-s-1p2.pt --model-type uma \
  --task oc20 --fmax 0.02 --device cuda --output runs/co-top
.venv/bin/mliport opt CO_fcc.vasp --model uma-s-1p2.pt --model-type uma \
  --task oc20 --fmax 0.02 --device cuda --output runs/co-fcc
```

如果空位能量更低，说明模型在正常工作：它复现了 GGA，连同 GGA 的误差。这是最清楚的提醒：
MLIP 不可能比它的训练数据更正确。

## 5. slab 上的小分子（构型搜索）

Li10GeP2S12 切出的 slab 上的甲醇，是[构型搜索](workflows/confsearch.md)验证脚本
（`validation/science/scripts/confsearch_suite.py`）使用的体系。在算大分子之前先跑一次：它在
相同起点下比较"随机摆放后只做弛豫"和完整搜索，报告 MD 循环有没有找到更低的结构，以及去重
在真实极小值上的表现。

```bash
.venv-mace/bin/mliport confsearch run --slab slab.vasp --molecule methanol.xyz \
  --n-starts 10 --cycles 3 --dry-run \
  --model model.model --model-type mace --task bulk --output runs/meoh-dry
```

输入和计算量看起来没问题后，去掉 `--dry-run` 正式运行。

## 需要记录什么

每个测试都要保留模型文件及其 SHA-256（mliport 两者都会记录）、后端版本、task 或 head、精度
和设备。只有这些都相同时，两个数才可比较；见[模型](models.md)。
