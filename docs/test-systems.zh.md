# 测试体系

把模型用到自己的课题之前，最好先在答案已知的体系上试一下。本页列出几个经典算例，说明各自检验
什么、该和哪个参考值比较。这些只是推荐，不属于 mliport 的验证记录：参考数字来自所引文献或本仓库
的[验证](validation.md)，你算出的结果还取决于所用的模型文件。

基础模型拟合的是训练数据所用的*泛函*（大多是 PBE / PBE+U，见[后端](backends/index.md)），目标并
非实验值。所以先和同一泛函的 DFT 结果比；和实验比，是为了了解这个泛函本身的误差有多大。

## 1. 晶格常数（Si、Cu）

这是最省事的检查，也是后面每个算例的第一步：对体相晶体做变晶胞优化。

```bash
.venv-mace/bin/mliport opt Si.cif --model model.model --model-type mace \
  --task bulk --cell-opt --fmax 0.01 --device cuda --output runs/si-opt
```

把 `CONTCAR` 里优化后的晶格常数和 [Materials Project](https://next-gen.materialsproject.org/)
上同一结构的 PBE 值比较（金刚石 Si 是 `mp-149`，fcc Cu 是 `mp-30`）。相差百分之几埃很常见；如果
差得很多，说明这个模型不适合这类体系。后面搭超胞时都用模型自己优化出的晶格常数，这样初始晶胞没有
额外的应变。

## 2. Cu 中的空位跃迁（NEB）

在 fcc Cu 里挖一个空位，让一个最近邻原子跳进去。路径短而且对称，主要用来检查 NEB 流程本身，对
模型的考验不大。

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

由于路径对称，正向和逆向势垒应当相等。仓库验证中四个模型给出 0.735–0.762 eV
（[验证报告](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md)）。

## 3. Li10GeP2S12 中的 Li 扩散（MD、输运、Arrhenius）

这个超离子导体的结构就在仓库里（`examples/structures/li10gep2s12_primitive.vasp`），用来检验长时间
MD 和整套输运分析。要用超胞，50 个原子的晶胞算扩散太小了。

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

然后用 `analyze ... arrhenius` 拟合这四个温度的扩散系数（见 [Arrhenius](analysis/arrhenius.md)）。
参考值：第一性原理 MD（PBE，600–1500 K，40 ps）得到活化能 0.21 eV，外推的室温电导率约 9 mS/cm
（范围 2–40）；实验值是 0.24 eV 和 12 mS/cm
（[Mo、Ong 与 Ceder，Chem. Mater. 2012](https://doi.org/10.1021/cm203303y)）。MLIP 可以跑更长、更大
的体系，误差棒应该比那项研究小，数值本身则不一定相同。

## 4. Pt(111) 上的 CO：模型继承它的泛函

低覆盖度下，实验观察到 CO 吸附在 Pt(111) 的顶位，而收敛的 GGA 计算却得出空位更稳定
（[Feibelman 等，J. Phys. Chem. B 2001](https://doi.org/10.1021/jp002302t)）。用 GGA 数据训练的模型
应该给出和 GGA 一样的结论。在四层 Pt(111) slab（底部两层固定）上分别优化顶位和 fcc 空位的 CO，比较
两者的能量。

```bash
.venv/bin/mliport opt CO_top.vasp --model uma-s-1p2.pt --model-type uma \
  --task oc20 --fmax 0.02 --device cuda --output runs/co-top
.venv/bin/mliport opt CO_fcc.vasp --model uma-s-1p2.pt --model-type uma \
  --task oc20 --fmax 0.02 --device cuda --output runs/co-fcc
```

如果空位能量更低，说明模型正常：它复现了 GGA 的结果，也继承了 GGA 的误差。这个例子很直观地说明，
MLIP 的准确度上限就是它的训练数据。

## 5. slab 上的小分子（构型搜索）

Li10GeP2S12 slab 上的甲醇是[构型搜索](workflows/confsearch.md)验证脚本
（`validation/science/scripts/confsearch_suite.py`）用的体系，建议在算大分子之前先跑一遍。脚本在相同
起点下比较"随机摆放后直接优化"和完整搜索，看 MD 循环能否找到更低的结构，也检查去重在真实极小值上
是否合理。

```bash
.venv-mace/bin/mliport confsearch run --slab slab.vasp --molecule methanol.xyz \
  --n-starts 10 --cycles 3 --dry-run \
  --model model.model --model-type mace --task bulk --output runs/meoh-dry
```

确认输入和计算量没问题后，去掉 `--dry-run` 正式运行。

## 需要记录什么

每次测试都要记下模型文件和它的 SHA-256（mliport 会自动记录）、后端版本、task 或 head、精度和设备。
这些都相同时，两次结果才能直接比较，见[模型](models.md)。
