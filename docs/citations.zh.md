# 引用与致谢

mliport 是一个工作流层。它不定义也不训练势函数；它在同一套 CLI、TUI 和 Python API 后面驱动别人的
软件（UMA/fairchem、MACE、DPA/DeePMD-kit、GRACE/tensorpotential、ASE、kinisi、GEMDAT）。所以当你
发表用它算出的结果时，主要应当致谢的是上游：那些软件，尤其是你实际用的那个模型检查点。

## 引用 mliport

mliport **目前还没有自己的 DOI**。在有之前：

- 请引用仓库 `https://github.com/hydrogen1222/mliport` 和你运行的版本（`mliport --version`）。
  [`CITATION.cff`](https://github.com/hydrogen1222/mliport/blob/main/CITATION.cff) 里的元数据可直接
  给文献管理软件使用；
- 不要把上游的 DOI（FAIRChem 的 Zenodo DOI、某篇模型论文的 DOI）当作 mliport 的 DOI；
- 如果以后 mliport 有了 DOI，本页和 `CITATION.cff` 会一起更新。

## 上游该引用什么

### ASE

mliport 用 ASE 处理结构、计算器、优化器、MD 积分器和文件格式。

> A. H. Larsen et al., *The atomic simulation environment — a Python library for
> working with atoms*, J. Phys.: Condens. Matter **29**, 273002 (2017),
> doi:10.1088/1361-648X/aa680e. <https://wiki.fysik.dtu.dk/ase/>

### FAIRChem 与 UMA（`uma` 后端）

`uma` 后端通过 Meta FAIR 的 `fairchem-core` 加载 UMA 检查点。请引用上游项目为你所用检查点列出的
UMA 论文，以及 fairchem 软件。

> <https://github.com/facebookresearch/fairchem> · <https://fair-chem.github.io/>

### MACE（`mace` 后端）

`mace` 后端通过 `mace-torch` 加载检查点。请引用 MACE 论文以及你所用检查点的发布。

> I. Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
> Networks for Fast and Accurate Force Fields*, NeurIPS 2022, arXiv:2206.07697.
> <https://github.com/ACEsuit/mace>

### DeePMD-kit 与 DPA（`dpa` 后端）

`dpa` 后端通过 `deepmd-kit` 加载检查点。

> H. Wang, L. Zhang, J. Han, W. E, *DeePMD-kit: A deep learning package for
> many-body potential energy representation and molecular dynamics*, Comput.
> Phys. Commun. **228**, 178–184 (2018), doi:10.1016/j.cpc.2018.03.016.
> <https://github.com/deepmodeling/deepmd-kit>

对于 DPA-2 或 DPA-3 检查点，还请引用 DeePMD 项目为它给出的技术报告。

### GRACE（`grace` 后端）

`grace` 后端通过 `tensorpotential` 加载导出的 SavedModel。请引用上游项目为你的检查点给出的 GRACE
论文。

> <https://github.com/ICAMS/grace-tensorpotential>

### kinisi

`mliport analyze transport` 用 kinisi 做考虑协方差的贝叶斯示踪与集体输运分析。请引用上游项目给出的
kinisi 论文。

> <https://github.com/bjmorgan/kinisi> · <https://kinisi.rtfd.io>

### GEMDAT

`mliport analyze electrolyte` 用 GEMDAT 做位点占据与跳跃分析。请引用上游项目给出的 GEMDAT 论文。

> <https://github.com/GEMDAT-repos/GEMDAT> · <https://gemdat.readthedocs.io>

## 数据与模型的来源

- 精度工作负载用到的 OMat24 子集来自公开发布的 OMat24；它固定的来源记录在
  `validation/science/data_manifest.json`。
- 模型以上游标识加 SHA-256 的方式记录在 `validation/science/model_manifest.json`。mliport 不会分发权
  重。

## 如果缺了什么

如果这里缺少某个上游项目、模型卡或许可证声明，或者已经过时，请提一个 issue。上游的版权与许可证声
明绝不会从仓库中删除。
