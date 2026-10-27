# 引用与致谢

mliport 只负责组织计算流程，势函数既不是它定义的，也不是它训练的。它调用的是别人的软件
（UMA/fairchem、MACE、DPA/DeePMD-kit、GRACE/tensorpotential、ASE、kinisi、GEMDAT）。发表用它算出
的结果时，最主要的是引用这些上游软件，特别是你实际用的模型。

## 引用 mliport

mliport **目前还没有 DOI**。在此之前：

- 请引用仓库 `https://github.com/hydrogen1222/mliport` 和你运行的版本（`mliport --version`）。
  [`CITATION.cff`](https://github.com/hydrogen1222/mliport/blob/main/CITATION.cff) 里的元数据可直接
  给文献管理软件使用；
- 上游项目的 DOI（例如 FAIRChem 的 Zenodo DOI、模型论文的 DOI）不能当作 mliport 的 DOI；
- 以后 mliport 有了 DOI，本页和 `CITATION.cff` 会同时更新。

## 需要引用的上游软件

### ASE

mliport 用 ASE 处理结构、计算器、优化器、MD 积分器和文件格式。

> A. H. Larsen et al., *The atomic simulation environment — a Python library for
> working with atoms*, J. Phys.: Condens. Matter **29**, 273002 (2017),
> doi:10.1088/1361-648X/aa680e. <https://wiki.fysik.dtu.dk/ase/>

### FAIRChem 与 UMA（`uma` 后端）

`uma` 后端通过 Meta FAIR 的 `fairchem-core` 加载 UMA 模型。请引用上游为你所用模型列出的 UMA 论文，
以及 fairchem 软件。

> <https://github.com/facebookresearch/fairchem> · <https://fair-chem.github.io/>

### MACE（`mace` 后端）

`mace` 后端通过 `mace-torch` 加载模型。请引用 MACE 论文，以及你所用模型对应的论文或发布页。

> I. Batatia et al., *MACE: Higher Order Equivariant Message Passing Neural
> Networks for Fast and Accurate Force Fields*, NeurIPS 2022, arXiv:2206.07697.
> <https://github.com/ACEsuit/mace>

### DeePMD-kit 与 DPA（`dpa` 后端）

`dpa` 后端通过 `deepmd-kit` 加载模型。

> H. Wang, L. Zhang, J. Han, W. E, *DeePMD-kit: A deep learning package for
> many-body potential energy representation and molecular dynamics*, Comput.
> Phys. Commun. **228**, 178–184 (2018), doi:10.1016/j.cpc.2018.03.016.
> <https://github.com/deepmodeling/deepmd-kit>

用 DPA-2 或 DPA-3 模型时，还要引用 DeePMD 项目为它发布的论文或技术报告。

### GRACE（`grace` 后端）

`grace` 后端通过 `tensorpotential` 加载导出的 SavedModel。请引用上游为你所用模型给出的 GRACE 论文。

> <https://github.com/ICAMS/grace-tensorpotential>

### kinisi

`mliport analyze transport` 用 kinisi 做示踪扩散和集体输运的贝叶斯分析（考虑了数据之间的相关性）。请
引用 kinisi 的论文。

> <https://github.com/bjmorgan/kinisi> · <https://kinisi.rtfd.io>

### GEMDAT

`mliport analyze electrolyte` 用 GEMDAT 分析位点占据和离子跳跃。请引用 GEMDAT 的论文。

> <https://github.com/GEMDAT-repos/GEMDAT> · <https://gemdat.readthedocs.io>

## 数据和模型的来源

- 精度测试用到的 OMat24 子集来自公开发布的 OMat24，具体来源记录在
  `validation/science/data_manifest.json`。
- 模型的上游名称和 SHA-256 记录在 `validation/science/model_manifest.json`。mliport 不分发模型文件。

## 发现遗漏

如果这里漏了某个上游项目、模型说明或许可证声明，或者信息已经过时，请提一个 issue。上游的版权和
许可证声明会一直保留在仓库里。
