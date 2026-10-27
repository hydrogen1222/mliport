# mliport

mliport 用来运行机器学习原子间势（MACE、DPA、GRACE、UMA），四种模型共用同一套命令行、
终端界面和 Python API。给它一个结构和一个模型文件，它可以做结构优化、分子动力学、NEB
势垒计算、slab 上吸附分子的构型搜索，也可以分析 MD 轨迹。输出文件沿用 VASP 的格式
（`OUTCAR`、`OSZICAR`、`CONTCAR`、`XDATCAR`），另外附带 JSON 文件，记录这次计算具体用了
什么设置。

mliport 只负责组织计算流程，本身不包含模型，不训练模型，也不做 DFT。输出文件只是格式和
VASP 一样，里面没有 SCF、k 点和 POTCAR。能量的绝对值只在同一个模型内部有比较意义。模型
适不适合你的体系，要看你选的模型文件，以及你自己做的检验。

## 主要功能

- **四个后端，用法一致。** 单点、结构优化（固定晶胞或变晶胞）、NVE/NVT 分子动力学、NEB
  和 CI-NEB、构型搜索、批量计算和任务队列，换哪个模型命令都一样。
- **不做默认猜测。** 没有默认后端。选项拼错，或者用了别的后端的选项，程序会直接报错停下。
  DPA 的多任务模型必须指定用哪个 head。
- **完整的计算记录。** 每次运行都会记录模型文件和它的哈希值、最终生效的配置（以及每个值
  来自哪里）、实际使用的设备和 mliport 版本。
- **数据不够时如实报告。** MSD、输运、RDF、VACF、GEMDAT 电解质分析和 Arrhenius 拟合在
  采样不足时会报告"采样不足"，不会硬给出一个不可信的数。

## 快速试用

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto

.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model --model-type mace --task bulk
```

[快速上手](quickstart.md)逐行解释了这几条命令，也说明了模型文件从哪里下载。

## 文档导航

| 你想… | 去看 |
|---|---|
| 安装并跑第一个计算 | [快速上手](quickstart.md) |
| 了解环境、Python 版本和 GPU | [安装](installation.md) |
| 在答案已知的体系上检验模型 | [测试体系](test-systems.md) |
| 选模型，或查看哪些模型验证过 | [模型](models.md)、[后端](backends/index.md) |
| 用命令行、INCAR 或 `settings.ini` 设置参数 | [配置](configuration.md) |
| 做结构优化、MD、NEB、构型搜索或批量计算 | [工作流](workflows/index.md) |
| 分析轨迹 | [轨迹分析](analysis/index.md) |
| 了解各个输出文件 | [输出文件](outputs.md) |
| 查看实际测试过哪些内容 | [验证](validation.md) |
| 排查报错 | [故障排查](troubleshooting.md) |
| 从旧名字 `mlipx` 迁移 | [迁移](migration-from-mlipx.md) |
| 引用 mliport 和所用模型 | [引用](citations.md) |
| 参与 mliport 开发 | [开发](development.md) |
