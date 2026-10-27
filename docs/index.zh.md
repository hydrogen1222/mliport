# mliport

mliport 让你用同一套命令行、终端界面和 Python API 运行机器学习原子间势——
MACE、DPA、GRACE 和 UMA。你给它一个结构和一个模型文件，它就能做结构弛豫、
分子动力学、用 NEB 求反应势垒、搜索 slab 上吸附物的低能构型，或者分析一条轨迹。结果写成用过 VASP 的人一眼
就认得的文件：`OUTCAR`、`OSZICAR`、`CONTCAR`、`XDATCAR`，旁边还有记录"这次到底
跑了什么"的 JSON。

它是一层工作流工具，不是模型，不是训练器，也不是 DFT 程序：文件格式长得像
VASP，但这里没有 SCF，没有 k 点网格，也没有 POTCAR，而且绝对能量只在产生它的
那个模型内部才有意义。势函数对你的体系"对不对"，取决于你选的模型文件和你自己
做的检验。

## 它替你做了什么

- **一套流程，四个后端。** 单点、弛豫（固定或可变晶胞）、NVE/NVT 分子动力学、
  NEB 与 CI-NEB、构型搜索、批量扫描和任务队列，不管底层是哪个模型，用法都一样。
- **不偷偷猜。** 没有默认后端。拼错的选项、或者属于别的后端的选项会直接让运行
  停下，而不是被悄悄忽略。多 head 的 DPA 模型会要求你说明用哪一个。
- **留下可追溯的记录。** 每次运行都会记下模型文件及其哈希、最终生效的配置
  （以及每个值来自哪一层）、实际使用的设备和 mliport 版本。
- **数据太短时，分析会直说。** MSD、输运、RDF、VACF、GEMDAT 电解质分析和
  Arrhenius 拟合在采样不足时会明确报告"采样不足"，而不是给你一个不该信的数。

## 先试一下

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto

.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model --model-type mace --task bulk
```

[快速上手](quickstart.md)会把这几行一句一句讲清楚，包括模型文件去哪里拿。

## 接下来看哪里

| 你想… | 去看 |
|---|---|
| 安装并跑第一个计算 | [快速上手](quickstart.md) |
| 弄清环境、Python 版本和 GPU | [安装](installation.md) |
| 在答案已知的体系上检验一个模型 | [测试体系](test-systems.md) |
| 选模型，或看哪些模型被验证过 | [模型](models.md)、[后端](backends/index.md) |
| 用命令行、INCAR 或 `settings.ini` 设置选项 | [配置](configuration.md) |
| 做弛豫、MD、NEB、构型搜索或批量计算 | [工作流](workflows/index.md) |
| 分析轨迹 | [轨迹分析](analysis/index.md) |
| 了解每个输出文件的内容 | [输出文件](outputs.md) |
| 看实际测过什么 | [验证](validation.md) |
| 排查报错 | [故障排查](troubleshooting.md) |
| 从旧名字 `mlipx` 迁移过来 | [迁移](migration-from-mlipx.md) |
| 引用 mliport 和它运行的模型 | [引用](citations.md) |
| 参与 mliport 开发 | [开发](development.md) |
