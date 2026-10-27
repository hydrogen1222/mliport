# 快速上手

这一页大约需要十分钟：装好一个后端，跑一次单点和一次简短的结构优化，并知道结果在哪里看。
例子用 MACE，因为它的模型文件公开且不大。其他后端放在最后，不需要四个都装。

## 1. 安装一个后端

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto
```

安装脚本会给这个后端单独建一个 Python 环境（`.venv-mace`），把 mliport 装进去，在 `bin/`
里生成一个启动脚本，最后运行一次 `mliport doctor`。没有 NVIDIA GPU 的机器用
`--device cpu`。离线安装、一次装多个后端、手动安装等情况见[安装](installation.md)。

## 2. 检查安装

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

如果环境跑不了这个后端，`doctor` 会以非零状态退出。有检查项失败时先解决它，常见原因见
[故障排查](troubleshooting.md)。

## 3. 准备模型和结构

需要一个周期性结构（CIF、POSCAR 或 XYZ）和一个 MACE 模型文件。mliport 不自带模型，也不
会自动下载。请从上游发布页下载 `mace-omat-0-medium.model`（链接见[模型](models.md)），记住
保存位置。

手头没有结构的话，仓库里带了一个 50 个原子的 Li₁₀GeP₂S₁₂（LGPS）晶胞，路径是
`examples/structures/li10gep2s12_primitive.vasp`。这是真实的周期性晶胞，CPU 也能跑。

## 4. 单点计算

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --output runs/mace-sp
```

运行时会打印能量和力的摘要，并在 `runs/mace-sp/` 下生成 `OUTCAR`、`OSZICAR`、`CONTCAR`、
`mliport_results.json`、`resolved_config.json` 和 `run.log`。各文件的内容见
[输出文件](outputs.md)。

如果 `runs/mace-sp` 里已经有结果，程序会拒绝写入。确定要覆盖时加 `--overwrite`。

## 5. 结构优化

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --fmax 0.05 --max-steps 50 \
  --output runs/mace-opt
```

50 步可能不够收敛，结果里会如实记录。是否收敛写在 `mliport_results.json` 里，没收敛就标
为没收敛。优化后的结构在 `CONTCAR` 里。

## 6. 读取结果

```python
import json

with open("runs/mace-opt/mliport_results.json") as handle:
    record = json.load(handle)

calculation = record["calculation"]
print(calculation["optimization"]["converged"], calculation["results"]["energy"])
```

同一目录下还有复现这次计算所需的全部信息：模型信息、每个参数的来源、请求的设备和实际使用
的设备、mliport 版本。

## 其他后端

用 `--engines dpa`、`grace` 或 `uma` 安装其他后端，然后用各自的环境运行：

```bash
# DPA：多任务模型要指定 head
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --output runs/dpa-sp

# GRACE：模型路径是一个 SavedModel 目录
.venv-grace/bin/mliport sp structure.cif --model saved_model \
  --model-type grace --output runs/grace-sp

# UMA：要指定 task
.venv/bin/mliport sp structure.cif --model uma.pt \
  --model-type uma --task omat --output runs/uma-sp
```

mliport 没有默认后端。不写 `--model-type`（也没有配置模型别名）会直接报错，这样每个数都
能说清是哪个模型算的。

## 用 LGPS 晶胞完整跑一遍

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda --output runs/lgps-mace-sp
```

用 MACE-OMAT-0 模型，应当得到约 −216 eV 的能量，力都是有限值，化学式为 `Ge2Li20P4S24`
（50 个原子）。如果用了 GPU，`mliport_results.json` 里会写 `actual_device_type: cuda`。
这个晶胞取自 2×2×2 的 LGPS 超胞，来源和哈希值见 `examples/structures/README.md`。

## 下一步

- 选模型：[模型](models.md)。
- 控制运行方式和严格模式：[配置](configuration.md)。
- MD、NEB 和批量计算：[工作流](workflows/index.md)。
- 分析轨迹：[轨迹分析](analysis/index.md)。
- 这些功能的测试情况：[验证](validation.md)。
