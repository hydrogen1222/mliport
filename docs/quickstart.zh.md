# 快速上手

大约十分钟，你会装好一个后端，跑一次单点计算和一次简短的结构弛豫，并且知道
结果该去哪里看。这里用 MACE 做例子，因为它的模型文件开放而且不大；其他后端
放在最后。你不需要四个都装。

## 1. 安装一个后端

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace --device auto
```

安装脚本会为这个后端单独建一个 Python 环境（`.venv-mace`），把 mliport 装进
去，在 `bin/` 里写一个启动器，最后跑一遍 `mliport doctor`。机器没有 NVIDIA
GPU 就用 `--device cpu`。离线机器、一次装多个后端、手动安装这些情况见
[安装](installation.md)。

## 2. 确认能用

```bash
.venv-mace/bin/mliport doctor --engine mace --device auto
```

环境真的跑不了这个后端时，`doctor` 会以非零状态退出。如果有检查项失败，先把
它解决；常见原因列在[故障排查](troubleshooting.md)里。

## 3. 准备模型和结构

你需要一个周期性结构（CIF、POSCAR 或 XYZ）和一个 MACE 模型文件。mliport 不会
下载也不会自带权重：请从上游发布页拿 `mace-omat-0-medium.model`（链接见
[模型](models.md)），记下保存的位置。

手边没有结构？仓库里带了一个 50 原子的 Li₁₀GeP₂S₁₂（LGPS）晶胞，路径是
`examples/structures/li10gep2s12_primitive.vasp`。它是真实的周期性晶胞，小到
CPU 就能跑。

## 4. 跑一次单点

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --output runs/mace-sp
```

运行时会打印能量和力的摘要，并在 `runs/mace-sp/` 里生成 `OUTCAR`、`OSZICAR`、
`CONTCAR`、`mliport_results.json`、`resolved_config.json` 和 `run.log`。每个文
件的含义见[输出文件](outputs.md)。

如果 `runs/mace-sp` 里已经有结果，mliport 会拒绝往里写；确实要覆盖时加
`--overwrite`。

## 5. 弛豫结构

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model \
  --model-type mace \
  --fmax 0.05 --max-steps 50 \
  --output runs/mace-opt
```

五十步可能不够，这没关系，结果里会如实写明。弛豫是否收敛记录在
`mliport_results.json` 里，没收敛的运行就是没收敛，不会被包装成成功。弛豫后的
结构在 `CONTCAR` 里。

## 6. 读取结果

```python
import json

with open("runs/mace-opt/mliport_results.json") as handle:
    record = json.load(handle)

calculation = record["calculation"]
print(calculation["optimization"]["converged"], calculation["results"]["energy"])
```

复现这次运行所需的信息都在旁边：模型身份、每个配置值的来源、请求的设备和实际
使用的设备，以及 mliport 版本。

## 其他后端

用 `--engines dpa`、`grace` 或 `uma` 安装想要的后端，然后使用它自己的环境：

```bash
# DPA —— 多任务模型需要指明你要用的分支
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --output runs/dpa-sp

# GRACE —— 模型是一个 SavedModel 目录
.venv-grace/bin/mliport sp structure.cif --model saved_model \
  --model-type grace --output runs/grace-sp

# UMA —— 要说明用哪个任务族
.venv/bin/mliport sp structure.cif --model uma.pt \
  --model-type uma --task omat --output runs/uma-sp
```

没有默认后端。省略 `--model-type`（又没有模型别名提供它）会直接报错，这样你
永远清楚一个数字是哪个模型算出来的。

## 在 LGPS 晶胞上完整跑一遍

```bash
.venv-mace/bin/mliport sp examples/structures/li10gep2s12_primitive.vasp \
  --model /path/to/mace-omat-0-medium.model \
  --model-type mace --task bulk --device cuda --output runs/lgps-mace-sp
```

用 MACE-OMAT-0 模型，你应当看到约 −216 eV 的有限能量、有限的力、化学式
`Ge2Li20P4S24`（50 个原子），以及——如果请求了 GPU——`mliport_results.json` 里的
`actual_device_type: cuda`。这个晶胞取自 2×2×2 的 LGPS 超胞，来源和哈希记录在
`examples/structures/README.md`。

## 接下来

- 认真选模型：[模型](models.md)。
- 控制运行与严格模式：[配置](configuration.md)。
- MD、NEB 和批量计算：[工作流](workflows/index.md)。
- 分析轨迹：[轨迹分析](analysis/index.md)。
- 看这些说法有多少测试支撑：[验证](validation.md)。
