# 单点

单点计算给出一个结构的能量、力，以及应力张量（模型支持时）。原子和晶胞都不动。在跑长时间计算
之前，可以先用单点确认模型、结构和设备都没问题。

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace --output runs/sp
```

结构可以是 ASE 能读的任何格式（CIF、XYZ、POSCAR 等），并且要和 task 对应：周期性晶体用体相
task（UMA 用 `omat`，其他后端用 `bulk`），孤立分子用分子 task（`omol` 或 `molecule`）。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--model`、`--model-type` | 模型文件和后端；没有用别名或 profile 时两个都必须写 |
| `--task` | UMA 的 task，或其他后端的 `bulk` / `molecule` |
| `--head` | 模型有多个分支时，指定 DPA 分支或 MACE head |
| `--device` | `cpu`（默认）、`cuda`、`gpu` 或 `cuda:N`；请求的和实际使用的设备都会记录 |
| `--charge`、`--spin` | 分子的电荷和自旋（UMA `omol`） |
| `--write-outcar`、`--write-stress` | 是否写出这些输出 |
| `--inference-mode`、`--dtype` | 各后端的精度和推理设置 |
| `--overwrite` | 允许覆盖输出目录里已有的结果 |

## 输出

输出目录里有 `OUTCAR`、`CONTCAR`（输入结构的副本）、记录能量、力和应力的
`mliport_results.json`、`resolved_config.json` 和 `run.log`。各文件的含义见
[输出文件](../outputs.md)。模型不支持应力时，结果里没有应力。

## 出错时

模型加载失败、多任务 DPA 模型没写 `--head`、task 和后端不匹配、运行时出错，都会以非零退出码结束
并说明原因，程序不会自动换用别的模型或后端。能量或力出现 NaN 时计算会中止，不会写进输出。

## 结果怎么用

单点能量只能和同一个模型（模型文件、head、精度、task 都相同）、同样电子态下算出的结果比较。
不同模型的绝对能量没有可比性，见[模型](../models.md)。

```bash
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda:0 \
  --output runs/dpa-sp
```
