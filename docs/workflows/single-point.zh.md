# 单点

单点计算会给出一个结构的能量、力，以及（模型提供时）应力张量。原子不动，晶胞也
不变。在投入更长的计算之前，用它检查模型、结构和设备能不能一起工作，是最便宜的
办法。

```bash
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace --output runs/sp
```

结构可以是 ASE 能读的任何格式（CIF、XYZ、POSCAR 等）。它必须符合模型的预期：周期
性晶体要用块体任务（UMA 用 `omat`，其他后端用 `bulk`），孤立分子用分子任务
（`omol` 或 `molecule`）。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--model`、`--model-type` | 模型文件与后端；除非别名或 profile 已提供，否则两者都必须给出 |
| `--task` | UMA 的任务族，或其他后端明确的 `bulk` / `molecule` 语义 |
| `--head` | 模型有多个分支时，指定 DPA 分支或 MACE head |
| `--device` | `cpu`（默认）、`cuda`、`gpu` 或 `cuda:N`；请求的设备与实际设备都会记录 |
| `--charge`、`--spin` | 分子的电子态元数据（UMA `omol`） |
| `--write-outcar`、`--write-stress` | 写出哪些输出 |
| `--inference-mode`、`--dtype` | 后端专属的精度与推理选项 |
| `--overwrite` | 允许替换输出目录中已有的结果 |

## 会得到什么

输出目录里有：`OUTCAR`、`CONTCAR`（输入结构的副本）、含能量/力/应力的
`mliport_results.json`、`resolved_config.json` 和 `run.log`。每个文件的含义见
[输出文件](../outputs.md)。只有加载的模型能算应力时，结果里才会有应力。

## 失败时

模型加载不了、多任务 DPA 模型没给 `--head`、任务与后端不匹配、运行时出错，都会让
运行以非零退出码结束并给出原因。mliport 不会为了绕过问题而换成别的模型或默认后
端。能量和力必须是有限值，否则运行会中止，而不是把 NaN 写进输出。

## 怎么读这个数

单点结果只能和模型身份（模型、head、dtype、task）相同、结构和电子态也相同的另一个
单点结果比较。绝对能量跨模型不是物理可观测量；见[模型](../models.md)。

```bash
.venv-dpa/bin/mliport sp structure.cif --model DPA-3.1-3M.pt \
  --model-type dpa --head Omat24 --device cuda:0 \
  --output runs/dpa-sp
```
