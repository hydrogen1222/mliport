# 弛豫

`mliport opt` 用 ASE 的优化器做结构优化，把原子受力降到阈值以下，也可以同时优化晶胞。

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/opt
```

## 常用选项

| 选项 | 含义 |
|---|---|
| `--fmax` | 力的收敛阈值，单位 eV/Å（默认 0.05） |
| `--max-steps` | 最大步数（默认 500） |
| `--optimizer` | `FIRE`（默认）、`BFGS` 或 `LBFGS` |
| `--cell-opt` | 同时优化晶胞；需要模型支持应力 |
| `--fix-symmetry` | 优化过程中保持晶体对称性 |
| `--model`、`--model-type`、`--task`、`--head`、`--device` | 模型和后端，用法同[单点](single-point.md) |

`--cell-opt` 要求模型支持应力、结构三维周期、task 是体相。孤立分子、部分周期性体系或不支持应力
的模型会直接报错并说明原因，程序不会自动改成只优化原子位置。

## 输出

`CONTCAR` 是优化后的结构。`OUTCAR` 和 `OSZICAR` 记录每一步的能量和力，`mliport_results.json`
记录是否收敛、用了多少步、最终的能量和力。另外还有 `resolved_config.json` 和 `run.log`。

## 是否收敛

如果步数用完还没达到 `--fmax`，程序以退出码 `2` 结束，保留已有的轨迹并记录原因。这样的结构没有
收敛，使用前要检查。后端或运行时出错以退出码 `1` 结束，保留已有的部分结果，并把错误写入
`mliport_results.json`。

## 优化结果的含义

得到的是*这个模型*势能面上的极小点，和 DFT 的极小点不一定相同，只能和同一个模型的结果比较。
晶胞优化会改变体积，比较能量前先看一下最终的晶胞。

```bash
.venv-grace/bin/mliport opt structure.cif --model saved_model \
  --model-type grace --fmax 0.03 --max-steps 200 --output runs/grace-opt
```
