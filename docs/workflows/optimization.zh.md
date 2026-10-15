# 弛豫

`mliport opt` 用 ASE 优化器在你选的后端上最小化原子受力，也可以同时最小化晶胞的
应力。

```bash
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/opt
```

## 常用选项

| 选项 | 含义 |
|---|---|
| `--fmax` | 力的收敛阈值，单位 eV/Å（默认 0.05） |
| `--max-steps` | 硬性步数上限（默认 500） |
| `--optimizer` | `FIRE`（默认）、`BFGS` 或 `LBFGS` |
| `--cell-opt` | 同时弛豫晶胞；需要模型提供应力 |
| `--fix-symmetry` | 弛豫过程中保持晶体对称性 |
| `--model`、`--model-type`、`--task`、`--head`、`--device` | 模型与后端的选择，和[单点](single-point.md)一样 |

`--cell-opt` 的要求很严格：模型要支持应力，结构要三维全周期，任务要是块体任务。对
于孤立分子、部分周期性体系或不带应力的模型，它会停下并说明原因，而不是悄悄只弛豫
原子位置。

## 会得到什么

`CONTCAR` 是最终结构。`OUTCAR` 和 `OSZICAR` 记录能量与力的变化过程，
`mliport_results.json` 记录弛豫是否收敛、用了多少步、最终的能量和力，旁边还有
`resolved_config.json` 和 `run.log`。

## 收敛了没有

如果在达到 `--fmax` 之前用完了步数，运行会以状态 `2` 退出，保留部分轨迹并记录原
因。这是一次没有收敛的弛豫，而不是成功的弛豫，它留下的结构在使用前应当检查。后端
或运行时错误以 `1` 退出，已有的部分结果会保留。

## 弛豫出来的结构意味着什么

结果是*这个模型*势能面上的极小点，不是 DFT 的极小点，并且只在同一模型身份、同一参
考下才有意义。晶胞优化会改变参考体积，所以比较能量之前要先看最终的晶胞。

```bash
.venv-grace/bin/mliport opt structure.cif --model saved_model \
  --model-type grace --fmax 0.03 --max-steps 200 --output runs/grace-opt
```
