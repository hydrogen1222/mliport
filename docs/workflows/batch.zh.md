# 批量

`mliport batch` 对一个目录里的所有结构做同一种计算（默认是单点），用同一个模型和同一套参数。

```bash
.venv-mace/bin/mliport batch structures/ \
  --model mace-omat-0-medium.model --model-type mace \
  --calc-type sp --pattern "*.cif" \
  --output batch_results
```

模型只加载一次，所有结构共用，省去了每次加载模型的时间。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--calc-type` | 计算类型：`sp`（默认）或 `opt` |
| `--pattern` | 相对输入目录的通配符，可以跨子目录（`'*/rank_*/1.cif'`、`'**/*.cif'`）；不写时只在输入目录本层查找 `*.cif`、`*.xyz`、`*.vasp` 和 `POSCAR*` |
| `--output` | 输出目录（默认 `batch_results`） |
| `--fmax`、`--max-steps`、`--optimizer`、`--cell-opt`、`--fix-symmetry` | `--calc-type opt` 的优化参数，含义和默认值与 `mliport opt` 相同；`--calc-type sp` 时给出会直接报错 |
| `--resume` | 在同一个 `--output` 目录里接着跑中断的批量任务 |
| `--model`、`--model-type`、`--task`、`--head`、`--device` | 模型和后端 |

## 优化分散在多个目录里的结构

生成的结构经常文件名相同、只是目录不同，例如 `ML-OPT/<组成>/rank_0001/1.cif`。
把顶层目录交给 batch，再用通配符选文件：

```bash
mliport batch ML-OPT/ \
  --model my-grace-model --model-type grace --device cuda \
  --calc-type opt --pattern '*/rank_*/1.cif' \
  --fmax 0.05 --max-steps 1000 --optimizer FIRE --no-cell-opt \
  --output runs/ml-opt
```

通配符要加引号，避免被 shell 展开。每个结构的输出目录对应它在输入目录下的路径
（`runs/ml-opt/<组成>/rank_0001/1/`），同名文件不会互相覆盖。如果仍有两个输入会落到同一个
输出目录，批量任务在开始计算前就会报错。

## 输出

`--output` 下每个结构一个子目录，里面是普通的单点或结构优化输出，另外还有整批的进度和失败汇总。
每个结构都单独记录模型信息和实际使用的参数。`batch_summary.json` 会分别标明
`completed`、`not_converged` 或 `failed`；只有真正完成的结果才标记 `success: true`。
未收敛的优化计入 `failed` 总数，并单独计入 `not_converged`，不会被误报为成功。

## 出错时

某个结构算失败了，会标记为失败，不会记成能量为零，然后继续算下一个结构。输入目录不存在或为空时直接报错。

## 中断与 `--resume`

每个结构一算完就记录结果：它自己目录下的 `batch_status.json`，以及批量目录下
`batch_progress.jsonl` 追加的一行。`batch_manifest.json` 记录决定这批计算的设置：模型、task、
计算器选项和优化参数。`batch_summary.json` 在整批结束时写出。

任务中断后，用同样的命令加上 `--resume` 重跑：

- 记录为 `completed` 或 `not_converged` 的结构直接复用，不重算（同样设置下重跑确定性的优化，结果不会变）；
- 记录为 `failed` 的结构，以及中断时正在算的那个结构，从输入结构重新算；
- 只要 `batch_manifest.json` 里有任何设置不同，就拒绝续跑，保证不会把不同设置算出的结果混在一起。
  想让未收敛的结构多算几步，请换一个新的 `--output` 目录。

`--resume` 不能和 `--overwrite` 一起用。中断时正在算的结构从输入几何重新开始，优化器不会从中断的那一步接着算。

## 注意

批量模式对所有结构用同一套设置。请确认每个结构都适合这个 task（周期性体系还是分子）；需要不同
head 或 task 的结构要分开跑，并在记录里写清楚。

```bash
.venv-dpa/bin/mliport batch structures/ \
  --model DPA-3.1-3M.pt --model-type dpa --head Omat24 \
  --calc-type sp --output runs/dpa-batch
```
