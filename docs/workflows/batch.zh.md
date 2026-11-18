# 批量计算

`mliport batch` 对一个目录下的每个结构执行同一种计算——单点计算或结构优化。模型只加载一次并复用于所有结构，因此启动开销只付出一次。

## 何时使用批量模式

批量模式面向**吞吐量**：你有许多结构和一个模型，想要所有结构的结果。常见用例包括：

- 筛选结构预测产生的一组候选结构
- 对训练集或验证集计算单点能量
- 用另一个模型重新优化一组结构
- 对照数据集基准测试一个模型

## 命令行

```bash
# 对目录中所有结构做单点计算
mliport batch structures/ \
  --model mace-omat-0-medium.model --model-type mace \
  --calc-type sp --pattern "*.cif" \
  --output batch_results

# 针对嵌套目录的优化
mliport batch ML-OPT/ \
  --model my-grace-model --model-type grace --device cuda \
  --calc-type opt --pattern '*/rank_*/1.cif' \
  --fmax 0.05 --max-steps 1000 --optimizer FIRE --no-cell-opt \
  --output runs/ml-opt

通配符要加引号，避免被 shell 展开。每个结构的输出目录对应它在输入目录下的路径
（`runs/ml-opt/<组成>/rank_0001/1/`），同名文件不会互相覆盖。如果仍有两个输入会落到同一个
输出目录，批量任务在开始计算前就会报错。

## 输出

`--output` 下每个结构一个子目录，里面是普通的单点或结构优化输出，另外还有整批的进度和失败汇总。
每个结构都单独记录模型信息和实际使用的参数。`batch_summary.json` 会分别标明
`completed`、`not_converged` 或 `failed`；只有真正完成的结果才标记 `success: true`。
未收敛的优化计入 `failed` 总数，并单独计入 `not_converged`，不会被误报为成功。

如果完整汇总无法序列化，会改写一份降级文档：带 `"summary_status": "degraded"`、
`metadata.summary_write_error` 里的原始错误，以及每个结构的状态。若连降级文档也写不出来，
整批会报错并同时给出两个错误，而不是假装汇总已保存；无论哪种情况，各结构自己的
`batch_status.json` 都不受影响。

## 出错时

某个结构算失败了，会标记为失败，不会记成能量为零，然后继续算下一个结构。输入目录不存在或为空时直接报错。

后端返回不可用数值时同样如此：`energy` 或 `fmax` 是 NaN/无穷，或者根本不是数字，该结构记为 `failed`，被替换的字段列在 `non_finite_fields` / `unserialisable_fields`，原因写进 `error`；数值存为 `null`（或占位字符串）而绝不伪造为 `0`，其他结构和整批汇总仍会正常写出。

## 中断与 `--resume`

每个结构一算完就记录结果：它自己目录下的 `batch_status.json`，以及批量目录下
`batch_progress.jsonl` 追加的一行。`batch_manifest.json` 记录决定这批计算的设置：模型（路径；本地模型文件还记录权重的 sha256）、task、
计算器选项和优化参数。`batch_summary.json` 在整批结束时写出。

任务中断后，用同样的命令加上 `--resume` 重跑：

- 记录为 `completed` 或 `not_converged` 的结构直接复用，不重算（同样设置下重跑确定性的优化，结果不会变）。
  前提是输入文件内容（sha256）和当时一致，且这次计算写出的每个输出文件都还在、大小没变；否则重算；
- 记录为 `failed` 的结构，以及中断时正在算的那个结构，从输入结构重新算；
- 只要 `batch_manifest.json` 里有任何设置不同，就拒绝续跑，保证不会把不同设置算出的结果混在一起。
  想让未收敛的结构多算几步，请换一个新的 `--output` 目录。

`--resume` 不能和 `--overwrite` 一起用，续跑时 `run.log` 追加写入。中断时正在算的结构从输入几何重新开始，优化器不会从中断的那一步接着算。

## 注意

批量模式对所有结构用同一套设置。请确认每个结构都适合这个 task（周期性体系还是分子）；需要不同
head 或 task 的结构要分开跑，并在记录里写清楚。

```bash
.venv-dpa/bin/mliport batch structures/ \
  --model DPA-3.1-3M.pt --model-type dpa --head Omat24 \
  --calc-type sp --output runs/dpa-batch
```

`--pattern` 需要加引号以防止 shell 展开。每个结构的输出目录镜像其在输入目录下的相对路径，因此同名文件不会互相覆盖。

## 选项

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--calc-type` | `sp` | 对每个结构执行的计算类型：`sp` 或 `opt` |
| `--pattern` | `*.cif`, `*.xyz`, `*.vasp`, `POSCAR*` | 文件 glob 模式；可跨子目录（`'*/rank_*/1.cif'`、`'**/*.cif'`） |
| `--output` | `batch_results` | 批量输出目录 |
| `--resume` | — | 继续被中断的批量计算 |
| `--fmax` | 0.05 eV/Å | 优化力收敛阈值（仅 `--calc-type opt`） |
| `--max-steps` | 500 | 优化步数上限（仅 `--calc-type opt`） |
| `--optimizer` | `FIRE` | 优化器（仅 `--calc-type opt`） |
| `--cell-opt` | 关闭 | 晶胞优化（仅 `--calc-type opt`） |
| `--fix-symmetry` | 关闭 | 保持对称性（仅 `--calc-type opt`） |
| `--model`、`--model-type`、`--task`、`--head`、`--device` | — | 模型与后端选择 |

优化选项（`--fmax`、`--max-steps`、`--optimizer`、`--cell-opt`、`--fix-symmetry`）在 `--calc-type` 为 `sp` 时会被拒绝。

## 输出结构

`--output` 目录下每个结构对应一个子目录，各自包含标准的单点或优化输出文件。每个结构保留自己的模型身份和解析后的配置。在批量层面：

| 文件 | 内容 |
|---|---|
| `batch_summary.json` | 总览状态：已完成、失败和未收敛的计数 |
| `batch_progress.jsonl` | 每个结构完成后追加一行 |
| `batch_manifest.json` | 定义批量的设置（模型、选项、文件哈希） |

`batch_summary.json` 中每条记录的 `status` 为 `completed`、`not_converged` 或 `failed`。只有 completed 的运行才有 `success: true`；未收敛的优化同时计入 `failed` 和 `not_converged`。

## 错误处理

失败的结构会被报告为 failed——永远不会变成零能量条目。批量计算继续处理下一个结构。空的或不存在的输入目录会在任何计算开始之前就报错。两个输入结构若映射到同一个输出目录，会在批量开始前被检测到并拒绝。

## 续算

每个结构在完成时立即记录结果：自身目录中的 `batch_status.json` 和 `batch_progress.jsonl` 中追加的一行。中断后继续：

```bash
mliport batch structures/ --model ... --calc-type sp \
  --output batch_results --resume
```

续算规则：

- **已完成**和**未收敛**的结构会被复用（不重新计算），但前提是输入文件的 sha256 相同且所有输出文件完好。
- **失败**的结构以及中断时正在运行的结构会从输入几何结构重新运行。
- 如果 `batch_manifest.json` 中的任何设置不同，批量计算**拒绝续算**——不同设置的结果绝不混合。
- `--resume` 不能与 `--overwrite` 组合使用。

## 重要注意事项

批量模式是为吞吐量设计的，而非一致性保证。请确保每个结构与后端的 task（周期性 vs 分子）相匹配，不要在同一批量中混用需要不同 head 或 task 的结构——请分开运行。
