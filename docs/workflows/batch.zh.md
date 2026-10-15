# 批量

`mliport batch` 对一个目录里的每个结构运行同一种计算（默认是单点），使用同一个模型、
同一个后端和同一套配置。

```bash
.venv-mace/bin/mliport batch structures/ \
  --model mace-omat-0-medium.model --model-type mace \
  --calc-type sp --pattern "*.cif" \
  --output batch_results
```

模型只加载一次，计算器对每个结构复用，这正是批量的意义：启动开销只付一次。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--calc-type` | 对每个结构做什么：`sp`（默认）或 `opt` |
| `--pattern` | 文件通配符；不写时 mliport 会找 `*.cif`、`*.xyz`、`*.vasp` 和 `POSCAR*` |
| `--output` | 批量输出目录（默认 `batch_results`） |
| `--model`、`--model-type`、`--task`、`--head`、`--device` | 模型与后端的选择 |

## 会得到什么

`--output` 下每个结构一个运行目录，里面是常规的单点或弛豫文件，另外还有整个批次的进
度和失败报告。每个结构都保留自己的模型身份和生效配置。

## 出错时

失败的结构会被报告为失败，绝不会变成一条能量为零的记录。输入目录缺失或为空，直接失败。

## 一点提醒

批量模式是为了吞吐量，不保证一致性。请确认每个结构都符合后端的任务类型（周期性还
是分子），并且不要把需要不同 head 或 task 的结构放进同一批；分开运行，并把这一点写
清楚。

```bash
.venv-dpa/bin/mliport batch structures/ \
  --model DPA-3.1-3M.pt --model-type dpa --head Omat24 \
  --calc-type sp --output runs/dpa-batch
```
