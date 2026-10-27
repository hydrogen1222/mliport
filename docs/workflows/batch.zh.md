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
| `--pattern` | 文件名通配符；不写时查找 `*.cif`、`*.xyz`、`*.vasp` 和 `POSCAR*` |
| `--output` | 输出目录（默认 `batch_results`） |
| `--model`、`--model-type`、`--task`、`--head`、`--device` | 模型和后端 |

## 输出

`--output` 下每个结构一个子目录，里面是普通的单点或结构优化输出，另外还有整批的进度和失败汇总。
每个结构都单独记录模型信息和实际使用的参数。

## 出错时

某个结构算失败了，会标记为失败，不会记成能量为零。输入目录不存在或为空时直接报错。

## 注意

批量模式对所有结构用同一套设置。请确认每个结构都适合这个 task（周期性体系还是分子）；需要不同
head 或 task 的结构要分开跑，并在记录里写清楚。

```bash
.venv-dpa/bin/mliport batch structures/ \
  --model DPA-3.1-3M.pt --model-type dpa --head Omat24 \
  --calc-type sp --output runs/dpa-batch
```
