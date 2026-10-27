# 从 mlipx 迁移

项目已经从 **mlipx** 改名为 **mliport**。计算代码一脉相承，变化的是名字、安装方式，以及几处以前会
自动替你选择的默认行为。

## 名称变化

| 以前 | 现在 |
|---|---|
| 包名与导入 `mlipx` | `mliport` |
| 命令 `mlipx ...` | `mliport ...` |
| `pyproject.toml` 里的 `mlipx` 入口 | `mliport` |
| 证据 schema 前缀 `mlipx.beta-validation-result/*` | `mliport.beta-validation-result/*` |

旧的验证记录都保留着。验证程序仍能读取 `mlipx.*` 格式的旧记录，并在元数据里标为
`legacy_namespace_records`，早期结果仍然可以查证；新记录统一用 `mliport.*` 格式。

## 安装

以前在一个环境里装好命令行，再由它去调用各后端的环境。现在 mliport 要装在**每个后端自己的环境里**，
使用时调用对应环境里的命令：`.venv-mace/bin/mliport`、`.venv-dpa/bin/mliport`、
`.venv-grace/bin/mliport`，UMA 是 `.venv/bin/mliport`；也可以用安装时生成的
`./bin/mliport-<engine>`。见[安装](installation.md)。

## 配置

1. **必须指定后端。** 以前默认用 UMA，现在没有默认值了。每次运行都要加 `--model-type`（或在
   `[model:...]` 别名里写 `engine = ...`），不写会报错。
2. **配置检查变严格了。** 拼错的参数名和属于别的后端的参数，以前只是警告，现在会报错。想恢复旧
   行为，加 `--lenient-config`，或在 `settings.ini` 里写 `[general] strict_config = false`。
3. **模型别名必须写 `engine`。** 以前默认是 UMA 的别名写法不再支持。
4. **task 按后端区分。** UMA 用自己的 task（`omat`、`omol` 等）；MACE、DPA 和 GRACE 用 `bulk` 和
   `molecule` 表示周期性体系和分子。
5. **环境变量改名。** 用 `MLIPORT_SETTINGS` 和 `MLIPORT_JOBS_DIR` 代替原来以 `mlipx` 开头的变量。
   `mliport config paths` 会列出设置文件的查找位置。

## Python API

```python
# 与后端无关的入口
from mliport import CalculatorFactory
```

UMA 的类现在是 `mliport.calculators.uma.UMACalculator`，和 MACE、DPA、GRACE 的类放在一起。旧的
`mliport.calculator` 兼容模块已在 2.0.0b4 删除，请从 `mliport.calculators` 导入，或者用
`CalculatorFactory` 并指定后端。

## 验证记录

旧记录仍然可以读取。新的验证使用 `mliport.beta-validation-result/2`、`mliport.beta-campaign/1` 和
`mliport.beta-archive-manifest/1` 格式，报告在 `validation/science/reports/` 下；见[验证](validation.md)。

## 检查清单

- [ ] 把脚本和 notebook 里的 `mlipx` 换成 `mliport`。
- [ ] 按后端重新安装（推荐用安装脚本）。
- [ ] 每次运行都指定后端或模型别名。
- [ ] 修改通不过严格检查的配置参数。
- [ ] 在每个后端环境里重新运行 `mliport doctor`。
