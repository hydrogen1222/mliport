# 从 mlipx 迁移

项目已经从 **mlipx** 更名为 **mliport**。科学代码是同一脉络；变的是名字、安装方式，以及几处以前会
替你猜测的默认行为。

## 改了什么名字

| 以前 | 现在 |
|---|---|
| 包名与导入 `mlipx` | `mliport` |
| 命令 `mlipx ...` | `mliport ...` |
| `pyproject.toml` 里的 `mlipx` 入口 | `mliport` |
| 证据 schema 前缀 `mlipx.beta-validation-result/*` | `mliport.beta-validation-result/*` |

旧证据没有被丢弃。验证加载器仍然会读取遗留的 `mlipx.*` 记录，并在验证活动的元数据里把它们列为
`legacy_namespace_records`，所以早期结果依然可审计，而新记录使用 `mliport.*` 的 schema。

## 安装

以前，一个工作环境就能装好 CLI，再去调用各后端环境。现在 mliport 要装在**每个后端自己的环境里**，
然后调用该环境里的命令：`.venv-mace/bin/mliport`、`.venv-dpa/bin/mliport`、
`.venv-grace/bin/mliport`，UMA 则是 `.venv/bin/mliport`，或者用生成的 `./bin/mliport-<engine>` 启动
器。见[安装](installation.md)。

## 配置

1. **后端必须显式。** 不再有隐含的 UMA 默认值。每次运行都要加 `--model-type`（或在
   `[model:...]` 别名里写 `engine = ...`）；省略它是致命的配置错误。
2. **配置是严格的。** 拼错的键和属于另一个后端的选项现在会失败，而不是警告。如果你需要旧行为，用
   `--lenient-config` 或在 `settings.ini` 里写 `[general] strict_config = false` 明确要求。
3. **模型别名必须声明 `engine`。** 旧的隐含 UMA 别名没有了。
4. **task 默认值跟随后端。** UMA 使用它自己的 task 族（`omat`、`omol` 等）；MACE、DPA 和 GRACE
   使用显式的 `bulk` 和 `molecule` 含义。
5. **环境变量。** `MLIPORT_SETTINGS` 和 `MLIPORT_JOBS_DIR` 取代了以 `mlipx` 命名的那些。
   `mliport config paths` 会打印设置文件的搜索位置。

## Python API

```python
# 与后端无关的入口
from mliport import CalculatorFactory
```

UMA 类现在位于 `mliport.calculators.uma.UMACalculator`，与 MACE、DPA、GRACE 的封装并列。旧的
`mliport.calculator` 兼容层已在 2.0.0b4 移除：请从 `mliport.calculators` 导入，或者使用
`CalculatorFactory` 并显式选择后端。`mliport.__all__` 从来没有特别抬高 UMA。

## 验证记录

旧记录依然可读。新的验证活动使用 `mliport.beta-validation-result/2`、`mliport.beta-campaign/1` 和
`mliport.beta-archive-manifest/1`，报告放在 `validation/science/reports/` 下；见[验证](validation.md)。

## 检查清单

- [ ] 把脚本和 notebook 里的 `mlipx` 换成 `mliport`。
- [ ] 按后端重新安装（安装器是受支持的方式）。
- [ ] 给每次运行一个显式的后端或别名。
- [ ] 修正现在通不过严格校验的配置键。
- [ ] 在每个后端环境里重新运行 `mliport doctor`。
