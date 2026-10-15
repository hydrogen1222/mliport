# 验证

这一页讲清楚 mliport 的验证声明各是什么意思，让你分得清事实和愿望，也说明证据放在哪里。数字
本身不在这里重复：它们属于各份报告，并且每次验证活动都会变。

报告由验证活动的记录生成：

- [BETA_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md)：完整的验证活动，英文
- [BETA_VALIDATION_CN.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION_CN.md)：同一份报告的中文版
- [README_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/README_VALIDATION.md)：一页纸的摘要

报告背后是机器可读的证据：

```text
validation/science/campaigns/<campaign>.json     验证活动清单：目标提交、记录
validation/science/archive_manifest.json         证据归档与校验和
validation/science/model_manifest.json           固定的模型 profile
validation/science/capability_matrix.json        哪个接口能表达哪个选项
```

## 状态是什么意思

报告里的每个单元格都带着下面某一个词，它们之间不能互相替换。

| 状态 | 含义 |
|---|---|
| `software_validated` | mliport 在所列的运行时与模型组合上，对所列工作流的代码路径运行正确 |
| `model_characterized` | 在某个基准上测过*模型*；这是模型的性质，不是 mliport 的 |
| `historical_evidence` | 早期验证活动留下的记录，仅供参考；它对当前提交不说明任何事 |
| `not_run` | 这个组合没有被执行，因此不作任何声明 |
| `unsupported` | 后端或接口无法表达该功能，文档里也这样写明 |
| `experimental` | 能运行，但在受支持的范围之外，可能变化 |
| `pending_revalidation` | 以前覆盖过，当前版本的证据还没有重新生成 |

`software_validated` 的冒烟测试说明适配器能工作，它不是关于精度的声明；`model_characterized`
的基准也不是对你所研究化学体系的声明。

## 一条记录包含什么

一条记录把这些信息绑在一起：软件提交、工作流与层级、后端与框架版本、模型标识（路径、task、
head、dtype、SHA-256）、设备（请求的与实际的）、状态，以及带哈希的产物。它只对这个提交和
这个模型有效，对别的都无效。

验证活动按层级组织，报告里是 T1 到 T8：后端与工作流冒烟测试、一致性检查、产出值检查、基准
这类数值与物理合理性检查，以及性能与输运分析。引用某一层的数字之前，请先在报告里读它的适用
范围。

## 安装冒烟证据

这张表早于上面的工作流级验证活动。保留它，是因为每个 `passed` 单元格都有可以打开查看的运行时
记录支撑。

| 后端 / 模型 | SP | 短 MD | 能量–力梯度 | NEB 冒烟 | GRACE 缓存 | 记录 |
|---|---|---|---|---|---|---|
| UMA | 未运行 | 未运行 | 未运行 | 未运行 | n/a | [uma.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/uma.json)（这台机器没有离线 wheelhouse） |
| MACE float64 | 通过 | 通过 | 通过 | 通过 | n/a | [mace.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/mace.json) |
| MACE float32 | 通过 | 通过 | 通过 | 通过 | n/a | [mace-float32.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/mace-float32.json) |
| DPA `Domains_Alloy` | 通过 | 通过 | 通过 | 通过 | n/a | [dpa.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/dpa.json) |
| GRACE float32 | 通过 | 通过 | 通过 | 通过 | 通过（开与关） | [grace.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/grace.json)、[grace-nocache.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/grace-nocache.json) |

NEB 冒烟是在 4 原子 Cu 晶胞上做的 5 个镜像、固定晶胞的短运行，`saddle_validation:
not_performed`。收敛的能带并不等于经过验证的过渡态。

## LGPS 全新安装验收

2.0.0b3 周期加了一轮沿着文档所述用户路径走的验收：安装命令、四个 GPU 后端、LGPS 工作流矩阵、
分析模块、一次 CPU 安装，以及最后一遍干净环境的复核。报告和证据在这里：

- [LGPS_ACCEPTANCE_REPORT.md](https://github.com/hydrogen1222/mliport/blob/main/validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- [beta-summary.json](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/beta-summary.json)

验收主机的安装矩阵、确切命令和遗留的限制都记在那里，而不是从 CI 推断出来。

## 验证到哪里为止

- 它只覆盖列出的运行时、GPU 和模型 profile，不为任意检查点、机器或化学体系背书。
- 性能数字属于测量它们的硬件和版本，报告时会附上提交。
- 任何来自 MD 的量都取决于采样。分析会记录窗口和不确定度。
- 没有证据的地方，状态是 `not_run`，绝不会是"默认能用"。
