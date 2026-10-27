# 验证

这一页说明 mliport 的各种验证结论分别代表什么、证据放在哪里。具体数字不在这里重复，它们在
各份报告里，每轮验证都会更新。

报告由每轮验证的记录自动生成：

- [BETA_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION.md)：完整验证报告，英文
- [BETA_VALIDATION_CN.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/BETA_VALIDATION_CN.md)：同一份报告的中文版
- [README_VALIDATION.md](https://github.com/hydrogen1222/mliport/blob/main/validation/science/README_VALIDATION.md)：一页的摘要

报告依据的原始记录：

```text
validation/science/campaigns/<campaign>.json     每轮验证的清单：对应的 commit 和记录
validation/science/archive_manifest.json         证据归档与校验和
validation/science/model_manifest.json           固定的模型 profile
validation/science/capability_matrix.json        哪个接口能表达哪个选项
```

## 状态标记的含义

报告表格里的每一格都标有下面其中一种状态，含义各不相同。

| 状态 | 含义 |
|---|---|
| `software_validated` | 在列出的软件环境和模型上，mliport 这条工作流运行正确 |
| `model_characterized` | 在某个基准上测过*模型*的表现，反映的是模型本身的性质 |
| `historical_evidence` | 早期验证留下的记录，仅供参考，不能说明当前版本 |
| `not_run` | 没有测试这个组合，不作任何结论 |
| `unsupported` | 后端或接口不支持该功能，文档里有说明 |
| `experimental` | 能运行，但不在支持范围内，以后可能变化 |
| `pending_revalidation` | 以前测过，当前版本还没有重新测 |

`software_validated` 只说明程序能正常运行，和模型精度无关；`model_characterized` 的基准结果
也只代表基准体系，换成你的体系需要另外检验。

## 一条记录包含的信息

每条记录包括：软件 commit、工作流和测试级别、后端和框架版本、模型信息（路径、task、head、
精度、SHA-256）、设备（请求的和实际的）、状态，以及输出文件和它们的哈希。一条记录只对这个
commit 和这个模型有效。

验证分为 T1 到 T8 几个级别，依次是后端和工作流的冒烟测试、一致性检查、输出数值检查、基准测试
等数值和物理合理性检查，以及性能和输运分析。引用某一级的数字之前，先看报告里写的适用范围。

## 安装冒烟证据

这张表比上面的工作流验证更早。保留它是因为每个"通过"都有对应的运行记录可以查看。

| 后端 / 模型 | SP | 短 MD | 能量–力梯度 | NEB 冒烟 | GRACE 缓存 | 记录 |
|---|---|---|---|---|---|---|
| UMA | 未运行 | 未运行 | 未运行 | 未运行 | n/a | [uma.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/uma.json)（这台机器没有离线 wheelhouse） |
| MACE float64 | 通过 | 通过 | 通过 | 通过 | n/a | [mace.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/mace.json) |
| MACE float32 | 通过 | 通过 | 通过 | 通过 | n/a | [mace-float32.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/mace-float32.json) |
| DPA `Domains_Alloy` | 通过 | 通过 | 通过 | 通过 | n/a | [dpa.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/dpa.json) |
| GRACE float32 | 通过 | 通过 | 通过 | 通过 | 通过（开与关） | [grace.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/grace.json)、[grace-nocache.json](https://github.com/hydrogen1222/mliport/blob/main/validation/runtime/v100/grace-nocache.json) |

NEB 冒烟测试是在 4 原子 Cu 晶胞上用 5 个 image、固定晶胞跑的短计算，`saddle_validation:
not_performed`。能带收敛了，也不代表过渡态已经确认。

## LGPS 全新安装验收

2.0.0b3 版本增加了一轮按文档操作的完整验收：安装命令、四个 GPU 后端、LGPS 上的各种工作流、
分析模块、一次 CPU 安装，最后在全新环境里再复核一遍。报告和记录在这里：

- [LGPS_ACCEPTANCE_REPORT.md](https://github.com/hydrogen1222/mliport/blob/main/validation/acceptance/LGPS_ACCEPTANCE_REPORT.md)
- [beta-summary.json](https://github.com/hydrogen1222/mliport/blob/main/validation/science/reports/beta-summary.json)

验收机器上的安装情况、具体命令和尚未解决的限制都写在报告里，都是实测的，不是从 CI 推断的。

## 验证的范围

- 只覆盖列出的软件环境、GPU 和模型 profile，换了模型、机器或体系，需要你自己检验。
- 性能数字只对测量用的硬件和版本有效，报告里附有对应的 commit。
- 所有从 MD 得到的量都依赖采样，分析结果会记录所用的时间窗口和不确定度。
- 没有测过的组合标为 `not_run`，不能当作"能用"。
