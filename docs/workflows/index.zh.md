# 工作流

mliport 提供下面几种计算。它们选模型和设参数的方式都一样，每次运行都会在结果旁边写一份
`resolved_config.json`。

| 工作流 | 命令 | 用来做什么 |
|---|---|---|
| [单点](single-point.md) | `mliport sp` | 一个结构的能量、力和应力 |
| [结构优化](optimization.md) | `mliport opt` | 优化原子位置，可选同时优化晶胞 |
| [分子动力学](md.md) | `mliport md` | 生成 NVE 或 NVT 轨迹，用于后续分析 |
| [NEB / CI-NEB](neb.md) | `mliport neb` | 反应路径和势垒 |
| [构型搜索](confsearch.md) | `mliport confsearch` | slab 上的低能吸附构型 |
| [批量](batch.md) | `mliport batch` | 对许多结构做同一种计算 |
| [队列](queue.md) | `mliport queue` | 排队运行多个作业，按 GPU 分配 |

`mliport run` 读取 INCAR 风格的输入文件，可以启动上面任何一种计算；`mliport template` 生成一份
带注释的模板，改一改就能用。`mliport tui` 在终端界面里提供同样的功能。

## 退出码

命令行、TUI 和队列用同一套退出码，脚本可以据此区分"正常完成"和"跑完但没收敛"：

| 退出码 | 含义 |
|---|---|
| `0` | 正常完成 |
| `1` | 失败（出错、配置有误、运行时故障） |
| `2` | 跑完了但没有收敛；部分结果和原因会保留 |
| `130` | 被取消（Ctrl-C 或主动取消） |

结构优化或 NEB 的结果如果没有收敛标记，一律按"未收敛"处理。进程正常退出不代表计算成功。

## 跟踪运行进度

每次运行都会在输出目录里实时写 `run.log`，并打印查看它的命令（`tail -f`）。长时间的计算（包括
NEB）会持续往里写进度，终端没有输出不代表作业卡住了。
