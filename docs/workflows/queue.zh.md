# 任务队列

队列允许你提交多个计算、启动一个本地调度器，然后无需值守。每个作业获得自己的 GPU 租约，两个作业绝不会争用同一块显卡。调度器跟踪状态并处理取消操作。

## 快速开始

```bash
# 从任务文件提交作业
mliport queue submit tasks.json

# 启动调度器（默认后台运行）
mliport queue start

# 查看进度
mliport jobs          # 或：mliport queue status

# 完成后停止调度器
mliport queue stop
```

`queue start` 启动一个后台调度器并立即返回。轮询 `mliport jobs` 直到你的作业变为 `done` 或 `failed`。TUI 的运行界面通过同一个辅助组件提交作业，因此从两个界面提交的作业行为完全一致。

## 任务文件

任务文件是一个包含 `tasks` 列表的 JSON 对象。每个任务指定工作流、结构、模型及其后端：

```json
{
  "max_concurrent": 1,
  "tasks": [
    {
      "name": "lgps-sp",
      "calc_type": "sp",
      "structure": "/data/LGPS.vasp",
      "model": "/data/mace-omat-0-medium.model",
      "model_type": "mace",
      "task": "bulk",
      "device": "cuda",
      "output_dir": "/data/runs/lgps-sp"
    },
    {
      "name": "uma-md-800K",
      "calc_type": "md",
      "python": "/data/mliport/.venv/bin/python",
      "structure": "/data/LGPS.vasp",
      "model": "/data/uma-s-1p2.pt",
      "model_type": "uma",
      "task": "omat",
      "device": "cuda",
      "options": {
        "ensemble": "NVT",
        "temperature": 800,
        "steps": 50000
      }
    }
  ]
}
```

### 关键字段

- `calc_type`：`sp`、`opt`、`md` 或 `neb` 之一
- `options`：传递给工作流的运行选项（与 CLI 标志相同）
- `python`：可选，指定另一个后端的 Python 解释器路径，使单个队列可以混用不同引擎（例如 MACE 在一个 venv 中、DPA 在另一个中）
- 模型路径和后端必须始终显式指定

## 调度器命令

| 命令 | 选项 | 含义 |
|---|---|---|
| `queue submit` | 任务文件 | 将文件中的所有任务排入队列 |
| `queue start` | `--max-concurrent N` | 同时运行的最大作业数（默认 1） |
| `queue start` | `--max-jobs-per-gpu N` | 共享一块物理 GPU 的作业数（默认 1 = 独占） |
| `queue start` | `--poll SECONDS` | 轮询间隔（默认 5 秒） |
| `queue start` | `--foreground` | 前台运行（Ctrl-C 停止） |
| `queue status` | — | 显示待处理、运行中和已完成的作业 |
| `queue stop` | — | 停止后台调度器 |
| `queue pause` / `resume` | 可选的作业 ID | 暂停/恢复单个作业或所有待分发的作业 |

`mliport kill` 和 `mliport clean` 用于操作单个作业记录。

## GPU 管理

每个作业获得一个物理 GPU 的租约。调度器将 `CUDA_VISIBLE_DEVICES` 设置为租借的 GPU——不要在作业内部覆盖它。

- 无法获得设备的作业保持 pending 状态；它不会挤进超出 `--max-jobs-per-gpu` 限制的 GPU。
- 崩溃或异常会将作业标记为 failed、记录原因并释放租约。
- 重启的调度器会恢复过时的声明。

仅当小体系可以在 GPU 显存中共存时，才提高 `--max-jobs-per-gpu`。

## 文件存储位置

作业注册表存储在 `MLIPORT_JOBS_DIR`（默认为用户级目录）下。每个作业也有自己标准的运行目录，包含 `run.log` 和 `mliport_results.json`。

## 错误处理

失败的作业被标记为 failed 并记录原因，租约随即释放。队列继续处理下一个作业。调度器本身具有韧性：杀掉并重启它会接续上次的进度。

## 大规模计算活动建议

- 队列顺序不影响科学结论，但同一队列中的作业共享一个配置空间。对于输运或 NEB 系列计算，在任务规格中保持模型身份、步长和保存间隔一致，以确保结果可比。
- 对于大规模活动，使用 `--max-concurrent` 匹配可用 GPU 数量，`--max-jobs-per-gpu 1`（默认值）避免显存争用。
- `mliport state` 命令显示所有队列中作业注册表的当前状态。
