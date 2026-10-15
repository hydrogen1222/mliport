# 队列

有了队列，你可以把几个计算一股脑交给 mliport，然后去忙别的。本地调度器会替你运行
它们，给每个作业单独的 GPU 租约（避免两个作业挤在同一张卡上），跟踪状态，并处理取
消。

```bash
.venv-mace/bin/mliport queue submit tasks.json
.venv-mace/bin/mliport queue start
.venv-mace/bin/mliport jobs
.venv-mace/bin/mliport queue stop
```

`queue start` 会启动后台调度器并立即返回。用 `mliport jobs`（或 `queue status`）轮询，
直到作业变成 `done` 或 `failed`，再停止调度器。TUI 的 Run 界面与命令行通过同一个辅助
函数提交作业，所以两边提交的作业行为一致。

## 任务文件

任务文件是一个 JSON 对象，里面有一个 `tasks` 列表。`calc_type` 取 `sp`、`opt`、`md`
或 `neb` 之一，路径必须存在：

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
      "options": {"ensemble": "NVT", "temperature": 800, "steps": 50000}
    }
  ]
}
```

运行选项放进 `options`。可选的 `python` 会让这一个任务在另一个后端的环境里运行，这样
同一个队列里可以混用不同引擎。模型路径和后端必须始终明确给出。

## 控制调度器

| 命令 | 选项 | 含义 |
|---|---|---|
| `queue submit` | 任务文件 | 把文件里的每个任务入队 |
| `queue start` | `--max-concurrent N` | 同时最多运行几个作业（默认 1，适用于单 GPU） |
| `queue start` | `--max-jobs-per-gpu N` | 同一块物理 GPU 上最多可共享几个作业（默认 1，即独占租约）；只有小体系、显存放得下时才调大 |
| `queue start` | `--poll SECONDS` | 调度器轮询间隔（默认 5） |
| `queue start` | `--foreground` | 在前台运行调度器（Ctrl-C 停止） |
| `queue status` | | 待处理、运行中和已结束的作业及其设备租约 |
| `queue stop` | | 停止后台调度器 |
| `queue pause` / `queue resume` | 可选的作业 id | 暂停或恢复某个待处理作业，或全部待处理的派发 |

`kill` 和 `clean` 针对单个作业记录操作。

## 东西放在哪

作业注册表位于 `MLIPORT_JOBS_DIR` 下（默认是用户级目录）。每个作业还会有自己常规的运
行目录，里面有 `run.log` 和 `mliport_results.json`，物理 GPU 租约则按作业记录。

## 出问题时

拿不到设备的作业会一直保持待处理，不会挤上超出 `--max-jobs-per-gpu` 的 GPU。崩溃或
异常会把作业标记为失败并写明原因，同时释放租约。调度器重启后会回收失效的占用。队列
会把 `CUDA_VISIBLE_DEVICES` 换成租到的那块物理 GPU，所以不要在作业里覆盖它。

队列顺序不会改变科学结果，但同一个队列里的作业共享一套配置面。做输运或 NEB 系列计算
时，请把模型身份、时间步长和保存间隔写进任务描述里，让得到的记录保持可比。
