# 队列

队列可以一次提交多个计算，交给本地调度器依次运行。调度器会给每个作业分配单独的 GPU（避免两个作业
挤在同一张卡上），记录作业状态，也支持取消。

```bash
.venv-mace/bin/mliport queue submit tasks.json
.venv-mace/bin/mliport queue start
.venv-mace/bin/mliport jobs
.venv-mace/bin/mliport queue stop
```

`queue start` 在后台启动调度器后立即返回。用 `mliport jobs`（或 `queue status`）查看状态，等作业变成
`done` 或 `failed` 后再停止调度器。TUI 的 Run 页面和命令行用同一套代码提交作业，行为一致。

## 任务文件

任务文件是一个 JSON 对象，包含一个 `tasks` 列表。`calc_type` 可以是 `sp`、`opt`、`md` 或 `neb`，文件
路径必须存在：

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

计算参数写在 `options` 里。`python` 字段可以让某个任务用另一个后端的环境运行，这样同一个队列里可以
混用不同后端。模型路径和后端每个任务都必须写明。

## 控制调度器

| 命令 | 选项 | 含义 |
|---|---|---|
| `queue submit` | 任务文件 | 把文件里的所有任务加入队列 |
| `queue start` | `--max-concurrent N` | 同时最多运行几个作业（默认 1，适用于单 GPU） |
| `queue start` | `--max-jobs-per-gpu N` | 一块 GPU 上最多同时跑几个作业（默认 1，即独占）；只有小体系、显存够用时才调大 |
| `queue start` | `--poll SECONDS` | 调度器轮询间隔（默认 5） |
| `queue start` | `--foreground` | 在前台运行调度器（Ctrl-C 停止） |
| `queue status` | | 等待中、运行中和已结束的作业，以及各自分配的 GPU |
| `queue stop` | | 停止后台调度器 |
| `queue pause` / `queue resume` | 可选的作业 id | 暂停或恢复某个等待中的作业；不写 id 则作用于全部 |

`kill` 和 `clean` 用于单个作业。

## 文件位置

作业记录保存在 `MLIPORT_JOBS_DIR` 下（默认在用户目录里）。每个作业有自己的运行目录，里面有 `run.log`
和 `mliport_results.json`；每个作业用的是哪块 GPU 也有记录。

## 出问题时

没有空闲 GPU 时，作业会一直等待，不会超出 `--max-jobs-per-gpu` 的限制。作业崩溃或出错时会标记为失败
并写明原因，同时释放 GPU。调度器重启后会清理失效的占用。队列会把 `CUDA_VISIBLE_DEVICES` 设为分配到
的那块 GPU，不要在作业里修改它。

作业的执行顺序不影响计算结果。做一组输运或 NEB 计算时，请在任务里写明模型、步长和保存间隔，保证
各次结果可以比较。
