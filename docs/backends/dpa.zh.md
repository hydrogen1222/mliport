# DPA（DeepMD-kit）

DPA 模型通过 `deepmd-kit` 加载。其中很多是带有多个命名分支的多任务模型，mliport
不会猜你想用哪个分支：你要显式地把它作为 head 传进来，否则运行会直接失败（fail
closed）。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit==3.1.3` (torch 2.10.0) |

```bash
./scripts/install_mliport.sh --engines dpa --device auto
```
<!-- END GENERATED -->

模型加载器需要 `deepmd-kit[torch]` 的元数据，安装脚本会一并装上匹配的 `mpich`。

## 固定的 profile

<!-- BEGIN GENERATED: profiles -->
| Profile | 标识 | dtype | task | head | 模型 SHA-256 |
|---|---|---|---|---|---|
| `dpa_omat` | DPA-3.1-3M, branch Omat24 | upstream/model-defined | bulk | Omat24 | `86dd3a804d78...` |
<!-- END GENERATED -->

## 运行

```bash
.venv-dpa/bin/mliport sp structure.cif \
  --model /path/to/DPA-3.1-3M.pt --model-type dpa --head Omat24
```

想知道一个模型有哪些分支，先问 DeePMD-kit，再把想用的那个作为 `--head`（INCAR 文
件里是 `HEAD`）传入：

```bash
dp --pt show DPA-3.1-3M.pt model-branch
```

## 需要知道的事

- **多任务模型不写 head 是致命的配置错误。** 没有悄悄使用的默认分支。
- 模型元数据检查失败后不会缓存计算器。对同一个包装器重试时，必须重新加载并通过
  检查后才能使用模型。
- **精度**由模型决定。mliport 记录后端实际报告的 dtype，从不拿自己的想法去替代。
- **应力**只有在你加载的模型和分支提供它时才会报告。
- **DPA 需要按进程隔离 GPU。** `CUDA_VISIBLE_DEVICES` 未设置时，命令行会重启命令；
  从 Python API 调用时，请在导入框架之前于调用进程里设置这个变量。
- DPA 没有 `inference_mode`。UMA 专用的选项在严格配置下会被拒绝，而不是被忽略。

mliport 采用 MIT 许可。上游软件和模型需要另行引用，见[引用](../citations.md)。
