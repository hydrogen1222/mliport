# DPA（DeepMD-kit）

DPA 模型通过 `deepmd-kit` 加载。很多 DPA 模型是多任务模型，带有多个分支（head）。用哪个分支
必须由你指定，不指定会直接报错。

## 模型是什么

- **结构。** DPA-3 是建立在"线图序列"上的消息传递网络：第一张图是原子和原子对，下一张
  是键角，依此类推，所以角度信息是显式构造进去的
  （[Zhang 等，2025](https://arxiv.org/abs/2506.01686)）。
- **示例检查点的训练数据。** `DPA-3.1-3M`（约 330 万参数）在 OpenLAM-v1 上预训练，这是 31
  个来自不同领域的数据集的集合，它们用的是*不同的* DFT 设置。模型用
  数据集编码把它们区分开，这个编码就是你选的 **head**：`Omat24` 复现 OMat24 的参考水平
  （PBE / PBE+U，体相无机，见 [MACE 页面](mace.md)）。
- **力**是原子能量之和的梯度（保守力）。
- **使用时要注意。** 选和你要对比的 DFT 计算设置一致的 head。不同 head 给出的绝对能量不能
  互相比较，因为每个数据集的能量零点不同，泛函也可能不同。用下面的
  `dp --pt show ... model-branch` 列出检查点的 head。

## 安装

<!-- BEGIN GENERATED: environment -->
| 环境 | Python | 运行时 |
|---|---|---|
| `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit==3.1.3` (torch 2.10.0) |

```bash
./scripts/install_mliport.sh --engines dpa --device auto
```
<!-- END GENERATED -->

加载模型需要 `deepmd-kit[torch]` 的元数据，安装脚本会同时装上对应版本的 `mpich`。

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

查看模型有哪些分支，用 DeePMD-kit 的命令，再把要用的分支传给 `--head`（INCAR 里写 `HEAD`）：

```bash
dp --pt show DPA-3.1-3M.pt model-branch
```

## 需要知道的事

- **多任务模型必须写 head，否则报错。** 没有默认分支。
- 模型信息检查失败时不会缓存计算器，再次尝试会重新加载并重新检查。
- **精度**由模型决定，mliport 记录后端实际报告的精度。
- **应力**只有所选模型和分支支持时才会输出。
- **DPA 需要每个进程单独指定 GPU。** 没有设置 `CUDA_VISIBLE_DEVICES` 时，命令行会自动新开进程；
  用 Python API 时，请在导入框架之前设置这个变量。
- DPA 没有 `inference_mode`。严格模式下使用 UMA 专用的参数会报错。

mliport 采用 MIT 许可证。上游软件和模型需要另外引用，见[引用](../citations.md)。
