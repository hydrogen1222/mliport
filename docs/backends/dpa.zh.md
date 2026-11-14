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
| `.venv-dpa` | 3.10, 3.11, 3.12, 3.13 | `deepmd-kit @ git+https://github.com/deepmodeling/deepmd-kit.git@0edeaef5617e9878edeb2346bf37bb6a035a4f51` (torch 2.11.0) |

```bash
./scripts/install_mliport.sh --engines dpa --device auto
```
<!-- END GENERATED -->

安装器固定上游提交 `0edeaef5617e9878edeb2346bf37bb6a035a4f51`（包版本
`3.2.1.dev63+g0edeaef56`）与 PyTorch 2.11.0。已发布的 3.2.0 wheel 能加载
DPA4-Neo，但不支持这些 DPA4C 检查点的 `gaussian/fix` 径向基。固定源码包含
上游对此基函数的实现；mliport 不改写检查点的设置或权重。

此安装路径需要 Git 和 C++ 编译工具链。使用 CUDA PyTorch 时，先配置 CUDA Toolkit
（设置 `CUDA_HOME` 或 `CUDA_PATH`，并把 `nvcc` 放入 `PATH`）。安装器将显式工具包路径
传给 CMake，使用选定的 PyTorch 构建，并关闭此环境中的 TensorFlow。V100 使用带
`sm_70` 内核的 `2.11.0+cu126` wheel。运行时还需要 e3nn、vesin/vesin-torch 和 mpich。

依赖包继续使用选定的镜像，固定源码需要访问 GitHub。离线安装从本地 `packages/`
目录读取版本完全一致的预构建 wheel，不访问 GitHub。

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

单任务的 DPA4-Neo、DPA4C-Nano、DPA4C-Mini 与 DPA-4.1C-Air 使用同一入口，
无需 `--head`，例如：

```bash
bin/mliport-dpa sp structure.cif --model-type dpa --task bulk --device cuda:0 \
  --model models/DPA4C-Nano-OMat24.pt
```

DeePMD 根据检查点参数的命名选择常规 PyTorch 或 PyTorch-exportable 求值器。
mliport 记录实际求值器、GPU 身份与参数精度，并对照选定分支核验 `.pt` 张量，
拒绝部分加载或训练张量发生变化。固定上游对 DPA4C 整数索引缓存的重建、
无桥接非自旋模型的描述符版本迁移，单独记录在 `checkpoint_migrations` 中。

## 可信训练 checkpoint 的兼容转换

部分 DPA-SSE 训练检查点在优化器状态和模型训练信息中包含 NumPy 标量。
DeePMD 明确使用 `torch.load(weights_only=True)`，因此无法直接读取。
**只有明确确认来源可信**时，才使用以下离线转换，原始文件不会被修改：

```bash
.venv-dpa/bin/python scripts/convert_dpa_checkpoint.py \
  models/SSE-VASP-dpa2.pt models/SSE-VASP-dpa2-inference.pt \
  --i-trust-this-file
```

转换脚本仅在显式授权后才执行可能运行任意代码的 pickle 反序列化，
不要用它处理不可信的模型文件。脚本会去掉优化器状态，把模型元数据中的
NumPy 标量转换为 Python 标量，并通过 `weights_only=True` 重新读取，
验证所有张量的名称、精度、形状和数值都没有变化。独立的
`.conversion.json` 记录原文件路径、两份 SHA-256 与转换项目。
这只能证明**文件兼容**，不能证明能量和力的预测等价；正式科研使用前，
仍需用真实 DeePMD 推理与可信参考结果进行对照。

2024 年 7 月的 `SSE-VASP-dpa2.pt` 还与当前网络存在参数布局差异，例如
检查点的 `(800, 128)` 矩阵在当前网络中为 `(640, 128)`。转换 NumPy 元数据不能
解决此问题；在匹配的旧版实现通过验证前，该文件仍不受支持。mliport 不重排权重，
也不初始化缺失的训练参数来强行运行它。

## 需要知道的事

- **多任务模型必须写 head，否则报错。** 没有默认分支。
- 模型信息检查失败时不会缓存计算器，再次尝试会重新加载并重新检查。
- **精度**由模型决定，mliport 记录后端实际报告的精度。
- **应力**只有所选模型和分支支持时才会输出。
- **DPA 需要每个进程单独指定 GPU。** 没有设置 `CUDA_VISIBLE_DEVICES` 时，命令行会自动新开进程；
  用 Python API 时，请在导入框架之前设置这个变量。
- DPA 没有 `inference_mode`。严格模式下使用 UMA 专用的参数会报错。

mliport 采用 MIT 许可证。上游软件和模型需要另外引用，见[引用](../citations.md)。
