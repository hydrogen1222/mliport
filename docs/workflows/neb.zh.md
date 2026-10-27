# NEB 与 CI-NEB

`mliport neb` 在固定晶胞下用 NEB 方法找两个端点结构之间的最小能量路径，可以再用 climbing image
精修鞍点，并给出势垒。

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace --images 7 \
  --climb --output runs/neb
```

需要两个原子顺序一致的端点结构（顺序不同时用 `--atom-map` 指定对应关系）、一个指定了后端的模型，
以及足够放下 `images + 2` 个结构计算的内存。

## 优化开始前的检查

NEB 要求模型的力等于能量的负梯度。如果不满足，能带会按照和势能面不一致的力去优化，得到的势垒
没有意义，而且从结果上看不出异常。所以 mliport 会先做检查。

检查在要优化的能带上进行（两个端点和插值得到的 image）：沿几个方向（力的方向和随机方向），比较
力的投影 `-F·u` 和能量的中心差分。用两种步长做 Richardson 外推，容差考虑了模型精度带来的舍入
误差。以下情况会拒绝运行：

- 力不是能量的梯度，
- 出现了非有限的值，
- 能量分辨率不足以下结论（比如较大的 float32 晶胞），
- 模型是直接预测力而不是对能量求导（预测器报告 `direct_forces = true`，这类模型一律被拒绝）。

检查结果保存在 `run_context.json` 和 `neb_results.json` 的 `gradient_preflight` 里。这只是在几个结构
和方向上的抽查，不能证明整个势能面都满足，也不说明模型的精度。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--initial`、`--final` | 两个端点结构（除非是续算，否则必需） |
| `--images` | 中间 image 的个数，不含端点（默认 7） |
| `--climb` | 用 climbing image 精修鞍点（默认关闭） |
| `--fmax`、`--max-steps` | 收敛阈值 eV/Å（默认 0.03）和步数上限（默认 1000） |
| `--pre-fmax`、`--pre-max-steps` | 配合 `--climb`：启用 climbing image 之前先用较宽松的阈值优化（默认 0.1 eV/Å，300 步） |
| `--spring`、`--maxstep` | 弹簧常数（默认 0.1 eV/Å²）和优化器最大步长（默认 0.1 Å） |
| `--interpolation` | 初始路径用 `linear`（默认）或 `idpp`；用 `--idpp-fmax`、`--idpp-steps`、`--idpp-mic` 调节 IDPP |
| `--path-convention` | `mic`（默认）或 `unwrapped` |
| `--endpoint-policy` | `validate`（默认）或 `relax`，见下文 |
| `--endpoint-fmax`、`--endpoint-steps` | 端点力阈值（默认 0.02 eV/Å）和弛豫步数预算（默认 500） |
| `--min-distance` | 有原子间距小于该值的 image 会被拒绝（默认 0.5 Å） |
| `--atom-map` | 按初态原子顺序列出对应的末态原子编号（从 0 开始），逗号分隔 |
| `--image-shifts` | 每个原子的整数晶格平移，例如 `0,0,0;1,0,0` |
| `--checkpoint-interval` | 每 N 步保存一次整条能带（默认 10） |
| `--resume` | 从检查点或 NEB 运行目录继续 |

原子对应关系和晶格平移必须是整数数组，在 Python 里直接构造 `BandInput` 也一样。浮点或布尔数组会被
拒绝，以免自动转换后改变了原子对应关系或路径。

### 端点：检查还是优化

默认的 `--endpoint-policy validate` 要求端点已经优化好：任何一端的受力超过 `--endpoint-fmax`，程序会
停下并指出是哪一端。`relax` 会先优化两个端点再建能带。两种方式得到的参考能量不同，请记录用的是
哪种。如果直接提供了整条能带，`relax` 不会改动它。

用 `--interpolation idpp` 时，判断是否达到 `--idpp-fmax` 用的是包含弹簧项的完整 IDPP 能带力。只看原始
的成对作用力很小，并不代表已经收敛；IDPP 没有完成的话会报错。

## 输出

`neb_results.json` 里有能量、力、收敛状态和势垒（`barrier_forward_sampled_eV`、
`barrier_reverse_sampled_eV`、`reaction_energy_eV`，以及说明有没有找到中间峰值的 `barrier_status`）。
另外还有 `run_context.json`、`resolved_config.json`、`artifacts.json`，保存完整能带检查点的
`checkpoints/` 目录，以及每个 image 一个 `POSCAR` 的 `vasp_path/` 目录，可以直接用 VASP 的 NEB 工具
处理。

势垒是直接从能带上读的：能量最高的 image 减去端点。mliport 不做鞍点拟合，拟合相关的字段留空。

ASE 的轨迹文件不保存自定义数组，所以检查点会把 `atom_id` / `atom_ids` 存到每个 image 的 `info` 里，
调用者的原始数组不受影响。修正后的 IDPP 收敛判据对应 NEB 科学版本 2，版本 1 的检查点不能直接续算。

## 失败或没收敛

没收敛的能带会标为未收敛，同时给出目前最好的能带和原因，这时的势垒不能当作结果。原子对应关系错误
或两个端点不匹配，会在建能带之前就报错。续算会锁定原来的运行目录，避免误用不同的设置重新开始。

## 怎么看待势垒

- NEB 收敛后找到的鞍点只是候选。要确认是过渡态，还需要另外做 Hessian（频率）分析。
- 只有用同一个模型和 head、同样的 NEB 设置（image 数、插值方法、端点处理、弹簧常数）算出的势垒
  才能相互比较。
- 端点必须是你真正想研究的状态，原子对应关系在化学上也要合理，势垒才有意义。

## 各入口支持的功能

并非每个 NEB 选项都能在每个入口设置。下表由 `validation/science/capability_matrix.json` 生成。

<!-- BEGIN GENERATED: neb-capability -->
| 功能 | API | 直接 CLI | INCAR | TUI |
|---|---|---|---|---|
| endpoints | yes | yes | yes | yes |
| atom map | yes | yes | no | yes |
| image shifts | yes | yes | no | yes |
| resume | yes | yes | no | yes |
| path convention | yes | yes | yes | yes |
| interpolation | yes | yes | yes | yes |
| method climb | yes | yes | yes | yes |
| pre relaxation | yes | yes | yes | yes |
| min distance guard | yes | yes | yes | yes |
<!-- END GENERATED -->

`yes` 只表示这个入口能设置该功能，不代表各入口结果等价或已经验证。`no` 和 `partial` 在那个文件里有
说明。例如原子对应关系和晶格平移只能通过 API、命令行和 TUI 设置，INCAR 文件不支持。

## 续算

```bash
.venv-mace/bin/mliport neb --resume runs/neb --output runs/neb
```

续算在原来的目录里继续，并核对记录下来的信息：模型、task、head、精度，晶胞和 PBC，约束，能带
设置和各项参数。当前 beta 版本里步数上限也在核对范围内，续算时改 `--max-steps` 会报
`Resume fingerprint is incompatible` 并拒绝运行，检查点保持不变。
