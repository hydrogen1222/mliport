# 分子动力学

`mliport md` 用选定的后端跑 NVE、NVT 或 NPT 动力学，生成的轨迹可以直接交给[分析](../analysis/index.md)工具。

```bash
.venv-mace/bin/mliport md structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --ensemble NVT --temp 300 --timestep 1.0 --steps 1000 \
  --equilibration-steps 500 --save-interval 10 \
  --output runs/md
```

需要一个周期性结构、一个模型，并指定后端。对分子做 MD 要用分子 task（`omol` / `molecule`），并设好
电荷和自旋。

## 各向同性 NPT 晶胞平衡（仅 ASE）

UMA/MACE/GRACE/DPA 的 `--ensemble NPT` 使用 ASE 的 **Martyna–Tobias–Klein (MTK)** 恒温恒压积分器，并非 Langevin/Bussi/NHC：

```bash
mliport md structure.cif --model model.pt --model-type mace \
  --ensemble NPT --temp 800 --pressure-gpa 0 \
  --npt-tdamp 100 --npt-pdamp 1000 \
  --timestep 0.5 --steps 50000 --save-interval 10 \
  --no-pre-relax --output runs/npt-800K
```

`--npt-tchain`、`--npt-pchain` 默认均为 3，`--npt-tloop`、`--npt-ploop` 默认均为 1。阻尼时间单位为 fs；**外压正值代表压缩（GPa）**，程序转换为 ASE 的 eV/Å³，并记录配置。NPT 不支持约束、分子/表面（非三维 PBC）、零温度或不提供应力的模型，也不支持 NEP/原生 GPUMD。自动质心策略只在初始化时清除一次总动量，不添加 ASE FixCom；MTK 的恒温设定点考虑减少的三个动能自由度。

标准 XDATCAR 只适用于定胞轨迹。NPT 会记录原因并禁用 XDATCAR；`raw/trajectory.traj` 保留逐帧晶胞和动量；`md.csv` 保存体积和**包含动能项的总压强**。NPT 可变胞轨迹明确标记为**不可直接用已有定胞 MSD/EH/输运分析计算扩散系数**。推荐在目标温度、压强下充分平衡，统计已平衡时间窗的**平均晶胞**并检查收敛，然后用该晶胞**另起一次**定胞 NVT/NVE 生产模拟；NPT 最后一帧并不等于平均晶胞。当前保存帧不能恢复 MTK 恒温链与压强耦合内部状态，所以从该帧重启并不等价于严格续算。对各向异性晶体或相变，只有各向同性体积控制可能不足。

## 常用选项

| 选项 | 含义 |
|---|---|
| `--ensemble` | `NVT`（默认）、`NVE` 或 `NPT` |
| `--temp` | 目标温度，单位 K（默认 300） |
| `--timestep` | 积分步长，单位 fs（默认 1.0） |
| `--steps` | 采样阶段的步数（默认 1000） |
| `--equilibration-steps` | 采样前在同一系综下的平衡步数（默认 0） |
| `--thermostat` | `LANGEVIN`（默认）、`BUSSI` 或 `NHC` |
| `--friction`、`--bussi-tau`、`--nhc-tdamp`、`--nhc-tchain`、`--nhc-tloop` | 恒温器参数 |
| `--save-interval` | 每 N 步写一帧（默认 10） |
| `--seed` | 随机种子；省略时自动生成并记录 |
| `--velocity-policy`、`--com-policy` | 速度和质心运动的处理方式 |
| `--pre-relax`、`--pre-relax-steps`、`--pre-relax-fmax` | MD 前先做一次简短的原子位置优化（NVT 默认开启，NVE 默认关闭；50 步，0.1 eV/Å） |
| `--fmax-abort` | 任何原子受力超过这个值（eV/Å）就停止（默认 20） |
| `--write-outcar`、`--write-xdatcar`、`--write-trajectory` | 是否写出这些文件 |

### 预优化没收敛会直接停止

如果预优化在 `--pre-relax-steps` 步内没有达到 `--pre-relax-fmax`，程序会**在正式 MD 开始前停下**。
从离极小点还很远的结构开始跑 MD，通常不是你想要的。报错信息会给出几种处理办法：增加步数；在有
理由的前提下放宽阈值；或者明确关闭预优化。预优化的结果记录在 `md_provenance.pre_relax` 里。

## 步长和保存间隔

`--timestep` 是积分步长，`--save-interval` 决定每隔多少步保存一帧。步长 1 fs、`--save-interval 10`
就是每 10 fs 存一帧。输运分析用的是**保存帧之间的时间间隔**。

平衡阶段不要计入统计：用 `--equilibration-steps` 设置平衡步数，只分析采样阶段。NVE 和 NVT 是不同
的动力学，两种轨迹不能放在一起平均。

## 恒温器和质心运动

`--thermostat` 选择 NVT 的控温方法。Langevin 带随机力，Bussi 和 NHC 是确定性的。扩散系数会受恒温器
影响，报告时要写明用的是哪种。

NHC 有几点要注意。它不能和 FixCom 或其他约束一起用。默认的 `--com-policy auto`（等同于
`initialize_only`）在初始化速度后把质心动量清零一次，之后 NHC 会保持质心动量为零。这样体系有
`3N-3` 个动能自由度，恒温器的设定温度相应乘以 `(3N-3)/3N`（记录为 `nhc_set_temperature_K`）。如果
设成 `--com-policy none`，初始速度里随机的质心动量会保留下来，整个体系会匀速漂移；分析时不扣除漂
移的话，可移动离子的 MSD 会被它主导。

约束和固定原子由计算器和运行程序施加，报告扩散结果时要写清楚。

## 接着跑

`md` 没有 `--resume` 参数。要接着之前的轨迹跑，读取一个带速度的保存帧，从它开始一次**新的**运行：

```bash
.venv-mace/bin/mliport md last_frame.traj --model mace-omat-0-medium.model \
  --model-type mace --velocity-policy preserve --no-pre-relax --output runs/md-part2
```

新运行保留输入的位置和速度，不会重新按 Maxwell–Boltzmann 分布生成速度。它有单独的运行目录和记录，
第一帧和上一段的最后一帧相同（验收测试检查的就是这一点）。

## 什么情况下会停止

任何原子受力超过 `--fmax-abort` 或出现 NaN，计算会停止，坏掉的帧不会写进文件。写输出失败时同样会
停止。模型或结构缺失、参数组合不支持，会在 MD 开始前就报错。

验收测试在四个后端上各跑了 5 步 NVE，以及 5 步 Langevin、Bussi、NHC 的冒烟测试。这些只检查流程
和恒温器能否正常运行，不能说明温度已经平衡。

## 一个接近实际使用的例子

```bash
.venv/bin/mliport md structure.cif --model uma.pt --model-type uma --task omat \
  --ensemble NVT --temp 800 --timestep 1.0 \
  --equilibration-steps 5000 --steps 50000 --save-interval 10 \
  --thermostat NHC \
  --output runs/md-800K
```

输出文件的说明见[输出文件](../outputs.md)：`raw/` 下是完整轨迹和 `md.csv`，`vasp/` 下是 VASP 格式的
文件，`artifacts.json` 列出所有文件及其单位和哈希。
