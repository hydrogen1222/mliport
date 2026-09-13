# mliport

mliport 用统一的、backend-neutral 的 CLI、TUI 与 Python API 运行 MACE、
DPA (DeepMD-kit)、GRACE (tensorpotential) 与 UMA (FAIRChem) 等机器学习势。它提供 VASP 风格的输入/输出,支持单点、结构弛豫、分子动力学、NEB 与
轨迹分析,并显式记录 provenance、对配置 fail closed。

**mliport 不是模型、不是训练器、也不是 DFT 程序。** 它运行第三方势;
模型精度不等于 mliport 的精度。

## 为什么用 mliport

- **真正 backend-neutral。** 不存在隐式 UMA 默认:backend 必须来自
  `--model-type` 或 model alias/profile,否则 fail closed。
- **诚实的 provenance。** 每次运行记录 model identity(路径、task、head、
  dtype、可用时的 SHA-256)、请求设备 vs 实际设备、配置解析来源与软件
  commit。
- **VASP 风格,但不是 VASP。** INCAR 风格键与 OUTCAR/OSZICAR/CONTCAR/
  XDATCAR 文件只提供熟悉感;没有 SCF、k 点、ENCUT,也不做数值等价声明。
- **严格的科学配置。** 拼错的键或跨 backend 选项默认是致命错误
  (`--lenient-config` 需显式开启宽松模式)。
- **证据优先。** 验证声明绑定到记录下来的 campaign、commit 与 model
  identity——包括诚实的 `not_run`。

## 它不能替代什么

DFT 参考计算、模型训练/微调、k 点/ENCUT 收敛测试、过渡态频率验证,以及
"这个势是否适合你的化学体系"这一科学判断。见 [科学边界](#科学边界)。

## 功能

| 领域 | 内容 |
|---|---|
| Backend | MACE、DPA、GRACE、UMA(UMA 的 `fairchem` 别名) |
| 计算 | 单点 `sp`、弛豫 `opt`、MD `md`、NEB `neb`、批量 `batch`、INCAR 风格 `run` |
| 分析 | `msd`、`transport`、`electrolyte`(GEMDAT)、`rdf`、`rmsd`、`vacf`、`spectrum`、`arrhenius`、`thermo`、`density`、`validate` |
| 界面 | CLI、Textual TUI(`tui`)、Python API |
| 运维 | 本地 `queue`、`jobs`、`kill`、`clean`、`doctor`、`setup`、`config`、`template` |
| 输出 | VASP 风格文件、ASE trajectory、JSON 结果与 artifact manifest |

## 安装

mliport 安装在**每个后端自己的 Python 环境内**,不是跨环境调度器。支持的
安装路径是安装脚本:

```bash
git clone https://github.com/hydrogen1222/mliport
cd mliport
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
```

核心 CI 覆盖 **3.10-3.12**;每个 backend 还有自己的约束(UMA 需要
>=3.11),`--python` 必须满足所有请求的后端,否则安装脚本 fail closed。
环境名、Python 范围与运行时锁定由
[docs/installation.md](docs/installation.md) 机器生成。

用对应后端的环境(或 launcher)运行 CLI:

```bash
.venv-mace/bin/mliport sp structure.cif --model model.model --model-type mace
.venv/bin/mliport      sp structure.cif --model uma.pt     --model-type uma
./bin/mliport-mace     ...   # 生成的 launcher,只负责选择 runtime
```

Windows 下每个环境提供 `Scripts\mliport.exe`。CPU-only、全后端、离线与手动
安装见 [docs/installation.md](docs/installation.md)。

## 5 分钟快速开始

先用 MACE 只是因为其 checkpoint 开放且轻量;其他 backend 完全等价。

```bash
# 1. doctor(运行时不可用会 fail closed)
.venv-mace/bin/mliport doctor --engine mace --device auto

# 2. 单点
.venv-mace/bin/mliport sp structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --output runs/mace-sp

# 3. 短弛豫
.venv-mace/bin/mliport opt structure.cif \
  --model mace-omat-0-medium.model --model-type mace \
  --fmax 0.05 --max-steps 50 --output runs/mace-opt
```

等价的第一次计算:

```bash
.venv-dpa/bin/mliport   sp structure.cif --model DPA-3.1-3M.pt --model-type dpa --head Omat24
.venv-grace/bin/mliport sp structure.cif --model GRACE-2L-OMAT-medium-base --model-type grace
.venv/bin/mliport       sp structure.cif --model uma-s-1p2.pt --model-type uma --task omat
```

之后查看 `runs/mace-opt/mliport_results.json` 与
`runs/mace-opt/resolved_config.json`;所有文件、单位与 provenance 字段见
[docs/outputs.md](docs/outputs.md)。

## 选择 backend

| Backend | 环境 | 固定示例 profile | 精度 / head 语义 |
|---|---|---|---|
| MACE | `.venv-mace` | `mace-omat-0-medium.model`(`mace_omat`) | float64 为主;float32 是独立 identity;`HEAD` 可选 |
| DPA | `.venv-dpa` | `DPA-3.1-3M.pt`(`dpa_omat`) | 多任务 checkpoint 必须显式 `--head`(如 `Omat24`) |
| GRACE | `.venv-grace` | `GRACE-2L-OMAT-medium-base`(`grace_omat`) | `--model` 指向 SavedModel 目录 |
| UMA | `.venv` | `uma-s-1p2.pt`(`uma_omat`) | 显式 UMA task family(`omat`/`omol`/...);`fairchem` 是别名 |

每个 backend 的 profile、stress/head 语义、引用与限制:
[docs/backends/](docs/backends/mace.md)。

## 主要 workflow

- [单点](docs/workflows/single-point.md)
- [结构弛豫](docs/workflows/optimization.md)
- [分子动力学](docs/workflows/md.md)
- [NEB / CI-NEB](docs/workflows/neb.md)
- [批量](docs/workflows/batch.md)
- [队列](docs/workflows/queue.md)
- [分析总览](docs/analysis/overview.md)

## 科学边界

- 没有 SCF,没有 k 点/ENCUT/POTCAR,不做 DFT 数值等价声明。
- 绝对能量依赖模型;形成能与势垒必须在同一 model/head/reference 组合内比较。
- OMat24 benchmark 只刻画固定模型在该 benchmark 上的表现,不会自动外推到
  其他化学体系。
- MD 导出的 transport 量依赖采样;分析会记录窗口、drift 策略与不确定度。
- 收敛的 NEB band 给出的是 saddle **候选**,不是已验证的过渡态。

见 [docs/models.md](docs/models.md) 与 [docs/validation.md](docs/validation.md)。

## 验证状态

发布状态:**beta candidate**。

高级显式 atom mapping / image-shift 控制只能通过 API/direct CLI/TUI 表达;
INCAR 风格键集刻意不暴露 `neb_atom_map`/`neb_image_shifts`。包含这些
expressibility 状态的机器可读 capability matrix 位于
`validation/science/capability_matrix.json`。

<!-- BEGIN GENERATED: validation/science/reports/README_VALIDATION.md -->
Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

Validation status per backend, rendered from the beta evidence records (`beta-summary.json`; t1-t8 tiers, 4 backends x OMat24 common subset). `software_validated` means the mliport integration and all recorded checks passed; `model_characterized` means the workflow ran and its behavior was recorded, including honest failures (e.g. float32 arithmetic noise). Full per-test tables: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md).

| Workflow | MACE | DPA | GRACE | UMA |
|---|---|---|---|---|
| Install & doctor (CI software tests) | software_validated* | software_validated* | software_validated* | software_validated* |
| Single-point inference (4 structures) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Energy-forces consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) |
| Stress (finite-difference cross-check) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx4, passx3) |
| Stress-energy consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx4) |
| Coordinate invariance & cache | not_run | not_run | not_run | not_run |
| Fixed-cell relaxation | not_run | not_run | not_run | not_run |
| Cell relaxation | not_run | not_run | not_run | not_run |
| EOS / bulk modulus | not_run | not_run | not_run | not_run |
| Elastic constants | not_run | not_run | not_run | not_run |
| Harmonic phonons | not_run | not_run | not_run | not_run |
| Harmonic thermodynamics | not_run | not_run | not_run | not_run |
| Vacancy formation energy | not_run | not_run | not_run | not_run |
| Surface energy | not_run | not_run | not_run | not_run |
| NEB | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Saddle-point Hessian | not_run | not_run | not_run | not_run |
| Short NVE / NVT MD | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) |
| Transport analysis (demonstration) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) | model_characterized (characterizedx1, passx1) |
| Mechanism analysis (GEMDAT) | model_characterized (characterizedx1) | model_characterized (characterizedx1) | model_characterized (characterizedx1) | model_characterized (characterizedx1) |
| Performance (SP/MD scaling) | software_validated (passx5) | software_validated (passx5) | software_validated (passx10) | software_validated (passx5) |

Held-out OMat24 accuracy: not run on this evidence set.

`*` = CI software test only, no model involved. A cell lists the recorded statuses for that workload; per-workload rows reuse the same evidence tiers, so row counts are not additive. Full per-test tables and limitations: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md). Model identities pinned in `validation/science/model_manifest.json`.
<!-- END GENERATED -->

## Python API

```python
from mliport import run_single_point, run_optimization, run_md, run_neb
from mliport import calculate_energy

result = run_single_point(
    structure="structure.cif",
    model="mace-omat-0-medium.model",
    model_type="mace",
)
```

API 同样要求显式 backend,并使用与 CLI 相同的 resolver 与 provenance 规则。

## 文档

完整文档在 [docs/](docs/index.md):安装、配置、模型/backend、workflow、分析、
输出、验证、troubleshooting 与 [mlipx 迁移指南](docs/migration-from-mlipx.md)。

## 引用

mliport 目前没有自己的 DOI。请引用 mliport(见
[docs/citations.md](docs/citations.md))以及实际运行的上游模型/软件;上游 DOI
永远不属于 mliport。机器可读元数据:[CITATION.cff](CITATION.cff)。

## 许可证

MIT,见 [LICENSE.md](LICENSE.md)。上游模型与框架保留各自许可证,再分发或
商用前请自行确认。

## 致谢

mliport(`hydrogen1222/mliport`)是独立项目,与 PyPI 上另一个同名
`mliport` 项目无关。
