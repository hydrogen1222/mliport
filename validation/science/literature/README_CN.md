# 文献基准测试套件（档位 `t9`）

用**已发表的 DFT 参考数据**对 mliport 各引擎做可复现的精度测量。它是对现有科学验证框架的扩展，而不是第二套框架：复用 `scripts/common.py`、`scripts/engines.py`、`evidence/` 证据层、结果 schema 与聚合器，只新增一个档位（`t9`）、一个注册表、若干导入器和报告生成器。

本套件回答两个不同的问题，并且从不混为一谈：

| 问题 | 回答位置 |
| --- | --- |
| mliport 是否正确地驱动了后端？ | 每条 t9 记录的 `status` 与 `metrics.software_verification` |
| 模型相对 DFT 的精度如何？ | 记录下来的误差指标，并附带明确的比较模式 |

`status = pass` 表示**执行完整性**：请求的每个参考结构都通过 mliport 计算器完成评估且数值有限。它**不是**精度合格判定，套件中不存在任何凭空的 MLIP 误差阈值。

## 数据集

注册表 `registry.json` 是来源、许可、DFT 设置、单位、校验和与抽样规则的唯一事实来源。

| id | 参考文献 | DOI | 参考理论水平 | 使用的标签 | 数据许可 |
| --- | --- | --- | --- | --- | --- |
| `lips25` | LiPS-25（Fragapane & Deringer, JCTC 2026） | 10.1021/acs.jctc.5c02006 | PBEsol / VASP 6.4.3 | 能量、力、连线形成能 | 上游仓库无 LICENSE 文件 → **仅下载，绝不入库** |
| `lips20` | LiPS20 / EGraFFBench（Bihani 等, Digital Discovery 2024） | 10.1039/D4DD00027G | PBE-D3 / CP2K | 能量、力（virial 仅记录，不使用） | 上游 LICENSE 文件为空 → **仅下载，绝不入库** |
| `omat24` | OMat24 验证子集（Meta FAIR） | — | PBE(+U) / VASP | 能量、力 | CC-BY-4.0，子集已由本框架提交 |
| `jarvis` | JARVIS-DFT 3D（`jdft_3d`） | 10.6084/m9.figshare.6815699 | OptB88vdW / VASP | 每原子能量 | CC-BY-4.0 |
| `ms25` | MS25（Maxson 等, JCIM 2025） | 10.1021/acs.jcim.5c01262 | 按子集（论文） | 能量、力、应力（若提供） | 取数时被阻断（Materials Data Facility / Globus）→ **仅本地导入** |

`omat24` 不会重新下载：直接复用已提交的 `validation/science/data/omat24_subset.extxyz`（256 个结构，种子 20260911）；若哈希变化，导入器会拒绝运行。

## 这里编码的科学规则

1. **参考同源 vs 跨方法。** `omat24` 是 `reference_matched`：检查点本身在 OMat24 上训练，因此绝对能量误差可解释。`lips25`（PBEsol）、`lips20`（PBE-D3）与 `jarvis`（OptB88vdW）是 `cross_method`：绝对总能量含有任意偏移（赝势 + 泛函），因此套件报告*力*、*同组相对能量*、*形成能*与*排序*，并把绝对能量块标注为诊断量。不同数据集的跨方法绝对能量绝不会被平均成一个数。
2. **virial 不是 stress。** LiPS-25 与 LiPS20 都提供 `virial` 数组，但其符号/归一化约定在论文和仓库中均未说明。导入器对其做哈希并记录为来源信息，同时写入 `reference_virial_used_as_stress: false`；`metrics.stress` 报告为 `available: false`。只有当数据集带有**有据可查的应力标签**（MS25 本地导入可以）时才计算应力 MAE。
3. **不虚构标签。** 缺失的力/应力标签变成 `{"available": false, "reason": ...}`。LiPS-25 的 10 个无标签 `task1_unrelaxed` 结构只提供几何，使用 ASE `ExpCellFilter` + `BFGS` 弛豫后与 DFT 弛豫几何比较。
4. **用组均值消除偏移。** 相对能量在参考侧与模型侧分别减去**同一组**（成分/化学式/轨迹）的每原子能量均值，参考值永远不会进入模型预测。
5. **形成能遵循已发表协议。** LiPS-25 公式 1：`Ef/atom = (E_cell/(7-4x) - x·E_Li2S^fu - (1-x)·E_P2S5^fu)/(7-4x)`，两个端点取自同一能量集合。回归测试断言端点严格为 0，且重建结果复现论文的凸包。
6. **弛豫使用 ASE 上游算法**（不自造优化器），逐结构记录 `fmax`、步数预算与收敛标志。预算耗尽是 `converged: false`，绝不隐藏。

## 快速开始

```bash
# 1. 把固定参考数据导入本地缓存（不需要模型）
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25,lips20,omat24 \
  --engine mace --profile-id mace_omat \
  --model /path/to/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/literature --import-only

# 2. 运行完整 core 档位（导入 + t9 评估 + 聚合 + 报告）
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25,lips20,omat24 \
  --engine mace --profile-id mace_omat \
  --model /path/to/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/literature

# 3. 读取结果
sed -n '1,80p' .validation-work/literature/literature-report.md
python -c "import json;d=json.load(open('.validation-work/literature/literature-summary.json'));print(d['entries'][0]['dataset'])"
```

引擎始终通过自己的虚拟环境启动（`--venv`、`MLIPORT_SCIENCE_VENV_<ENGINE>` 或 `<repo>/.venv-<engine>`），与 `run_suite.py` 现有行为一致；编排脚本本身不导入任何后端。

## 档位

| 档位 | 文献数据 | 现有框架档位 |
| --- | --- | --- |
| `core` | 10 个弛豫晶体、4 个熔淬快照、24 个测试结构、80 帧 LiPS20、24 个 JARVIS 结构、32 个 OMat24 结构 | 无（适合 CI） |
| `standard` | 更大的确定性抽样（96 个 LiPS-25 测试结构、320 帧 LiPS20、120 个 JARVIS 结构、全部 256 个 OMat24） | `t1` 推理、`t4` relax/cellrelax/EOS、`t6` NVE 漂移 |
| `research` | 完整抽样（384 个 LiPS-25 测试结构、800 帧 LiPS20、640 个 JARVIS 结构） | `t1`、`t2`、`t2fd`、`t4`、`t5`、`t6`、`t7`、`t8`（各档位自身预算） |

精确抽样从注册表解析，写入 `<out>/resolved.json`，并在每个 `<out>/data/<dataset>/selection.json` 中重复记录种子、规则、被选中的源索引、上游 commit 与每个输入文件的 SHA-256。

## 引擎与能力门控

`t9` 与框架其余部分一样，通过 `engines.build_engine` → 产品代码的 `CalculatorFactory` 加载模型，覆盖四个固定引擎（`uma`、`mace`、`dpa`、`grace`）。不做任何静默替换：

* `--dtype` 只对 MACE 生效；其余引擎记录为 `upstream/model-defined`。
* `--head` 选择 DPA 分支 / UMA-MACE head，并完整记录。
* NEP **不**属于本框架：`model_manifest.json` 中没有固定的 NEP profile，其 ASE 与原生 GPUMD 运行时属于独立产品，`literature_eval.py` 会在加载任何东西之前拒绝 `--engine nep`。这是 `unsupported`/`blocked` 能力，绝不是科学测试失败。
* 缺少后端虚拟环境时，编排器写出 `blocked` 记录；模型无法加载属于 `blocked` 执行，而不是科学差异。
* 应力、NPT、NEB 与色散支持仍受现有框架的能力限制；`t9` 只测量计算器真正暴露的量。

## 数据管理

* 缓存位置：`--cache`，否则 `$MLIPORT_LITERATURE_CACHE`，否则 `~/.cache/mliport/literature`。上游原始文件与导入后的 extxyz **绝不**入库（默认就位于仓库之外）。
* 每个文件在下载后与每次复用前都按注册表 SHA-256 校验；不匹配时删除文件并报告期望值与实际值后中止。
* `--offline` 永不下载：复用已校验的缓存，或给出可用于填充缓存的命令后失败。
* 下载重试三次，写入进程唯一的 `.part` 文件后再原子替换目标文件。
* 无法获取的数据集记录为带原因的 `blocked` 结果记录，绝不伪装成空数据集。

## 输出

| 路径 | 内容 |
| --- | --- |
| `<out>/resolved.json` | 完全解析后的配置：档位计划、抽样、引擎虚拟环境、精确命令 |
| `<out>/data/<dataset>/selection.json` | 导入参考数据的来源（哈希、规则、种子、commit） |
| `<out>/data/<dataset>/fetch.json` | 每个上游文件的下载/复用状态 |
| `<out>/data/<dataset>/*.extxyz` | 归一化后的参考结构与参考标签 |
| `<out>/t9/*.json` | 每个数据集×划分×profile 一条 `mliport.beta-validation-result/2` 记录 |
| `<out>/t9/t9_<dataset>_<split>_<engine>.csv` | 逐结构误差 |
| `<out>/summary.json` | 全部记录的规范聚合（复用现有聚合器） |
| `<out>/literature-summary.json` | 机器可读的文献报告 |
| `<out>/literature-report.md`、`literature-report_CN.md` | 人类可读报告（英文 / 中文） |

## 如何阅读报告

* **E/atom** 是每原子总能量的直接差。只有模式为 `reference_matched` 时才可当作精度数字；否则明确标注为含任意跨方法偏移的诊断量。
* **Relative E** 是去除组均值后的每原子能量误差，衡量模型能否复现同一成分内的结构与热差异。
* **Force** 块报告分量 MAE/RMSE、逐原子矢量范数 P95/P99、参考力 > 0.1 eV/Å 时的方向夹角、按参考力大小分层以及按元素分解。
* **Formation energy** 块报告相对 LiPS-25 协议的 RMSE/MAE、相排序的 Spearman ρ，以及参考与模型各自位于凸包上的相。
* **Stress** 对所有没有可靠应力标签的数据集都是 `available: false`，并记录原因。
* **Limitations** 列出这些数字不能支撑的结论（跨方法偏差、缺少应力、原子迁移身份等）。

## MS25 与其他用户自备参考数据

MS25 无法下载：其 DOI 指向 Materials Data Facility 的页面，下载需要 MDF/Globus 客户端，DataCite 元数据中也没有权利声明。注册表将其记录为 `access: blocked`，唯一支持的路径是显式本地清单：

```json
{
  "schema": "mliport.literature-local-manifest/1",
  "dataset": "ms25",
  "subset": "mgo_surfaces",
  "license": {"id": "user-supplied", "redistributable": false},
  "reference_level": "PBE (VASP)",
  "dft": {"software": "VASP 6.4.3", "functional": "PBE", "cutoff_eV": 500},
  "comparison_mode": "cross_method",
  "units": {"energy": "eV", "forces": "eV/Angstrom", "stress": null},
  "files": [{"path": "mgo_surfaces.xyz", "sha256": "<64 hex>"}]
}
```

```bash
python validation/science/scripts/benchmark.py --tier core --datasets ms25 \
  --reference-manifest /data/ms25/manifest.json \
  --engine mace --profile-id mace_omat --model ... \
  --manifest validation/science/model_manifest.json --out .validation-work/ms25
```

导入器在读取任何一帧之前校验每个哈希，强制要求 DFT 元数据（从而不可能发生无标注比较），并拒绝没有 `SinglePointCalculator` 能量标签的文件。

## 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `DatasetAccessError: ... is not available in the local cache ... --offline` | 去掉 `--offline` 先运行一次，或用 `--cache` 指向已有缓存 |
| `... does not match the pinned sha256` | 上游变更或下载中断；文件会被自动删除，重新运行即可再次获取 |
| `--tiers t9 requires --dataset and --split` | `run_suite.py` 是底层入口；请使用 `benchmark.py`，或同时传入两个参数与 `--subset` |
| `E/atom MAE` 大得离谱（几百 eV） | 这是 `cross_method` 的绝对能量偏移，不是误差；请看相对能量与力 |
| `Stress: n/a` | 该数据集没有可靠的应力标签；virial 不会被转换 |
| 弛豫记录中 `converged: false` | 记录的步数预算耗尽；数字照实报告，不会静默延长 |
| 找不到后端虚拟环境 | 使用 `--venv`、`MLIPORT_SCIENCE_VENV_<ENGINE>` 或 `<repo>/.venv-<engine>` |

## 新增数据集

1. 在 `registry.json` 增加一个条目：引用/DOI、许可（论文**与**数据，并写明核查依据）、来源 commit/版本、DFT 设置、单位（包括应力是否真的标注）、带逐文件 SHA-256 的划分、各档位抽样规则、训练集重叠评估。
2. 在 `importers.py` 增加导入函数，写出归一化 extxyz 与 `selection.json`，并在 `import_dataset` 中注册。
3. 增加使用合成夹具的 CPU 测试（不联网）；若许可允许再分发，可提交一个很小的已知答案文件，否则数据只留在缓存中。
4. 运行 `python validation/science/scripts/benchmark.py --tier core --datasets <id> --import-only ...` 与 `pytest tests/mliport/mliport/test_literature_benchmark.py -q`。

## 本套件不做的主张

* 不对引擎排名：不同参考水平、数据集与抽样并列报告，绝不合并成单一分数。
* 不声称 MD 导出的性质已统计收敛；LiPS-25 的动力学任务（电导率、Arrhenius）属于研究级扩展，输运测量由现有 `t7` 分析套件负责。
* 不再分发许可不明确的数据集。
