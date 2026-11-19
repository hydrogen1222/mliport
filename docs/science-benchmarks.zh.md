# 文献基准测试

mliport 提供文献基准测试套件，用已发表的 DFT 参考数据（LiPS-25、LiPS20、OMat24、JARVIS-DFT，以及本地导入的 MS25）测量各引擎的精度。它在现有科学验证框架上新增一个档位 `t9`，并复用其结果 schema、证据聚合器与后端隔离机制。

两个问题保持分离。`status=pass` 表示计算通过 mliport 正确执行且数值有限，并不是精度合格判定。精度由记录下来的误差指标表示，并始终标注比较模式。

## 快速开始

```bash
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25,lips20,omat24 \
  --engine mace --profile-id mace_omat \
  --model /path/to/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/literature
```

该命令把固定版本的参考数据导入本地缓存，在 core 抽样上评估模型，聚合证据，并在 `<out>` 中写出 `literature-report.md`、`literature-report_CN.md` 与 `literature-summary.json`。

## 数据集与许可

| 数据集 | 参考文献 | 参考理论水平 | 使用标签 | 数据许可 |
| --- | --- | --- | --- | --- |
| LiPS-25 | Fragapane & Deringer, JCTC 2026 | PBEsol / VASP | 能量、力、形成能 | 不明确（上游无 LICENSE） |
| LiPS20 | Bihani 等, Digital Discovery 2024 | PBE-D3 / CP2K | 能量、力 | 不明确（LICENCE 为空） |
| OMat24 | Meta FAIR 验证子集 | PBE(+U) / VASP | 能量、力 | CC-BY-4.0 |
| JARVIS-DFT | NIST jdft_3d | OptB88vdW / VASP | 每原子能量 | CC-BY-4.0 |
| MS25 | Maxson 等, JCIM 2025 | 按子集 | 能量、力、应力 | 仅本地导入 |

数据许可不明确的数据集在运行时下载，绝不提交进仓库。权威来源信息（DOI、上游 commit、逐文件 SHA-256、DFT 设置、单位）位于 `validation/science/literature/registry.json`。

只有 OMat24 是 `reference_matched`：检查点在其上训练，因此绝对能量误差可解释。LiPS-25、LiPS20 与 JARVIS-DFT 是 `cross_method`，套件因此报告力、同组相对能量、形成能与排序，并把绝对能量块标注为诊断量。除非数据集明确给出应力标签，存储的 virial 绝不会被转换成 stress。

## 档位

| 档位 | 文献抽样 | 附加验证档位 |
| --- | --- | --- |
| `core` | 每个数据集的确定性小样本 | 无，适合 CI |
| `standard` | 更大的确定性样本 | 推理、弛豫/EOS、短 NVE 漂移 |
| `research` | 完整样本 | `t1`、`t2`、`t2fd`、`t4`、`t5`、`t6`、`t8`；T7 输运为独立的两阶段手动任务 |

每个抽样都记录在 `resolved.json` 以及各 `data/<dataset>/selection.json` 中，包含种子、规则与上游文件哈希，因此报告中的数字可以追溯到产生它的确切结构。

## 阅读结果

- `E/atom` 是总能量的直接差；只在 `reference_matched` 数据集上作为精度解读。
- 相对能量在两侧同时去除组均值，衡量同一成分内的结构或热能量差异。
- 力块报告分量 MAE/RMSE、逐原子矢量 P95/P99、方向夹角、按参考力大小分层以及按元素误差。
- 形成能遵循已发表的 LiPS-25 协议，包括相排序与凸包一致性。
- 没有可靠 DFT 应力标签时，应力报告为不可用。
- 被阻断的数据集与不支持的性质都会连同原因列出，绝不静默丢弃或伪造。
- `--max-structures N` 可限制试跑规模；记录的 `n_structures`、
  `sampling.n_structures` 与逐结构 CSV 始终对应真正被评估的结构。

## 数据、缓存与离线使用

```bash
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25 --import-only \
  --engine mace --profile-id mace_omat --model /path/to/weights \
  --manifest validation/science/model_manifest.json \
  --cache /data/mliport-literature --offline \
  --out .validation-work/literature-lips25
```

缓存默认位于 `~/.cache/mliport/literature`，可用 `--cache` 或 `MLIPORT_LITERATURE_CACHE` 指定。`--offline` 只复用已校验的缓存，并给出可用于填充缓存的命令后失败。哈希不匹配会删除文件并报告期望值与实际值后中止。

## 故障排查

| 现象 | 原因与处理 |
| --- | --- |
| `--offline` 时缺少缓存条目 | 去掉 `--offline` 先运行一次，或用 `--cache` 指向已有缓存 |
| 固定 sha256 不匹配 | 过期或不完整的下载已被删除；重新运行即可再次获取 |
| `E/atom` 出现几百 eV | 跨方法偏移而非误差；请看相对能量与力 |
| `Stress: n/a` | 该数据集没有可靠的应力标签；virial 不会被转换 |
| 找不到后端虚拟环境 | 传入 `--venv`、设置 `MLIPORT_SCIENCE_VENV_<ENGINE>`，或创建 `.venv-<engine>` |

## 另见

- 完整指南与注册表说明：[literature suite README](https://github.com/hydrogen1222/mliport/blob/main/validation/science/literature/README_CN.md)
- 现有科学验证档位：[validation](validation.md)
- 引用与许可细节：[citations](citations.md)
