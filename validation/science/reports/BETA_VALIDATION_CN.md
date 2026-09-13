# mliport beta 科学验证报告

由 `.validation-work` 下的证据记录生成；证据对应提交 `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`。本文档全部表格由 `validation/science/scripts/generate_beta_report.py` 从结果 JSON 渲染，更新方式是重新生成并 diff，不要手工编辑。

重新认证状态：**target_commit_revalidated** —— campaign manifest declares completion and every record was produced at the target software commit

- software_commit（声称被验证）：`5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`
- validation_code_commit：`6e1a945f91c76a4f17be10ed189fba0232936514`
- report_generator_commit：`6e1a945f91c76a4f17be10ed189fba0232936514`
- evidence_campaign：`20260913-current-head-5f8d91d`
- evidence_source_commits：`5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`

模型身份与产物哈希固定在 `validation/science/model_manifest.json`；OMat24 评估子集见 `validation/science/data/`（种子 20260911）。

## Beta GO / NO-GO 清单（任务书 §28）

- 结论：**beta GO**（24 ✅ / 1 ⚠️ / 0 ❌）
- 科学 campaign target（full campaign）：`5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`
- Release candidate commit（bridge）：`6e1a945f91c76a4f17be10ed189fba0232936514`
- 证据 campaign：`20260913-current-head-5f8d91d`，101 条记录（58 pass / 43 characterized / 0 fail / 0 blocked）

| # | 条目 | 状态 | 证据 | evidence query |
|---|---|---|---|---|
| 1 | 新项目名无已知同领域 namespace collision | ✅ | `mliport` 在 PyPI 镜像与 GitHub 搜索均可用；仓库已改名 `hydrogen1222/mliport` | `external=pypi:mliport,github:hydrogen1222/mliport` |
| 2 | Python package / CLI / repo identity 完成迁移 | ✅ | 改名提交链；clean break，旧 `mlipx` import 不再发布 | `canonical=record_counts; scope=current_campaign` |
| 3 | 目标提交 CI 3.10/3.11/3.12 全绿 | ✅ | `5f8d91d4` 及后续提交的 tests/lint/package-build 全绿 | `ci=github_actions:tests,lint,package-build; python=3.10,3.11,3.12; commit=5f8d91d4` |
| 4 | wheel clean install 全绿 | ✅ | wheel-install-smoke 3.10/3.11/3.12 + `mliport-2.0.0b3` capability smoke | `ci=wheel-install-smoke; python=3.10,3.11,3.12` |
| 5 | strict evidence loader 唯一 | ✅ | `validation/science/evidence/` + report 不得直接读 raw record 的静态测试 | `loader=evidence.load_evidence; raw_selection=forbidden` |
| 6 | report/README 不再直接解释 raw records | ✅ | 报告与 figures 全部消费 `load_evidence`；README block 机器渲染 | `renderer=generate_beta_report; readme_block=generated` |
| 7 | malformed evidence fail-closed | ✅ | V01-T5/T6/T7 测试；report 退出码 2/3；archive builder 拒绝 | `tests=V01-T5,V01-T6,V01-T7; exit_codes=2,3` |
| 8 | profile identity 不混 precision/model/commit | ✅ | V01-T2/T3/T4；engine 单元格显示 `mixed` 并附 `worst_status` | `tests=V01-T2,V01-T3,V01-T4; identity=precision,model,commit` |
| 9 | current campaign manifest 固定 | ✅ | `20260913-current-head-5f8d91d` manifest status = `complete` | `campaign_manifest=20260913-current-head-5f8d91d; status=complete` |
| 10 | MACE/DPA/GRACE/UMA scientific-campaign GPU smoke | ✅ | 4 个后端 x t1/t2fd/t5/t6/t7/t8 | `tier_records=t1,t2fd,t5,t6,t7,t8; engines=DPA,GRACE,MACE,UMA` |
| 11 | Release-candidate bridge smoke | ✅ | 20260913-release-bridge-6e1a945f @ `6e1a945f91c76a4f17be10ed189fba0232936514`: dpa:pass, grace:pass, mace:pass, uma:pass; negatives dpa:no_implicit_fallback:pass, dpa:strict_config:pass, grace:no_implicit_fallback:pass, grace:strict_config:pass, mace:no_implicit_fallback:pass, mace:strict_config:pass, uma:no_implicit_fallback:pass, uma:strict_config:pass; fairchem alias pass; bridge archive `4fadb6db5258e4b4…` https://github.com/hydrogen1222/mliport/releases/download/v2.0.0b3/mliport-validation-release-bridge-6e1a945f.tar.zst | `release_bridge=20260913-release-bridge-6e1a945f; release_commit=6e1a945f91c76a4f17be10ed189fba0232936514; engines=DPA,GRACE,MACE,UMA; negatives=no_backend,strict_config; alias=fairchem` |
| 12 | repeat inference target-commit | ✅ | T1 记录包含各后端的 repeat inference 指标 | `tier=t1; metric=repeat_inference` |
| 13 | FD target-commit / 证据重分类 | ✅ | T2FD：32 条记录 | `tier=t2fd; records=32` |
| 14 | saddle 当前语义 | ⚠️ | CPU/解析 known-answer 层全绿；GPU `t5h` 未纳入四后端 smoke | `workflow=t5h; layer=cpu_known_answer` |
| 15 | NEB fixed-band validation | ✅ | 全 image 约束校验 + 回归测试 | `tier=t5; checks=fixed_band,constraint` |
| 16 | kinisi skew / exact-unwrap known-answer | ✅ | PR-E contract 测试；T7 transport 使用 `exact_unwrapped`（12 条） | `tests=PR-E; tier=t7; contract=exact_unwrapped` |
| 17 | GEMDAT backend error 不再变物理 0 | ✅ | PR-C 状态合同；T7 12 条记录 | `tier=t7; status_contract=gemdat_backend_error` |
| 18 | alpha weighting/validity 收口 | ✅ | local Δlog(t) 权重 + finite & valid summary | `analysis=alpha; version=2` |
| 19 | analysis version bump | ✅ | alpha estimator `/2`；task revision msd 7 / transport 6 | `analysis=msd,transport; revisions=msd:7,transport:6` |
| 20 | T7 transport 重算 | ✅ | 12 条 T7 记录（md/transport/gemdat） | `tier=t7; records=12` |
| 21 | T8 按新身份重跑/重建 | ✅ | 25 条 T8 记录 | `tier=t8; records=25` |
| 22 | historical reused evidence 明确标识 | ✅ | version block + campaign scope 的 `not_in_scope` / `historical_reuse` | `tiers=historical_reuse:t2,t3,t4; scope=explicit` |
| 23 | report 可从正式 evidence archive 重建 | ✅ | sha256 `e6a95777f471461f…`，101 条记录，https://github.com/hydrogen1222/mliport/releases/download/v2.0.0b2/mliport-validation-20260913-current-head-5f8d91d.tar.zst | `archive_manifest; sha256=e6a95777f471461f; url=https://github.com/hydrogen1222/mliport/releases/download/v2.0.0b2/mliport-validation-20260913-current-head-5f8d91d.tar.zst` |
| 24 | README 声明与证据一致 | ✅ | 生成块 + 明确的历史 T3/T4 声明 | `summary=beta-summary.json; readme_block=README_VALIDATION.md` |
| 25 | 无 secret/model weights/临时 probe/attic 污染发行包 | ✅ | hygiene/distribution 检查；archive 排除权重、轨迹与 attic | `hygiene=repository_hygiene,distribution_manifest,rename_guard` |

范围说明:tier `t2, t3, t4` 不在当前 campaign 内运行或不产出当前记录,其历史记录只列在 historical 小节,不作为当前提交声明;GPU `t5h` saddle 工作流不在四后端 smoke 范围内,其语义由 CPU/解析 known-answer 层覆盖。

## Release bridge（精确 release candidate）

完整科学 campaign 的目标提交是 `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`；release candidate `6e1a945f91c76a4f17be10ed189fba0232936514` 含产品化/配置层改动，因此在该提交上执行了四后端 release bridge（campaign `20260913-release-bridge-6e1a945f`）。bridge 只验证 backend 选择、模型加载、单点推断、strict config 与 no-backend fail-closed 路径，没有重新生成任何长时间科学轨迹。

| 后端 | model load | SP | no-backend | strict config |
|---|---|---|---|---|
| MACE | pass | pass | pass | pass |
| DPA | pass | pass | pass | pass |
| GRACE | pass | pass | pass | pass |
| UMA | pass | pass | pass | pass |

负例检查：dpa:no_implicit_fallback=pass, dpa:strict_config=pass, grace:no_implicit_fallback=pass, grace:strict_config=pass, mace:no_implicit_fallback=pass, mace:strict_config=pass, uma:no_implicit_fallback=pass, uma:strict_config=pass；fairchem alias：pass。
Bridge archive：`4fadb6db5258e4b454d49b2fa804a4fb14cbb45eb3ec8dcf5b4e196da9065169`（4 条）https://github.com/hydrogen1222/mliport/releases/download/v2.0.0b3/mliport-validation-release-bridge-6e1a945f.tar.zst

## T1: inference (energy / forces / stress)

| engine | model identity | dtype | records | by status |
|---|---|---|---|---|
| mace | MACE-OMAT-0 medium, float64 inference | float64 | 3 | pass 3 |
| dpa | DPA-3.1-3M, branch Omat24 | upstream/model-defined | 3 | pass 3 |
| grace | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | 3 | pass 3 |
| uma | UMA-s 1.2 (OMat task head) | upstream/model-defined | 3 | pass 3 |

## T2: invariance, repeatability, A→B→A

Not run in this campaign.

Historical evidence exists for this tier; it is listed in the historical section and is not part of this campaign's claim.

## T2fd: force/stress vs finite-difference derivatives

| profile | records | by status |
|---|---|---|
| dpa-pr6 | 8 | characterized 8 |
| grace-pr6 | 8 | characterized 8 |
| mace-float64-pr6 | 8 | characterized 8 |
| uma-pr6 | 8 | characterized 8 |

## T3: OMat24 held-out evaluation

Not run in this campaign.

Historical evidence exists for this tier; it is listed in the historical section and is not part of this campaign's claim.

## T4: static workflows

Not run in this campaign.

Historical evidence exists for this tier; it is listed in the historical section and is not part of this campaign's claim.

## T5: NEB and saddle validation

### NEB endpoints (Cu vacancy, relaxed fmax)

mace/float64#c1f5e8: pass, converged True, symmetry dE 0 eV<br>dpa/upstream/model-defined/Omat24#b8c075: pass, converged True, symmetry dE 6.557e-07 eV<br>grace/upstream/model-defined#efb19e: pass, converged True, symmetry dE 9.537e-07 eV<br>uma/upstream/model-defined#1d1a75: pass, converged True, symmetry dE 7.199e-07 eV

### Path initialisation (linear vs IDPP)

Linear band peak-vs-initial dE 1.335 eV; IDPP 1.335 eV; max segment 0.6248 A (linear) / 0.6248 A (IDPP); atom mapping: identity permutation; single unwrap-free hop verified in build_vacancy_pair.

### CI-NEB barriers (sampled from recorded band energies)

| path | mace | dpa | grace | uma |
|---|---|---|---|---|
| Cu vacancy NEB-5 (CI) | fwd 0.735 eV (converged_path_estimate, fmax 0.0292) | fwd 0.762 eV (converged_path_estimate, fmax 0.0292) | fwd 0.746 eV (converged_path_estimate, fmax 0.0295) | fwd 0.741 eV (converged_path_estimate, fmax 0.0291) |
| Cu vacancy NEB-7 (CI) | — | — | — | — |
| Na3PS4 Na-hop NEB-5 (CI) | — | — | — | — |

## T6: molecular dynamics (Cu32)

### NVE timestep sweep (|drift| eV/atom/ps)

| dt (fs) | mace | dpa | grace | uma | all finite |
|---|---|---|---|---|---|
| 0.25 | 1.56e-08 | 2.17e-09 | 4.02e-08 | 2.5e-08 | True |
| 0.5 | 5.03e-08 | 2.86e-08 | 7.74e-08 | 6.03e-08 | True |
| 1 | 9.82e-07 | 8.69e-07 | 8.04e-07 | 9.63e-07 | True |
| 2 | 6.5e-06 | 7.41e-06 | 6.66e-06 | 5.46e-06 | True |

Drift improves with smaller timestep: mace/float64#e04aba True, dpa/upstream/model-defined/Omat24#71d42d True, grace/upstream/model-defined#7ca4f2 True, uma/upstream/model-defined#d4e03a True.

### NVT thermostats (300 K target)

| case | mace | dpa | grace | uma |
|---|---|---|---|---|
| Langevin | mean 298 K, std 42 K (pass) | mean 298 K, std 42.8 K (characterized) | mean 298 K, std 42.2 K (characterized) | mean 299 K, std 42.2 K (characterized) |
| Bussi (CSVR) | — | — | — | — |
| NHC | — | — | — | — |

## T7：输运与机制分析（alpha-Na3PS4）

### Tracer diffusion (kinisi; D in m^2/s)

| engine | T (K) | D | kinisi 95% credible interval | sigma_NE (S/m) | fit window (ps) | native MSD diag |
|---|---|---|---|---|---|---|
| mace | 700 | 1.95e-08 | [1.92e-08, 1.98e-08] | 915 | fit from 2 ps (MSD 312 A^2) | 5.2e-08 |
| dpa | 700 | 1.6e-08 | [1.56e-08, 1.63e-08] | 751 | fit from 2 ps (MSD 322 A^2) | 5.34e-08 |
| grace | 700 | 1.57e-08 | [1.52e-08, 1.62e-08] | 735 | fit from 2 ps (MSD 318 A^2) | 5.3e-08 |
| uma | 700 | 1.8e-08 | [1.77e-08, 1.83e-08] | 846 | fit from 2 ps (MSD 324 A^2) | 5.35e-08 |

### Estimator comparison (same trajectory)

| engine | T (K) | D_ols native window | D_ols kinisi window | native MSD diagnostic D | kinisi posterior D | kinisi/plain | kinisi/same-window OLS | NE sigma (S/m) | windows native/kinisi (ps) | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| mace | 700 | 5.2e-08 | 3.93e-08 | 5.2e-08 | 1.95e-08 | 0.496 | 0.496 | 915 | 7.5-15.0 / 2.0-15.0 | estimator_disagreement_warning, estimator_model_difference |
| dpa | 700 | 5.34e-08 | 4.06e-08 | 5.34e-08 | 1.6e-08 | 0.394 | 0.394 | 751 | 7.5-15.0 / 2.0-15.0 | estimator_disagreement_warning, estimator_model_difference |
| grace | 700 | 5.3e-08 | 3.99e-08 | 5.3e-08 | 1.57e-08 | 0.392 | 0.392 | 735 | 7.5-15.0 / 2.0-15.0 | estimator_disagreement_warning, estimator_model_difference |
| uma | 700 | 5.35e-08 | 4.04e-08 | 5.35e-08 | 1.8e-08 | 0.445 | 0.445 | 846 | 7.5-15.0 / 2.0-15.0 | estimator_disagreement_warning, estimator_model_difference |

Units: ``D = slope(MSD)/(2d)`` with ``1 A^2/ps = 1e-8 m^2/s`` (known-answer audited). The two ``D_ols`` columns use the native diagnostic window and the kinisi fit window on the same trajectory, so the window effect is separated from the estimator effect.
The kinisi 95% credible interval is estimator/model uncertainty conditional on the analysed trajectory; it does not include independent initial-condition uncertainty, finite-size convergence, MLIP model error, temperature sampling convergence or long-time rare-event sampling.
``estimator_disagreement_warning`` (kinisi/plain < 0.5 or > 2) is a prompt to explain the difference. When kinisi still differs from OLS on the same window, ``estimator_model_difference`` records that the remaining gap comes from the estimator's statistical model (overlapping displacement correlations, posterior inference) rather than the fit interval; no estimator is forced to match another.

### Production MD health (700 K)

dpa: T std 50.8 K, drift -0.000485 eV/atom/ps<br>grace: T std 49.3 K, drift 0.000419 eV/atom/ps<br>mace: T std 50.1 K, drift -0.000209 eV/atom/ps<br>uma: T std 47.4 K, drift -0.000156 eV/atom/ps

### GEMDAT mechanism crosscheck

| engine | T (K) | jumps | mean/max jump dist (A) | status |
|---|---|---|---|---|
| mace | 700 | 101 | 4.71/7.78 | characterized |
| dpa | 700 | 111 | 4.69/7.78 | characterized |
| grace | 700 | 110 | 4.79/7.78 | characterized |
| uma | 700 | 112 | 4.65/7.78 | characterized |

## T8: performance (V100-16GB)

### Single-point warm latency (median [p05-p95] ms, n timed)

| atoms | mace | dpa | grace | uma |
|---|---|---|---|---|
| 32 | 48.8 [48.6-53.5] n=20 | 155 [154-157] n=20 | grace/upstream/model-defined#efb19e: 8.38 [8.32-8.45] n=20 (cache on)<br>grace/upstream/model-defined#efb19e: 8.99 [8.92-9.26] n=20 (cache off) | 124 [123-127] n=20 |
| 128 | 51.5 [50.8-59.6] n=20 | 156 [155-159] n=20 | grace/upstream/model-defined#efb19e: 13.5 [13.3-13.7] n=20 (cache on)<br>grace/upstream/model-defined#efb19e: 15.4 [15.2-16.1] n=20 (cache off) | 126 [124-127] n=20 |
| 512 | 169 [167-172] n=20 | 166 [165-169] n=20 | grace/upstream/model-defined#efb19e: 31 [30.5-32] n=20 (cache on)<br>grace/upstream/model-defined#efb19e: 41.3 [40.9-42.5] n=20 (cache off) | 222 [220-223] n=20 |

### Single-point startup and spread (per engine/size)

| engine | variant | atoms | model_load_s | first_inference_ms | warmup | n_timed | median_ms | mean_ms | p05_ms | p95_ms | min_ms | max_ms | atoms/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mace | default | 32 | 3.78 | 539 | 8 | 20 | 48.8 | 49.8 | 48.6 | 53.5 | 48.6 | 56.9 | 7e+02 |
| dpa | default | 32 | 5.82 | 5.33e+03 | 8 | 20 | 155 | 155 | 154 | 157 | 154 | 159 | 2e+02 |
| grace | cache on | 32 | 10.4 | 8.76e+03 | 5 | 20 | 8.38 | 8.39 | 8.32 | 8.45 | 8.32 | 8.55 | 4e+03 |
| grace | cache off | 32 | 10.4 | 8.88e+03 | 5 | 20 | 8.99 | 9.06 | 8.92 | 9.26 | 8.9 | 9.34 | 4e+03 |
| uma | default | 32 | 9.46 | 876 | 5 | 20 | 124 | 124 | 123 | 127 | 123 | 128 | 3e+02 |
| mace | default | 128 | 3.78 | 55.7 | 5 | 20 | 51.5 | 52.9 | 50.8 | 59.6 | 50.6 | 61.2 | 2e+03 |
| dpa | default | 128 | 5.82 | 197 | 5 | 20 | 156 | 156 | 155 | 159 | 155 | 159 | 8e+02 |
| grace | cache on | 128 | 10.4 | 9.64e+03 | 5 | 20 | 13.5 | 13.5 | 13.3 | 13.7 | 13.3 | 13.7 | 9e+03 |
| grace | cache off | 128 | 10.4 | 9.7e+03 | 5 | 20 | 15.4 | 15.5 | 15.2 | 16.1 | 15.2 | 16.2 | 8e+03 |
| uma | default | 128 | 9.46 | 132 | 5 | 20 | 126 | 126 | 124 | 127 | 124 | 131 | 1e+03 |
| mace | default | 512 | 3.78 | 176 | 6 | 20 | 169 | 169 | 167 | 172 | 167 | 172 | 3e+03 |
| dpa | default | 512 | 5.82 | 228 | 5 | 20 | 166 | 167 | 165 | 169 | 165 | 169 | 3e+03 |
| grace | cache on | 512 | 10.4 | 9.48e+03 | 5 | 20 | 31 | 31.2 | 30.5 | 32 | 30.5 | 34.8 | 2e+04 |
| grace | cache off | 512 | 10.4 | 9.43e+03 | 5 | 20 | 41.3 | 41.6 | 40.9 | 42.5 | 40.8 | 47.3 | 1e+04 |
| uma | default | 512 | 9.46 | 222 | 5 | 20 | 222 | 221 | 220 | 223 | 219 | 223 | 2e+03 |

Benchmark provenance: warmup stop `fixed_count:5` x10; warmup stop `stability:last3<=0.05` x15; GPU sync `tensorflow:implicit-host-transfer(cuda:0)` x10; GPU sync `torch.cuda.synchronize(cuda:0)` x15. Startup (`model_load_s`, `first_inference_ms`) is measured separately and excluded from the warm median. TensorFlow 2.20 has no explicit device-sync API, so GRACE reports its implicit host-transfer synchronization.

### NVE MD throughput (steps/s / atom-steps/s)

| atoms | mace | dpa | grace | uma |
|---|---|---|---|---|
| 128 | 20.9 / 2.68e+03 | 6.34 / 811 | grace/upstream/model-defined#efb19e: 76.5 / 9.79e+03<br>grace/upstream/model-defined#efb19e: 61.3 / 7.84e+03 | 7.93 / 1.01e+03 |
| 512 | 7.1 / 3.64e+03 | 5.99 / 3.07e+03 | grace/upstream/model-defined#efb19e: 32 / 1.64e+04<br>grace/upstream/model-defined#efb19e: 24.1 / 1.23e+04 | 4.52 / 2.32e+03 |

## 不在当前 campaign 内的历史证据

| tier | 记录数 | campaign | software commit | 未重跑原因 |
|---|---|---|---|---|
| t1 | 3 | untagged | `360e1d2034e7...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t1 | 12 | untagged | `7cfce9f908fe...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t2 | 120 | untagged | `05e548c41f8e...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t2fd | 41 | untagged | `05e548c41f8e...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t3 | 4 | untagged | `7b22280d5eb7...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t4 | 112 | untagged | `293c1ed36f6a...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t4 | 200 | untagged | `38abb85d87c9...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t5 | 36 | untagged | `e08ab41788ee...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t6 | 6 | untagged | `b79f0d0c7947...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t6 | 18 | untagged | `c676c7d2f1ac...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t7 | 40 | untagged | `293c1ed36f6a...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |
| t8 | 20 | untagged | `7cfce9f908fe...` | not re-run in the current campaign; kept as historical evidence and excluded from current tables |

这些记录仅作为元数据列出:不会渲染进当前 tier 表格,也不构成对当前提交的验证声明。

## Support matrix (section 39 classification)

`software_validated` and `model_characterized` coexist; `fail`/`characterized` records keep the workload classified as characterized, with the record counts visible.

| workload | mace | dpa | grace | uma |
|---|---|---|---|---|
| install | software_validated * | software_validated * | software_validated * | software_validated * |
| single_point | software_validated | software_validated | software_validated | software_validated |
| forces | model_characterized | model_characterized | model_characterized | model_characterized |
| stress | model_characterized | model_characterized | model_characterized | model_characterized |
| force_energy_consistency | model_characterized | model_characterized | model_characterized | model_characterized |
| stress_energy_consistency | model_characterized | model_characterized | model_characterized | model_characterized |
| invariance_caching | not_run | not_run | not_run | not_run |
| fixed_cell_relax | not_run | not_run | not_run | not_run |
| cell_relax | not_run | not_run | not_run | not_run |
| eos | not_run | not_run | not_run | not_run |
| elastic | not_run | not_run | not_run | not_run |
| phonon | not_run | not_run | not_run | not_run |
| thermodynamics | not_run | not_run | not_run | not_run |
| defect | not_run | not_run | not_run | not_run |
| surface | not_run | not_run | not_run | not_run |
| formation_energy | not_run | not_run | not_run | not_run |
| neb | software_validated | software_validated | software_validated | software_validated |
| saddle_hessian | not_run | not_run | not_run | not_run |
| short_nve | software_validated | software_validated | software_validated | software_validated |
| short_nvt | software_validated | model_characterized | model_characterized | model_characterized |
| transport_demo | model_characterized | model_characterized | model_characterized | model_characterized |
| mechanism_analysis | model_characterized | model_characterized | model_characterized | model_characterized |
| arrhenius | not_run | not_run | not_run | not_run |
| performance | software_validated | software_validated | software_validated | software_validated |

\* install rows cover software validation only (installer, doctor, runtime validation in CI); no model is involved.

## Known limitations

- The OMat24 validation split ships no reference stress labels, so
  stress parity against OMat24 data is not computable; stress is
  validated numerically instead (T1 consistency, T2fd derivative
  agreement).
- Upstream-float32 builds (DPA, GRACE, UMA) show 1e-7..1e-6 eV
  coordinate-order arithmetic noise on transformed-structure
  comparisons; float64 profiles pass bitwise. Recorded as-is.
- Physical transport conclusions (Arrhenius) need longer trajectories
  than the beta budget; stage-2 results are exploratory and honestly
  classified.
- No NPT, no electronic structure, no model uncertainty unless a
  backend implements and audits it.
- V100 validation does not prove every GPU architecture.
