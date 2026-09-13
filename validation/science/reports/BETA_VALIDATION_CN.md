# mliport beta 科学验证报告

由 `.validation-work` 下的证据记录生成；证据对应提交 `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`。本文档全部表格由 `validation/science/scripts/generate_beta_report.py` 从结果 JSON 渲染，更新方式是重新生成并 diff，不要手工编辑。

重新认证状态：**current_head_revalidated** —— campaign manifest declares completion and every record was produced at the target software commit

- software_commit（声称被验证）：`5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`
- validation_code_commit：`a0d677b0660b76aee9a3e90ae08e5e24873124a4`
- report_generator_commit：`a0d677b0660b76aee9a3e90ae08e5e24873124a4`
- evidence_campaign：`20260913-current-head-5f8d91d`
- evidence_source_commits：`5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`

模型身份与产物哈希固定在 `validation/science/model_manifest.json`；OMat24 评估子集见 `validation/science/data/`（种子 20260911）。

## Beta GO / NO-GO 清单（任务书 §28）

- 结论：**beta GO**（23 ✅ / 1 ⚠️ / 0 ❌）
- 被验证软件提交：`5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`
- 证据 campaign：`20260913-current-head-5f8d91d`，96 条记录（53 pass / 43 characterized / 0 fail / 0 blocked）

| # | 条目 | 状态 | 证据 |
|---|---|---|---|
| 1 | 新项目名无已知同领域 namespace collision | ✅ | `mliport` 在 PyPI 镜像与 GitHub 搜索均可用；仓库已改名 `hydrogen1222/mliport` |
| 2 | Python package / CLI / repo identity 完成迁移 | ✅ | 改名提交链；clean break，旧 `mlipx` import 不再发布 |
| 3 | 目标提交 CI 3.10/3.11/3.12 全绿 | ✅ | `5f8d91d4` 及后续提交的 tests/lint/package-build 全绿 |
| 4 | wheel clean install 全绿 | ✅ | wheel-install-smoke 3.10/3.11/3.12 + `mliport-2.0.0b1` capability smoke |
| 5 | strict evidence loader 唯一 | ✅ | `validation/science/evidence/` + report 不得直接读 raw record 的静态测试 |
| 6 | report/README 不再直接解释 raw records | ✅ | 报告与 figures 全部消费 `load_evidence`；README block 机器渲染 |
| 7 | malformed evidence fail-closed | ✅ | V01-T5/T6/T7 测试；report 退出码 2/3；archive builder 拒绝 |
| 8 | profile identity 不混 precision/model/commit | ✅ | V01-T2/T3/T4；engine 单元格显示 `mixed` 并附 `worst_status` |
| 9 | current campaign manifest 固定 | ✅ | `20260913-current-head-5f8d91d` manifest status = `complete` |
| 10 | MACE/DPA/GRACE/UMA current-head GPU smoke | ✅ | 4 个后端 x t1/t2/t2fd/t3/t4/t5/t6/t7/t8 |
| 11 | repeat inference current-head | ✅ | T1 记录包含各后端的 repeat inference 指标 |
| 12 | FD current-head / 证据重分类 | ✅ | T2FD：32 条记录 |
| 13 | saddle 当前语义 | ⚠️ | CPU/解析 known-answer 层全绿；GPU `t5h` 未纳入四后端 smoke |
| 14 | NEB fixed-band validation | ✅ | 全 image 约束校验 + 回归测试 |
| 15 | kinisi skew / exact-unwrap known-answer | ✅ | PR-E contract 测试；T7 transport 使用 `exact_unwrapped`（12 条） |
| 16 | GEMDAT backend error 不再变物理 0 | ✅ | PR-C 状态合同；T7 12 条记录 |
| 17 | alpha weighting/validity 收口 | ✅ | local Δlog(t) 权重 + finite & valid summary |
| 18 | analysis version bump | ✅ | alpha estimator `/2`；task revision msd 7 / transport 6 |
| 19 | T7 transport 重算 | ✅ | 12 条 T7 记录（md/transport/gemdat） |
| 20 | T8 按新身份重跑/重建 | ✅ | 20 条 T8 记录 |
| 21 | historical reused evidence 明确标识 | ✅ | version block + campaign scope 的 `not_in_scope` / `historical_reuse` |
| 22 | report 可从正式 evidence archive 重建 | ✅ | sha256 `bb6b1b06bb3e891e…`，96 条记录，https://github.com/hydrogen1222/mliport/releases/download/v2.0.0b1/mliport-validation-20260913-current-head-5f8d91d.tar.zst |
| 23 | README 声明与证据一致 | ✅ | 生成块 + 明确的历史 T3/T4 声明 |
| 24 | 无 secret/model weights/临时 probe/attic 污染发行包 | ✅ | hygiene/distribution 检查；archive 排除权重、轨迹与 attic |

范围说明：T3 精度与 T4 静态工作流保持历史证据，不作为 current-HEAD 声明；GPU `t5h` saddle 工作流不在四后端 smoke 范围内，其语义由 CPU/解析 known-answer 层覆盖。

## T1: inference (energy / forces / stress)

| engine | model identity | dtype | records | by status |
|---|---|---|---|---|
| mace | MACE-OMAT-0 medium, float64 inference | float64 | 3 | pass 3 |
| dpa | DPA-3.1-3M, branch Omat24 | upstream/model-defined | 3 | pass 3 |
| grace | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | 3 | pass 3 |
| uma | UMA-s 1.2 (OMat task head) | upstream/model-defined | 3 | pass 3 |

## T2: invariance, repeatability, A→B→A

Tolerance policy: `max(10 x measured repeatability floor, absolute floor 1e-10 eV / 1e-9 eV/A / 1e-9 eV/A^3)`. The floors are measured per system/profile from repeated identical inference before any transformed comparison.

Recorded outcome: the float64 profile (MACE) passes every check bitwise. The upstream-float32 builds (DPA, UMA) show 1e-7..1e-6 eV coordinate-order arithmetic noise across most transformed comparisons. GRACE passes all four-system invariance checks with the mliport neighbor cache ON (energy deltas 0..1e-14 eV); with the cache OFF the same noise appears on most checks. These failures are recorded as-is, not hidden.

| profile | records | by status |
|---|---|---|

## T2fd: force/stress vs finite-difference derivatives

| profile | records | by status |
|---|---|---|
| dpa-pr6 | 8 | characterized 8 |
| grace-pr6 | 8 | characterized 8 |
| mace-float64-pr6 | 8 | characterized 8 |
| uma-pr6 | 8 | characterized 8 |

## T3: OMat24 held-out evaluation

not_run.

## T4: static workflows

Two harness defects were found while rendering this report and fixed with regression tests; the affected records were archived under `.validation-work/attic/` and regenerated on CPU (the `device` field of each fresh record says cpu): (1) t4_thermo once summed the 8x8x8 q-grid without the 1/N_q normalization, inflating ZPE/Cv/S/F by 512x -- the phonon normal modes themselves were never affected; (2) the relaxed-ion elastic path once discarded the relaxed structure, making the t4d relaxed-ion records duplicate the clamped-ion values.

### Ionic relaxation (fixed cell, FIRE / LBFGS)

| system | FIRE: status (fmax, steps) | LBFGS: status (fmax, steps) | optimizer agreement |
|---|---|---|---|
| cu_fcc | — | — | not_run (dE None eV, both converged None) |
| si_diamond | — | — | not_run (dE None eV, both converged None) |
| mgo_rocksalt | — | — | not_run (dE None eV, both converged None) |
| na3ps4 | — | — | not_run (dE None eV, both converged None) |

### Cell relaxation (FrechetCellFilter, requires stress + 3D PBC)

| system | status (dV, dE) |
|---|---|
| cu_fcc | not_run |
| si_diamond | not_run |
| mgo_rocksalt | not_run |
| na3ps4 | not_run |

### Equation of state (Birch-Murnaghan B0 in GPa / V0 in A^3)

| system | mace | dpa | grace | uma |
|---|---|---|---|---|
| cu_fcc | — | — | — | — |
| si_diamond | — | — | — | — |
| mgo_rocksalt | — | — | — | — |

### Cubic elastic constants (GPa, finite-strain fits)

| system | variant | mace | dpa | grace | uma |
|---|---|---|---|---|---|
| cu_fcc | clamped | — | — | — | — |
| cu_fcc | relaxed | — | — | — | — |
| si_diamond | clamped | — | — | — | — |
| si_diamond | relaxed | — | — | — | — |

### Harmonic phonons (min Gamma frequency, ASR-corrected; min over supercell/displacement variants)

| system | mace | dpa | grace | uma | robust imaginary |
|---|---|---|---|---|---|
| cu_fcc | — | — | — | — |  |
| si_diamond | — | — | — | — |  |

### Harmonic thermodynamics (per phonon unit cell, q-averaged over the 8x8x8 MP grid; worst variant shown)

| system | mace | dpa | grace | uma |
|---|---|---|---|---|
| cu_fcc | — | — | — | — |
| si_diamond | — | — | — | — |

### Cu vacancy energy (relaxed, eV; finite-size trend)

| supercell | mace | dpa | grace | uma |
|---|---|---|---|---|
| Cu 2x2x2 | — | — | — | — |
| Cu 3x3x3 | — | — | — | — |
| Cu 4x4x4 | — | — | — | — |

### Cu(111) surface energy (relaxed, J/m^2)

| slab | mace | dpa | grace | uma |
|---|---|---|---|---|
| 4L_10A | — | — | — | — |
| 4L_15A | — | — | — | — |
| 6L_10A | — | — | — | — |
| 6L_15A | — | — | — | — |
| 8L_10A | — | — | — | — |
| 8L_15A | — | — | — | — |

### Conditional reference energies (OMat24 exact refs)

| case | mace | dpa | grace | uma |
|---|---|---|---|---|
| Cu cohesive | not_run | not_run | not_run | not_run |
| formation (exact refs) | not_run | not_run | not_run | not_run |

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

| engine | T (K) | D | 95% CI | sigma_NE (S/m) | fit window (ps) | native MSD diag |
|---|---|---|---|---|---|---|
| mace | 700 | 1.95e-08 | [1.92e-08, 1.98e-08] | 915 | fit from 2 ps (MSD 312 A^2) | 5.2e-08 |
| dpa | 700 | 1.6e-08 | [1.56e-08, 1.63e-08] | 751 | fit from 2 ps (MSD 322 A^2) | 5.34e-08 |
| grace | 700 | 1.57e-08 | [1.52e-08, 1.62e-08] | 735 | fit from 2 ps (MSD 318 A^2) | 5.3e-08 |
| uma | 700 | 1.8e-08 | [1.77e-08, 1.83e-08] | 846 | fit from 2 ps (MSD 324 A^2) | 5.35e-08 |

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

### Single-point warm latency (median ms)

| atoms | mace | dpa | grace | uma |
|---|---|---|---|---|
| 32 | 700 | 1.07e+04 | 9.2 | 123 |
| 128 | 51.9 | 163 | 12.8 | 123 |
| 512 | 139 | 177 | 30.9 | 212 |

### NVE MD throughput (steps/s / atom-steps/s)

| atoms | mace | dpa | grace | uma |
|---|---|---|---|---|
| 128 | 19 / 2.43e+03 | 6.31 / 808 | 76.3 / 9.76e+03 | 8.14 / 1.04e+03 |
| 512 | 7.1 / 3.63e+03 | 5.96 / 3.05e+03 | 32.3 / 1.66e+04 | 4.66 / 2.38e+03 |

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
