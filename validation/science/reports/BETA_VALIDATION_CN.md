# mlipx beta 科学验证报告

由 `.validation-work` 下的证据记录生成；证据对应提交 `05e548c41f8e863f0121702d2c8cc3c60ca8add0, 293c1ed36f6aea843929e4d3ca2fe1da9c5e88bc, 38abb85d87c92799b9b370758c9f920b4e982cbc, 7b22280d5eb723a0a9edde0ebf8d925bf410a341, 7cfce9f908feedcc64e790d6eb893fe3b02b6e5d, b79f0d0c7947ccada2cd60b384a2701060e7d2f2, c676c7d2f1ac43b1ffc12e14f1efb1024396ca2f, e08ab41788ee17aa2dc924271230ae21ad01c4e8`。本文档全部表格由 `validation/science/scripts/generate_beta_report.py` 从结果 JSON 渲染，更新方式是重新生成并 diff，不要手工编辑。

模型身份与产物哈希固定在 `validation/science/model_manifest.json`；OMat24 评估子集见 `validation/science/data/`（种子 20260911）。

## T1: inference (energy / forces / stress)

| engine | model identity | dtype | records | by status |
|---|---|---|---|---|
| mace | MACE-OMAT-0 medium, float64 inference | float64 | 3 | pass 3 |
| dpa | DPA-3.1-3M, branch Omat24 | upstream/model-defined | 3 | pass 3 |
| grace | GRACE-2L-OMAT-medium-base (upstream default precision build) | upstream/model-defined | 3 | pass 3 |
| uma | UMA-s 1.2 (OMat task head) | upstream/model-defined | 3 | pass 3 |

## T2: invariance, repeatability, A→B→A

Tolerance policy: `max(10 x measured repeatability floor, absolute floor 1e-10 eV / 1e-9 eV/A / 1e-9 eV/A^3)`. The floors are measured per system/profile from repeated identical inference before any transformed comparison.

Recorded outcome: the float64 profile (MACE) passes every check bitwise. The upstream-float32 builds (DPA, UMA) show 1e-7..1e-6 eV coordinate-order arithmetic noise across most transformed comparisons. GRACE passes all four-system invariance checks with the mlipx neighbor cache ON (energy deltas 0..1e-14 eV); with the cache OFF the same noise appears on most checks. These failures are recorded as-is, not hidden.

| profile | records | by status |
|---|---|---|
| dpa | 24 | characterized 4, fail 18, pass 2 |
| grace | 24 | characterized 4, pass 20 |
| grace-neighoff | 24 | characterized 4, fail 18, pass 2 |
| mace-float64 | 24 | characterized 4, pass 20 |
| uma | 24 | characterized 4, fail 18, pass 2 |

## T2fd: force/stress vs finite-difference derivatives

| profile | records | by status |
|---|---|---|
| dpa | 8 | characterized 8 |
| grace | 9 | characterized 8, fail 1 |
| grace-neighoff | 8 | characterized 8 |
| mace-float64 | 8 | characterized 8 |
| uma | 8 | characterized 8 |

## T3: OMat24 held-out evaluation (256-structure subset)

No elemental offsets are fitted; model energies are compared to the OMat24 reference energies directly. The official OMat24 validation split carries no reference stress labels, so stress parity is not computable and is recorded as absent.

| engine | E/atom MAE (eV) | E/atom RMSE | median | p95 | F comp MAE (eV/A) | F cosine min | records |
|---|---|---|---|---|---|---|---|
| dpa | 0.01839 | 0.04354 | 0.009547 | 0.05439 | 0.09062 | -0.8992 | 256 |
(dpa: reference stress absent in the official validation split)

| grace | 0.01433 | 0.04869 | 0.005908 | 0.03586 | 0.07113 | -0.8893 | 256 |
(grace: reference stress absent in the official validation split)

| mace | 0.01725 | 0.0491 | 0.007634 | 0.03875 | 0.08063 | -0.9041 | 256 |
(mace: reference stress absent in the official validation split)

| uma | 0.01083 | 0.04053 | 0.00323 | 0.02664 | 0.05276 | -0.9119 | 256 |
(uma: reference stress absent in the official validation split)


Per-structure errors: `t3/t3_errors_dpa.csv`, `t3/t3_errors_grace.csv`, `t3/t3_errors_mace.csv`, `t3/t3_errors_uma.csv`.

## T4: static workflows

Two harness defects were found while rendering this report and fixed with regression tests; the affected records were archived under `.validation-work/attic/` and regenerated on CPU (the `device` field of each fresh record says cpu): (1) t4_thermo once summed the 8x8x8 q-grid without the 1/N_q normalization, inflating ZPE/Cv/S/F by 512x -- the phonon normal modes themselves were never affected; (2) the relaxed-ion elastic path once discarded the relaxed structure, making the t4d relaxed-ion records duplicate the clamped-ion values.

### Ionic relaxation (fixed cell, FIRE / LBFGS)

| system | FIRE: status (fmax, steps) | LBFGS: status (fmax, steps) | optimizer agreement |
|---|---|---|---|
| cu_fcc | pass (fmax 0.009756, 18 steps) | pass (fmax 0.001218, 6 steps) | pass (dE 1.54e-05 eV, both converged True) |
| si_diamond | pass (fmax 0.009808, 25 steps) | pass (fmax 0.004673, 11 steps) | pass (dE 2.44e-05 eV, both converged True) |
| mgo_rocksalt | pass (fmax 0.009676, 31 steps) | pass (fmax 0.004192, 9 steps) | pass (dE 6.56e-06 eV, both converged True) |
| na3ps4 | pass (fmax 0.008422, 55 steps) | pass (fmax 0.009593, 30 steps) | pass (dE 1.1e-05 eV, both converged True) |

### Cell relaxation (FrechetCellFilter, requires stress + 3D PBC)

| system | status (dV, dE) |
|---|---|
| cu_fcc | dpa: pass (V 51.62 -> 47.11 A^3, dV -8.74%, fmax 5.95e-08)<br>grace: pass (V 51.62 -> 47.01 A^3, dV -8.93%, fmax 1.29e-07)<br>mace: pass (V 51.62 -> 47.96 A^3, dV -7.1%, fmax 2.81e-15)<br>uma: pass (V 51.62 -> 47.28 A^3, dV -8.42%, fmax 5.37e-08) |
| si_diamond | dpa: pass (V 174.95 -> 163.54 A^3, dV -6.52%, fmax 6.91e-06)<br>grace: pass (V 174.95 -> 162.49 A^3, dV -7.12%, fmax 9.03e-07)<br>mace: pass (V 174.95 -> 159.64 A^3, dV -8.75%, fmax 6.8e-14)<br>uma: pass (V 174.95 -> 162.9 A^3, dV -6.89%, fmax 1.96e-05) |
| mgo_rocksalt | dpa: pass (V 81.65 -> 77.05 A^3, dV -5.64%, fmax 3.71e-07)<br>grace: pass (V 81.65 -> 77.23 A^3, dV -5.42%, fmax 2.68e-07)<br>mace: pass (V 81.65 -> 76.91 A^3, dV -5.81%, fmax 5.84e-15)<br>uma: pass (V 81.65 -> 77.33 A^3, dV -5.29%, fmax 1.7e-07) |
| na3ps4 | dpa: pass (V 370.72 -> 356.52 A^3, dV -3.83%, fmax 0.00549)<br>grace: pass (V 370.72 -> 347.24 A^3, dV -6.33%, fmax 0.0102)<br>mace: pass (V 370.72 -> 350.21 A^3, dV -5.53%, fmax 0.00676)<br>uma: pass (V 370.72 -> 352.65 A^3, dV -4.88%, fmax 0.00855) |

### Equation of state (Birch-Murnaghan B0 in GPa / V0 in A^3)

| system | mace | dpa | grace | uma |
|---|---|---|---|---|
| cu_fcc | 137.1 / 47.96 | 143.4 / 47.1 | 130.4 / 47.03 | 144.6 / 47.26 |
| si_diamond | 92.05 / 159.5 | 78.45 / 163.5 | 86.68 / 162.5 | 88.16 / 162.8 |
| mgo_rocksalt | 153.6 / 76.87 | 151.1 / 77.06 | 147.6 / 77.22 | 152.9 / 77.3 |

### Cubic elastic constants (GPa, finite-strain fits)

| system | variant | mace | dpa | grace | uma |
|---|---|---|---|---|---|
| cu_fcc | clamped | 181.4/130.9/79.85 | 175.6/124/82.49 | 167/108.2/79.9 | 181.9/127.1/76.84 | born: stable |
| cu_fcc | relaxed | 181.4/130.9/79.85 | 175.6/124/82.49 | 167/108.2/79.9 | 181.9/127.1/76.84 | born: stable |
| si_diamond | clamped | 131.5/65.5/99.74 | 131.7/61.21/84.83 | 134.7/68.22/96.64 | 155.3/65.26/102.5 | born: stable |
| si_diamond | relaxed | 131.5/65.5/64.48 | 131.7/61.21/50.38 | 134.7/68.22/62.75 | 155.3/65.26/72.74 | born: stable |

### Harmonic phonons (min Gamma frequency, ASR-corrected; min over supercell/displacement variants)

| system | mace | dpa | grace | uma | robust imaginary |
|---|---|---|---|---|---|
| cu_fcc | -3.16e-07 meV | -3.89e-07 meV | -4.42e-07 meV | -3.27e-07 meV | mace: no<br>dpa: no<br>grace: no<br>uma: no |
| si_diamond | -1.46e-05 meV | -8e-05 meV | -0.0448 meV | -0.000515 meV | mace: no<br>dpa: no<br>grace: no<br>uma: no |

### Harmonic thermodynamics (per phonon unit cell, q-averaged over the 8x8x8 MP grid; worst variant shown)

| system | mace | dpa | grace | uma |
|---|---|---|---|---|
| cu_fcc | ZPE 0.0304 eV/cell<br>Cv(300K) 0.000245 eV/K/cell | ZPE 0.0307 eV/cell<br>Cv(300K) 0.000245 eV/K/cell | ZPE 0.0307 eV/cell<br>Cv(300K) 0.000245 eV/K/cell | ZPE 0.0305 eV/cell<br>Cv(300K) 0.000245 eV/K/cell |
| si_diamond | ZPE 0.123 eV/cell<br>Cv(300K) 0.000412 eV/K/cell | ZPE 0.114 eV/cell<br>Cv(300K) 0.000422 eV/K/cell | ZPE 0.123 eV/cell<br>Cv(300K) 0.00041 eV/K/cell | ZPE 0.123 eV/cell<br>Cv(300K) 0.000411 eV/K/cell |

### Cu vacancy energy (relaxed, eV; finite-size trend)

| supercell | mace | dpa | grace | uma |
|---|---|---|---|---|
| Cu 2x2x2 | 1.033 | 1.105 | 1.172 | 1.113 |
| Cu 3x3x3 | 1.026 | 1.094 | 1.179 | 1.101 |
| Cu 4x4x4 | 1.026 | 1.095 | 1.18 | 1.098 |

### Cu(111) surface energy (relaxed, J/m^2)

| slab | mace | dpa | grace | uma |
|---|---|---|---|---|
| 4L_10A | 1.236 | 1.383 | 1.347 | 1.354 |
| 4L_15A | 1.236 | 1.383 | 1.347 | 1.354 |
| 6L_10A | 1.234 | 1.379 | 1.35 | 1.362 |
| 6L_15A | 1.234 | 1.379 | 1.35 | 1.362 |
| 8L_10A | 1.232 | 1.392 | 1.35 | 1.371 |
| 8L_15A | 1.232 | 1.392 | 1.35 | 1.371 |

### Conditional reference energies (OMat24 exact refs)

| case | mace | dpa | grace | uma |
|---|---|---|---|---|
| Cu cohesive | unsupported (unsupported_for_common_comparison) | unsupported (unsupported_for_common_comparison) | unsupported (unsupported_for_common_comparison) | unsupported (unsupported_for_common_comparison) |
| formation (exact refs) | unsupported (unsupported_for_common_comparison) | unsupported (unsupported_for_common_comparison) | unsupported (unsupported_for_common_comparison) | unsupported (unsupported_for_common_comparison) |

## T5: NEB and saddle validation

### NEB endpoints (Cu vacancy, relaxed fmax)

dpa: converged True, symmetry dE 5.96e-08 eV<br>grace: converged True, symmetry dE 7.153e-07 eV<br>mace: converged True, symmetry dE 2.842e-14 eV<br>uma: converged True, symmetry dE 1.2e-06 eV

### Path initialisation (linear vs IDPP)

Linear band peak-vs-initial dE 1.335 eV; IDPP 1.335 eV; max segment 0.6248 A (linear) / 0.6248 A (IDPP); atom mapping: identity permutation; single unwrap-free hop verified in build_vacancy_pair.

### CI-NEB barriers (sampled from recorded band energies)

| path | mace | dpa | grace | uma |
|---|---|---|---|---|
| Cu vacancy NEB-5 (CI) | fwd 0.735 eV (converged_path_estimate, fmax 0.0292) | fwd 0.762 eV (converged_path_estimate, fmax 0.0292) | fwd 0.746 eV (converged_path_estimate, fmax 0.0295) | fwd 0.741 eV (converged_path_estimate, fmax 0.0291) |
| Cu vacancy NEB-7 (CI) | fwd 0.735 eV (converged_path_estimate, fmax 0.0242) | fwd 0.762 eV (converged_path_estimate, fmax 0.0288) | fwd 0.746 eV (converged_path_estimate, fmax 0.025) | fwd 0.742 eV (converged_path_estimate, fmax 0.0299) |
| Na3PS4 Na-hop NEB-5 (CI) | fwd 0.0466 eV (converged_path_estimate, fmax 0.0262) | fwd 0.0826 eV (converged_path_estimate, fmax 0.0287) | fwd 0.0581 eV (converged_path_estimate, fmax 0.0273) | fwd 0.0955 eV (converged_path_estimate, fmax 0.0297) |

### Image convergence

dpa: pass (None/None images)<br>grace: pass (None/None images)<br>mace: pass (None/None images)<br>uma: pass (None/None images)

### Checkpoint resume identity

endpoint identity True, atom map True, run id True, options fingerprint True; status pass.

### Saddle Hessian validation (Cu vacancy candidate)

| engine | verdict | persistent negative deltas | eigenvalue spread ok |
|---|---|---|---|
| dpa | pass | 3 | True |
| grace | pass | 3 | True |
| mace | pass | 3 | True |
| uma | pass | 3 | True |

## T6: molecular dynamics (Cu32)

### NVE timestep sweep (|drift| eV/atom/ps)

| dt (fs) | mace | dpa | grace | uma | all finite |
|---|---|---|---|---|---|
| 0.25 | 1.56e-08 | 5.83e-09 | 4.6e-08 | 2.67e-08 | True |
| 0.5 | 5.03e-08 | 2.92e-08 | 9.3e-08 | 5.1e-08 | True |
| 1 | 9.82e-07 | 8.54e-07 | 8.17e-07 | 9.54e-07 | True |
| 2 | 6.5e-06 | 7.43e-06 | 6.61e-06 | 5.49e-06 | True |

Drift improves with smaller timestep: dpa True, grace True, mace True, uma True.

### NVT thermostats (300 K target)

| case | mace | dpa | grace | uma |
|---|---|---|---|---|
| Langevin | mean 298 K, std 42 K (pass) | mean 298 K, std 42.8 K (characterized) | mean 298 K, std 42.2 K (characterized) | mean 299 K, std 42.2 K (characterized) |
| Bussi (CSVR) | mean 302 K, std 47.4 K (pass) | mean 308 K, std 53.2 K (pass) | mean 308 K, std 50.8 K (pass) | mean 301 K, std 50.4 K (pass) |
| NHC | mean 295 K, std 52.9 K (pass) | mean 305 K, std 56.2 K (pass) | mean 304 K, std 55.6 K (pass) | mean 299 K, std 55.1 K (pass) |

### Constraint exactness (fixed atoms + COM)

fixatoms DOF 72/72 (max deviation 0 A), COM DOF 93/93 (max drift 5.329e-15 A); status pass.

### Force-safety abort

abort raised True at step 0 (raw force 2.38 eV/A vs threshold 1.19); checkpoint True, manifest aborted; status pass.

## T7：输运与机制分析（alpha-Na3PS4）

### Tracer diffusion (kinisi; D in m^2/s)

| engine | T (K) | D | 95% CI | sigma_NE (S/m) | fit window (ps) | native MSD diag |
|---|---|---|---|---|---|---|
| mace | 600 | 1.09e-08 | [1.07e-08, 1.11e-08] | 598 | fit from 2 ps (MSD 267 A^2) | 4.42e-08 |
| mace | 700 | 1.98e-08 | [1.95e-08, 2.01e-08] | 930 | fit from 2 ps (MSD 311 A^2) | 5.23e-08 |
| mace | 800 | 1.65e-08 | [1.61e-08, 1.68e-08] | 678 | fit from 2 ps (MSD 361 A^2) | 6.07e-08 |
| dpa | 600 | 1.5e-08 | [1.47e-08, 1.53e-08] | 822 | fit from 2 ps (MSD 279 A^2) | 4.62e-08 |
| dpa | 700 | 2.01e-08 | [1.97e-08, 2.04e-08] | 943 | fit from 2 ps (MSD 323 A^2) | 5.36e-08 |
| dpa | 800 | 1.92e-08 | [1.88e-08, 1.96e-08] | 790 | fit from 2 ps (MSD 361 A^2) | 5.99e-08 |
| grace | 600 | 2.06e-08 | [2.04e-08, 2.09e-08] | 1.13e+03 | fit from 2 ps (MSD 274 A^2) | 4.57e-08 |
| grace | 700 | 1.81e-08 | [1.79e-08, 1.84e-08] | 852 | fit from 2 ps (MSD 313 A^2) | 5.22e-08 |
| grace | 800 | 2.82e-08 | [2.79e-08, 2.86e-08] | 1.16e+03 | fit from 2 ps (MSD 367 A^2) | 6.08e-08 |
| uma | 600 | 1.6e-08 | [1.57e-08, 1.63e-08] | 877 | fit from 2 ps (MSD 274 A^2) | 4.56e-08 |
| uma | 700 | 2.04e-08 | [2.01e-08, 2.08e-08] | 960 | fit from 2 ps (MSD 329 A^2) | 5.42e-08 |
| uma | 800 | 2.12e-08 | [2.08e-08, 2.16e-08] | 872 | fit from 2 ps (MSD 366 A^2) | 6.05e-08 |

### Production MD health (700 K)

dpa: T std 50.1 K, drift -0.000269 eV/atom/ps<br>grace: T std 48.7 K, drift -0.000193 eV/atom/ps<br>mace: T std 50.1 K, drift -1.34e-06 eV/atom/ps<br>uma: T std 49 K, drift 0.000142 eV/atom/ps

### Arrhenius fit (stage 2, exploratory)

| engine | temperatures (K) | D (m^2/s) | Ea (eV) | r^2 | status |
|---|---|---|---|---|---|
| dpa |  | 1.5e-08; 2.01e-08; 1.92e-08 | 0.0558 | 0.699 | characterized |
| grace |  | 2.06e-08; 1.81e-08; 2.82e-08 | 0.0611 | 0.382 | characterized |
| mace |  | 1.09e-08; 1.98e-08; 1.65e-08 | 0.103 | 0.503 | characterized |
| uma |  | 1.6e-08; 2.04e-08; 2.12e-08 | 0.0604 | 0.901 | characterized |

### GEMDAT mechanism crosscheck

| engine | T (K) | jumps | mean/max jump dist (A) | status |
|---|---|---|---|---|
| mace | 600 | 97 | 4.93/7.78 | characterized |
| mace | 700 | 107 | 4.65/7.78 | characterized |
| mace | 800 | 136 | 4.67/7.78 | characterized |
| dpa | 600 | 105 | 4.81/7.78 | characterized |
| dpa | 700 | 113 | 4.62/7.78 | characterized |
| dpa | 800 | 130 | 4.74/7.78 | characterized |
| grace | 600 | 94 | 4.64/7.78 | characterized |
| grace | 700 | 101 | 4.83/7.78 | characterized |
| grace | 800 | 120 | 4.6/7.78 | characterized |
| uma | 600 | 96 | 4.64/7.78 | characterized |
| uma | 700 | 115 | 4.75/7.78 | characterized |
| uma | 800 | 124 | 4.59/7.78 | characterized |

## T8: performance (V100-16GB)

### Single-point warm latency (median ms)

| atoms | mace | dpa | grace | uma |
|---|---|---|---|---|
| 32 | 714 | 1.07e+04 | 9.8 | 121 |
| 128 | 64.5 | 167 | 15.3 | 121 |
| 512 | 202 | 176 | 38.7 | 303 |

### NVE MD throughput (steps/s / atom-steps/s)

| atoms | mace | dpa | grace | uma |
|---|---|---|---|---|
| 128 | 14.8 / 1.89e+03 | 6.64 / 850 | 62.4 / 7.99e+03 | 8.1 / 1.04e+03 |
| 512 | 4.95 / 2.53e+03 | 5.52 / 2.83e+03 | 26 / 1.33e+04 | 3.3 / 1.69e+03 |

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
| invariance_caching | model_characterized | model_characterized | model_characterized | model_characterized |
| fixed_cell_relax | software_validated | software_validated | software_validated | software_validated |
| cell_relax | software_validated | software_validated | software_validated | software_validated |
| eos | software_validated | software_validated | software_validated | software_validated |
| elastic | software_validated | software_validated | software_validated | software_validated |
| phonon | software_validated | model_characterized | software_validated | model_characterized |
| thermodynamics | model_characterized | model_characterized | model_characterized | model_characterized |
| defect | software_validated | software_validated | software_validated | software_validated |
| surface | software_validated | software_validated | software_validated | software_validated |
| formation_energy | model_characterized + unsupported | model_characterized + unsupported | model_characterized + unsupported | model_characterized + unsupported |
| neb | model_characterized | model_characterized | model_characterized | model_characterized |
| saddle_hessian | software_validated | software_validated | software_validated | software_validated |
| short_nve | software_validated | software_validated | software_validated | software_validated |
| short_nvt | software_validated | model_characterized | model_characterized | model_characterized |
| transport_demo | model_characterized | model_characterized | model_characterized | model_characterized |
| mechanism_analysis | model_characterized | model_characterized | model_characterized | model_characterized |
| arrhenius | model_characterized | model_characterized | model_characterized | model_characterized |
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
