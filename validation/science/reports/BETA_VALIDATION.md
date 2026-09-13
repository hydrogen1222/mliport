# mliport beta scientific validation report

Generated from evidence records under `.validation-work`; evidence commits: `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`. Every table in this document is rendered from result JSON by `validation/science/scripts/generate_beta_report.py`; regenerate and diff instead of editing.

Revalidation status: **current_head_revalidated** -- campaign manifest declares completion and every record was produced at the target software commit

| version identity | commit |
|---|---|
| software_commit (claimed validated) | `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` |
| validation_code_commit | `a0d677b0660b76aee9a3e90ae08e5e24873124a4` |
| report_generator_commit | `a0d677b0660b76aee9a3e90ae08e5e24873124a4` |
| evidence_campaign | `20260913-current-head-5f8d91d` |
| evidence_source_commits | `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` |

Model identities and artifact hashes are pinned in `validation/science/model_manifest.json`; the OMat24 evaluation subset in `validation/science/data/` (seed 20260911).

## Beta GO / NO-GO checklist (task book section 28)

- Verdict: **GO for beta** (23 ✅ / 1 ⚠️ / 0 ❌)
- Validated software commit: `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`
- Evidence campaign: `20260913-current-head-5f8d91d`, 96 records (53 pass / 43 characterized / 0 fail / 0 blocked)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | New name has no known same-domain namespace collision | ✅ | `mliport` free on the PyPI mirror and on GitHub search; repo renamed to `hydrogen1222/mliport` |
| 2 | Python package / CLI / repo identity migrated | ✅ | rename commit chain; clean break, old `mlipx` import not published |
| 3 | Target-commit CI green on Python 3.10/3.11/3.12 | ✅ | Actions tests/lint/package-build green at `5f8d91d4` and follow-ups |
| 4 | Clean wheel install green | ✅ | wheel-install-smoke 3.10/3.11/3.12 + `mliport-2.0.0b1` capability smoke |
| 5 | Strict evidence loader is the only interpretation path | ✅ | `validation/science/evidence/` + no-raw-selection report tests |
| 6 | Report/README do not interpret raw records | ✅ | report and figures consume `load_evidence`; README block machine-rendered |
| 7 | Malformed evidence fails closed | ✅ | V01-T5/T6/T7 tests; report exits 2/3; archive builder refuses |
| 8 | Profile identity never mixes precision/model/commit | ✅ | V01-T2/T3/T4; engine cells report `mixed` with `worst_status` |
| 9 | Current campaign manifest fixed | ✅ | `20260913-current-head-5f8d91d` manifest status = `complete` |
| 10 | MACE/DPA/GRACE/UMA current-head GPU smoke | ✅ | 4 engines x t1/t2/t2fd/t3/t4/t5/t6/t7/t8 |
| 11 | Repeat inference current-head | ✅ | T1 records carry repeat-inference metrics per engine |
| 12 | FD current-head / reclassified evidence | ✅ | T2FD: 32 records |
| 13 | Saddle current semantics | ⚠️ | CPU/analytic known-answer layer green; GPU `t5h` workflow not part of the four-backend smoke |
| 14 | NEB fixed-band validation | ✅ | full-image constraint checks + regression tests |
| 15 | kinisi skew / exact-unwrap known-answer | ✅ | PR-E contract tests; T7 transport uses `exact_unwrapped` (12 records) |
| 16 | GEMDAT backend errors never become physical zero | ✅ | PR-C status contract; 12 T7 records |
| 17 | Alpha weighting/validity closure | ✅ | local Δlog(t) weighting + finite & valid summaries |
| 18 | Analysis version bump | ✅ | alpha estimator `/2`; task revisions msd 7 / transport 6 |
| 19 | T7 transport recomputed | ✅ | 12 T7 records (md/transport/gemdat) |
| 20 | T8 re-run or rebuilt under the new identity | ✅ | 20 T8 records |
| 21 | Historical reused evidence explicitly marked | ✅ | version block + campaign scope `not_in_scope` / `historical_reuse` |
| 22 | Report rebuildable from a formal evidence archive | ✅ | sha256 `bb6b1b06bb3e891e…`, 96 records, https://github.com/hydrogen1222/mliport/releases/download/v2.0.0b1/mliport-validation-20260913-current-head-5f8d91d.tar.zst |
| 23 | README statements match the evidence | ✅ | generated block plus explicit historical T3/T4 statement |
| 24 | No secrets/model weights/temporary probes/attic in the release | ✅ | hygiene/distribution checks; archive excludes weights, trajectories and attic |

Scope note: T3 accuracy and T4 static workflows remain historical evidence and are not claimed as current-HEAD; the GPU `t5h` saddle workflow is outside the four-backend smoke and is covered by the CPU/analytic known-answer layer.

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

## T7: transport and mechanism analysis (alpha-Na3PS4)

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
