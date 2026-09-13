Status: post-fix beta candidate. Software CI is validated on Python 3.10-3.12. Historical scientific evidence has been retained and is being reclassified under the current validation semantics. Current-HEAD scientific revalidation is pending.

Validation status per backend, rendered from the beta evidence records (`beta-summary.json`; t1-t8 tiers, 4 backends x OMat24 common subset). `software_validated` means the mliport integration and all recorded checks passed; `model_characterized` means the workflow ran and its behavior was recorded, including honest failures (e.g. float32 arithmetic noise). Full per-test tables: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md).

| Workflow | MACE | DPA | GRACE | UMA |
|---|---|---|---|---|
| Install & doctor (CI software tests) | software_validated* | software_validated* | software_validated* | software_validated* |
| Single-point inference (4 structures) | software_validated (passx26) | software_validated (passx23) | software_validated (passx23) | software_validated (passx23) |
| Energy-forces consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx8) | model_characterized (characterizedx4) |
| Stress (finite-difference cross-check) | model_characterized (characterizedx4, passx6) | model_characterized (characterizedx4, passx3) | model_characterized (characterizedx8, passx3) | model_characterized (characterizedx4, passx3) |
| Stress-energy consistency | model_characterized (characterizedx4) | model_characterized (characterizedx4) | model_characterized (characterizedx8) | model_characterized (characterizedx4) |
| Coordinate invariance & cache | model_characterized (characterizedx4, passx20) | model_characterized (characterizedx4, failx18, passx2) | model_characterized (characterizedx8, failx18, passx22) | model_characterized (characterizedx4, failx18, passx2) |
| Fixed-cell relaxation | software_validated (passx12) | software_validated (passx12) | software_validated (passx12) | software_validated (passx12) |
| Cell relaxation | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) |
| EOS / bulk modulus | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Elastic constants | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) | software_validated (passx4) |
| Harmonic phonons | software_validated (passx12) | model_characterized (failx1, passx11) | software_validated (passx12) | model_characterized (failx1, passx11) |
| Harmonic thermodynamics | model_characterized (failx6, passx6) | model_characterized (failx3, passx9) | model_characterized (failx6, passx6) | model_characterized (failx6, passx6) |
| Vacancy formation energy | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) | software_validated (passx3) |
| Surface energy | software_validated (passx6) | software_validated (passx6) | software_validated (passx6) | software_validated (passx6) |
| NEB | model_characterized (characterizedx1, passx7) | model_characterized (characterizedx1, passx7) | model_characterized (characterizedx1, passx7) | model_characterized (characterizedx1, passx7) |
| Saddle-point Hessian | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) |
| Short NVE / NVT MD | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) | software_validated (passx1) |
| Transport analysis (demonstration) | model_characterized (characterizedx3, passx3) | model_characterized (characterizedx3, passx3) | model_characterized (characterizedx3, passx3) | model_characterized (characterizedx3, passx3) |
| Mechanism analysis (GEMDAT) | model_characterized (characterizedx3) | model_characterized (characterizedx3) | model_characterized (characterizedx3) | model_characterized (characterizedx3) |
| Performance (SP/MD scaling) | software_validated (passx5) | software_validated (passx5) | software_validated (passx5) | software_validated (passx5) |

Held-out OMat24 accuracy (E/atom MAE over 256 structures, no elemental offsets fitted): MACE (float64) 0.0172 eV; DPA 0.0184 eV; GRACE 0.0143 eV; UMA 0.0108 eV. Stress parity is not computable: the official OMat24 validation split carries no reference stress labels.

`*` = CI software test only, no model involved. A cell lists the recorded statuses for that workload; per-workload rows reuse the same evidence tiers, so row counts are not additive. Full per-test tables and limitations: [BETA_VALIDATION.md](validation/science/reports/BETA_VALIDATION.md). Model identities pinned in `validation/science/model_manifest.json`.
