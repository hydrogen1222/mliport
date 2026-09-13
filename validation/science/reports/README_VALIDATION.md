Status: beta validation completed at software commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` (campaign `20260913-current-head-5f8d91d`).

The full scientific beta campaign targets commit `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`. The release candidate `6e1a945f91c7` retains that validated scientific implementation and adds productization/release-layer changes. A four-backend release-bridge smoke (`20260913-release-bridge-6e1a945f`) was executed at the release-candidate commit to verify configuration, backend selection, model loading and single-point inference paths. No long scientific trajectories were regenerated.

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
