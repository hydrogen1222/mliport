# Scientific Validation Harness (beta)

Reproducible GPU validation of the four production engines that mliport
wraps (UMA, MACE-OMAT-0, DPA-3.1-3M, GRACE-2L-OMAT).  This directory holds
the harness: manifests, deterministic fixtures, per-tier scripts, JSON
schemas, and the aggregator.  **Run outputs stay out of git** — only the
harness, the pinned manifests, and the OMat24 subset provenance are
committed.

## Layout

| Path | Purpose |
| --- | --- |
| `model_manifest.json` | Pinned model identities (sha256 of the exact artifacts; no local paths) |
| `capability_matrix.json` | Machine-readable interface capability matrix (API / direct CLI / INCAR / TUI); not an equivalence claim |
| `data_manifest.json` | Pinned OMat24 validation-subset provenance (corrected 241220 files) |
| `scripts/common.py` | Shared schemas, identity/UUID hashing, result records |
| `evidence/` | Canonical evidence layer: strict loader, identity, schema dispatch and profile-aware aggregation. Report/README/figure generators only consume this package |
| `scripts/engines.py` | Engine construction through the product `CalculatorFactory` (no fallbacks) |
| `scripts/fixtures.py` | Deterministic bulk fixtures (Cu, Si, MgO + triclinic/distorted extras) |
| `scripts/inference.py` | T1 — first-call/warm-call timings, E/F/stress, VRAM |
| `scripts/invariance.py` | T2 — repeatability floor, permutation/PBC/translation/rotation invariance, A→B→A cache-state check |
| `scripts/finite_difference.py` | T2-FD — ASE native FD sweep vs analytical forces/stress |
| `scripts/omat_eval.py` | T3 — common OMat24 held-out accuracy on the pinned 256-structure validation subset (seed 20260911, 4 strata): energy/atom MAE-RMSE-median-p95, force component MAE/RMSE + cosine statistics + error-vs-magnitude strata, stress parity only if reference labels exist (official val LMDB carries none). Per-structure errors exported to CSV for figures |
| `scripts/static_suite.py` | T4 — static properties: single points, relaxations, EOS, elastic constants, phonons/thermo, vacancy, surface, energetics |
| `scripts/neb_suite.py` | T5 — NEB/CI-NEB: endpoint symmetry, path init, warm-up, CI convergence, resume identity, FD-Hessian saddle verdict, Na3PS4 hop |
| `scripts/md_suite.py` | T6 — MD: NVE timestep drift sweep, NVT Langevin seed reproducibility, Bussi/CSVR + FixAtoms, Nose-Hoover chain with rejection matrix, constraint exactness/DOF, force-safety abort |
| `scripts/analysis_suite.py` | T7 — physical transport analysis on alpha-Na3PS4 (mp-28782 2x2x2): product MDRunner NVT-NHC trajectory (700 K, literature-motivated superionic regime), kinisi tracer diffusion + Nernst-Einstein (transport authority), native-MSD diagnostic + sampling sufficiency gate, GEMDAT site/jump diagnostic on explicitly pinned crystallographic Na sites, optional >=3-temperature Arrhenius. Two-phase: `--cases md` runs in the engine venv (produces `md_runs/`); `--cases transport,gemdat` runs in the analysis venv on the saved trajectories. Demonstration only — not converged bulk conductivity |
| `scripts/benchmark.py` | Tiered entry point (`core`/`standard`/`research`) for the literature suite: imports the pinned references, runs `t9`, optionally runs the tier plan below, aggregates and writes the bilingual report |
| `scripts/literature_eval.py` | T9 — literature-referenced DFT accuracy (LiPS-25, LiPS20, OMat24, JARVIS-DFT, local MS25): energy/atom, within-group relative energies, formation energy + ranking, force component/vector/direction statistics per element and force stratum, and stress only when a verified label exists |
| `scripts/literature_report.py` | T9 report: Markdown (EN/中文) + machine-readable JSON built only from the canonical evidence loader |
| `literature/` | Literature registry (`registry.json`: DOI, upstream commit, SHA-256, licences, DFT settings, units, selection rules), download/cache layer, deterministic importers and the pure-NumPy metric library |
| `scripts/performance_suite.py` | T8 — GPU performance characterization (section 33): single-point latency/VRAM scaling on alpha-Na3PS4 supercells (32/128/512 atoms; first call, warmed median + spread, atoms/s), NVE MD throughput through the ASE upstream VelocityVerlet integrator (steps/s, atom-steps/s, peak VRAM, explicit synchronization), GPU out-of-memory recorded as `characterized` with reason (never `pass`) |
| `scripts/transforms.py` | Rotation/Voigt algebra with hand-verified known answers |
| `scripts/build_cases.py` | Regenerates/pins `cases/manifests/fixtures.json` |
| `scripts/omat_subset.py` | OMat24 val-subset acquisition + deterministic selection |
| `scripts/run_suite.py` | Sequential per-engine orchestrator (backend venvs) |
| `scripts/aggregate.py` | Thin CLI over `evidence/` (validates + aggregates records into the canonical summary) |
| `schemas/` | JSON Schemas for result records and summaries |

## T9 literature benchmark semantics

T9 is additive: it reuses the result schema, the evidence layer, the
aggregator, `engines.build_engine` and the per-engine venv isolation.

- **Data never enters git unless its licence is clear.** `literature/registry.json`
  pins the paper, DOI, upstream commit, per-file SHA-256, DFT settings,
  units and licence of every dataset. `lips25` and `lips20` have no usable
  data licence, so they are downloaded into a local cache and imported there;
  the committed `omat24` subset is reused as-is (and its hash is verified).
- **Software verification and model accuracy are separate.** `status=pass`
  means every requested structure was evaluated with finite values.
  Accuracy is recorded, never converted into a pass/fail claim; no universal
  MLIP error threshold exists.
- **Comparison mode is explicit.** `omat24` is `reference_matched` (the
  checkpoints trained on it); `lips25` (PBEsol), `lips20` (PBE-D3) and
  `jarvis` (OptB88vdW) are `cross_method`. For cross-method references the
  absolute energy block is a labelled diagnostic; forces, within-group
  relative energies, formation energies and rankings are the interpretable
  quantities, and different reference levels are never averaged together.
- **A virial is not a stress.** LiPS-25 and LiPS20 publish a `virial` array
  whose sign/normalisation is undocumented; it is hashed and recorded as
  provenance (`reference_virial_used_as_stress: false`) and stress is
  reported unavailable. No hand-flipped sign is ever introduced.
- **Missing labels are reported, not invented.** Unlabelled LiPS-25
  `task1_unrelaxed` structures are relaxed with ASE `ExpCellFilter`+`BFGS`
  and compared against the DFT-relaxed geometry, with the per-structure
  convergence flag, `fmax` and step budget recorded.
- **Selections are deterministic and traced.** Per-tier rules resolve from
  the registry seed; `<out>/data/<dataset>/selection.json` records the rule,
  chosen source indices, upstream commit and input hashes.
- **The research tier does not claim T7.** T7 is a two-phase task (MD in the
  engine venv, transport/GEMDAT in the analysis venv) and `run_suite.py` has
  no analysis-venv channel, so `benchmark.py` records both phases under
  `resolved.json -> manual_tasks.t7` and leaves T7 to a deliberate manual run.
- **Post-processing cannot fake success.** `aggregate.py` and
  `literature_report.py` exit codes are collected in `postprocess.json` and
  propagated; a failed roll-up returns non-zero while the already-written t9
  records remain untouched.

Quickstart for one engine:

```bash
python validation/science/scripts/benchmark.py \
  --tier core --datasets lips25,lips20,omat24 \
  --engine mace --profile-id mace_omat --model models/mace/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/literature
```

## Core rules encoded here

- **Calculator identity is proven, not assumed** — every record stores the
  live wrapper module/class; anything outside the mliport wrappers aborts
  the run (`common.calculator_identity`).
- **Tolerances are never hard-coded by hand** — they are `max(10 × measured
  repeatability floor, absolute floor)`, and the policy is stored in every
  record's `parameters`.
- **A changed artifact is a new identity** — `resolve_profile` refuses to
  run on hash mismatch; the manifest must be updated deliberately.
- **Stress signs are never hand-flipped**; FD stress vs analytical stress
  is compared component-wise (normals and shears separately).
- **GPU UUIDs are hashed** before entering any record.
- **Status semantics for T4** — `pass` = workflow-internal convergence/fit
  criteria met; `characterized` = done but the value characterizes the
  model (defect/surface energies) or the criterion describes the model
  itself; `fail` = non-convergence, bad fit, or robust imaginary modes;
  `unsupported` = a required reference is not pinned (e.g. isolated-atom
  energies for cross-engine formation energies), never silently faked.
- **Supercell convergence is proven, not assumed** — phonons run 2³ and
  3³ supercells at three displacement amplitudes; a robust imaginary mode
  (< -0.1 meV) anywhere on the sampled q-grid fails the record, and the
  2×2×2 vs 3×3×3 delta-convergence is part of the evidence.
- **Na3PS4 (mp-28782) is a geometry fixture only** — pinned provenance in
  `data_manifest.json`; never compared against Materials Project energies.

- **T5 barrier semantics** — the barrier is the sampled maximum minus the
  initial endpoint energy; the reverse barrier is reported even when the
  endpoints are inequivalent, and no fitted saddle value is ever invented
  (`barrier_status` states what was actually computed).  A bounded budget
  that runs out is recorded `not_converged`, never extended silently.
- **T5 saddle verdicts are gated, not claimed** — a CI image is a
  `validated_first_order_candidate` only when exactly one robust negative
  FD-Hessian eigenvalue (< -0.01 eV/A^2, i.e. 10x the zero-mode scale)
  persists at 0.005/0.010/0.020 A displacements, the eigenvalue spread stays
  within 50% of the mean, and |<v-, tangent>| >= 0.8; otherwise it is
  `ci_neb_candidate_only`, and two or more robust negative modes fail the
  record as `bad_saddle_candidate`.

- **T6 MD semantics** — every run drives the product `MDRunner` with a
  pinned seed and `velocity_policy="initialize"` so all engines start from
  identical coordinates and velocities (identical KE(0) is part of the
  evidence); trajectories across backends are never compared atom-by-atom
  (chaotic divergence) — only integration/ensemble diagnostics are.  The NVE
  drift gate is the broad trend `|slope(0.25 fs)| < |slope(2.0 fs)|` in
  eV/atom/ps; no global drift cutoff is invented.  NHC is validated only in
  its capability-matrix-allowed form (unconstrained, `com_policy="none"`);
  every wrong-DOF combination (NHC+constraints, NHC+COM removal, compound
  constraints) must be refused, never silently integrated.  The force-safety
  abort derives its threshold from a measured pre-flight force (half of it,
  both numbers recorded) and must checkpoint the unsafe frame, write an
  `aborted` artifacts manifest with `error.type=force_safety_abort`, and
  leave a CONTCAR.

## Quickstart (single engine, backend venv)

```bash
python validation/science/scripts/run_suite.py \
  --tiers t1,t2,t2fd \
  --engine mace --profile-id mace_omat \
  --model models/mace/mace-omat-0-medium.model \
  --manifest validation/science/model_manifest.json \
  --out .validation-work
```

Aggregate everything:

```bash
python validation/science/scripts/aggregate.py \
  --results .validation-work \
  --manifest validation/science/model_manifest.json \
  --out .validation-work/summary.json
```

The aggregator validates every declared result record against the committed
versioned JSON Schema plus semantic identity checks, aggregates per
**profile** (engine × artifact hash × dtype × task/head × inference mode ×
device class × commit × seed) so a float32 failure can never be hidden by a
float64 pass, and keeps every run of the same computation visible.  Legacy
`mliport.beta-validation-result/1` records are migrated in memory, so existing
raw evidence can be re-aggregated without rerunning any model.  Exit codes:
`0` clean, `1` harness violation (calculator identity outside the allowed
wrappers), `2` import problem, `3` no records, `4` `--expect-complete` was
requested and a declared engine produced no evidence.  New records declare
`profile_id` / `record_id` / `run_id`; `common.write_result` never
overwrites a different record under the same name — it publishes an
explicitly versioned `...__run-<id>.json` sibling instead.

CPU-only regression tests (no backend needed):

```bash
pytest tests/mliport/mliport/test_science_harness.py -q
```
