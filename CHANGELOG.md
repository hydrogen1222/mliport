# Changelog

## Unreleased

Internal (no user-visible behaviour change):

- **Shared runner safety helpers.** The numerical-safety gates that MD already
  enforced (atomic-state validation, finite energy/force checks, raw vs
  constraint-applied force views, the raw-force `fmax_abort` evaluation and the
  per-frame safe-point validation) now live in `mliport/runners/safety.py`, and
  `MDRunner`/`BaseRunner` delegate to them. The energy–force gradient pre-flight
  accepts one or more structures (NEB still requires at least two images and
  keeps its exact refusal wording and schema), and the model-identity record is
  a shared helper that additionally records `dispersion`. No scientific record
  invalidated and no result changes.

New features:

- **Configuration search (`confsearch`, new `CALC_TYPE = CONFSEARCH`).** MD-based
  configuration search for one adsorbate on a periodic slab (spec revision 4,
  tracking PR #61): multiple starts (user-prepared mode A, or automatic
  placement mode B with `cs_place_anchor` chemisorption-first starts),
  Langevin NVT heat/hold/cool cycles with per-step finite/force gates and an
  early adsorbate-integrity check, FIRE quenches at fixed cell, Metropolis
  restarts, deduplication by energy plus a direct geometric identity rule `G`
  (fixed slab) or the focused Oganov-Valle fingerprint `D`, top-K refinement,
  saturation diagnostics ("lowest found", never a global minimum), and exports
  for DFT (`POSCAR` with selective dynamics, `structure.extxyz`,
  `cp2k_coord.xyz`, `cp2k_cell.inc`, `cp2k_fixed_atoms.inc`, `minimum.json`).
  Checkpoint/resume is bitwise-identical on CPU; the model identity includes
  `dispersion`, so a D3 search can never be merged with a plain one. Interfaces:
  `mliport confsearch run|place|resume`, `mliport.api.run_confsearch`,
  `CS_*` INCAR keys, and a capability-matrix entry. `scipy` and `matplotlib`
  were added to the `dev` extra so the ASE OFP known-answer test runs in CI.
  `mliport confsearch extend RUN --cycles N` adds cycles to a finished search
  (previous results move to `history/extension_N/`; only `cs_cycles` is exempt
  from the resume fingerprint), and `mliport confsearch merge RUN1 RUN2
  --output DIR` combines comparable finished searches without new calculations
  (refusing runs whose comparability key differs). The bilingual workflow page
  `docs/workflows/confsearch.md` (+ `.zh.md`) documents the scope, the outputs,
  the dedup rules and the caveats. The TUI entry, queue support for
  confsearch and the real-model validation campaign are later phases and are
  marked `no` in the capability matrix.

- **DFT-D3 dispersion.** `--d3` / `D3 = .TRUE.` / API `d3=True` adds Grimme
  DFT-D3 (via `torch-dftd`, float64) on top of any backend as
  `SumCalculator([model, D3])`; `--d3-xc` (default `pbe`) names the model's
  reference functional and `--d3-damping` (default `bj`) the damping. Fails
  closed: UMA tasks whose reference already contains dispersion (`omol`,
  `omc`, `odac`, `oc25`) are refused, UMA `omat`/`oc20` must use `pbe`/`rpbe`,
  D3 sub-options without `--d3` are an error, and NEB refuses D3 until its
  model identity records the term. The correction is recorded as
  `dispersion` in the run metadata and in the OUTCAR-style header. Optional
  extra: `mliport[d3]`.

Documentation site:

- **Docs site.** The documentation is now a MkDocs Material site published to
  GitHub Pages (`mkdocs.yml`, `.github/workflows/docs.yml`, toolchain in
  `requirements-docs.txt`). English is the default; a language switch in the
  header selects 中文 (`page.md` / `page.zh.md`, mkdocs-static-i18n).
- **Generated facts, hand-written prose.** `scripts/generate_docs.py` no longer
  writes whole pages. It refreshes marker-delimited blocks (backend/Python
  matrix, GPU install routes, per-backend environment and model profiles, NEB
  capability matrix) inside pages written by hand, in both languages.

Review fixes (scientific correctness, data safety, robustness):

- **IDPP convergence.** Check the projected-plus-spring NEB band force used by
  ASE's IDPP optimizer, not just raw pair forces. Unfinished bands are refused.
  NEB scientific revision is now 2, preventing resume across this change.
- **NEB checkpoint identity.** Preserve custom-array `atom_id` / `atom_ids` in
  ASE image info so integer/string IDs and explicit atom maps survive repeated
  save/resume cycles without mutating caller data.
- **DPA load retries.** A model that fails metadata/head inspection is never
  cached; retrying cannot bypass validation.
- **NEB identity input.** Direct `BandInput` construction rejects float and
  boolean atom maps/lattice shifts instead of silently converting them to integers.
- **Spectrum compatibility.** Area normalization works with NumPy 1.x as well
  as 2.x. The bilingual docs now state that VACF/spectrum require stored velocities.
- **MD / NHC drift.** NHC no longer keeps the random Maxwell-Boltzmann
  centre-of-mass momentum: `com_policy=auto` (and `initialize_only`) zero it
  once, NHC conserves it, the run reports `3N-3` kinetic DOF and the
  thermostat set point is scaled by `(3N-3)/3N` (`nhc_set_temperature_K`).
  Previously every NHC run translated the whole cell ballistically.
- **Analysis COM-drift guard.** `msd` and `transport` record
  `com_drift_diagnostics` and warn when an uncorrected whole-system COM
  displacement exceeds 5% of the mobile-ion MSD.
- **No silent overwrite.** sp/opt/md/batch refuse an output directory that
  already holds calculation outputs (earlier runs, or VASP OUTCAR/CONTCAR)
  unless `--overwrite`/`overwrite=True`, and hold an exclusive run lock.
  `calculate_energy()` uses a temporary directory by default.
- **Pre-relaxation stays fail-closed.** If it does not reach
  `pre_relax_fmax` within `pre_relax_steps`, the run stops before production
  MD with an actionable error (raise the step budget, loosen the threshold
  with scientific justification, or disable pre-relaxation explicitly). The
  outcome of a completed pre-relaxation is recorded in
  `md_provenance.pre_relax`.
- **UMA MD precision.** MD no longer defaults to `inference_mode=turbo`
  (TF32 + torch.compile); turbo is opt-in.
- **TUI templates** are the backend-neutral single-source templates (the old
  ones hard-coded UMA and used unknown NEB keys).
- **Analysis memory.** Trajectories are streamed frame by frame; MD frames no
  longer duplicate the constraint-applied forces in `atoms.info`.
- **GPU identity.** torch backends verify the device inside the CUDA
  runtime's own enumeration (no nvidia-smi dependency, correct on mixed-GPU
  nodes); CLI/queue default `CUDA_DEVICE_ORDER=PCI_BUS_ID`; a single MIG
  instance in `CUDA_VISIBLE_DEVICES` is supported.
- **Queue.** `mliport queue start --max-jobs-per-gpu N` lets small jobs share
  a GPU (default 1, exclusive).
- **Static NEB evidence removed.** The packaged `mliport/data/validation`
  records, the evidence matching in `mliport.capabilities` and the wheel
  smoke assertions that depended on them are gone: the runtime pre-flight is
  the only energy-force check. `--allow-unvalidated-neb` /
  `allow_unvalidated_neb` / `NEB_ALLOW_UNVALIDATED` no longer exist in the
  CLI, config, TUI, queue and API (strict config rejects them). Direct-force
  models are still rejected. The V100 provenance records under `validation/`
  are kept as history.
- **NEB docs.** README wording on NEB validation fixed (evidence matching is
  unchanged and stays exact).
- **NEB runtime pre-flight.** Before the first step NEB measures whether the
  model's forces are the gradient of its energy on the band to be optimized
  (endpoints and interpolated images; force and random directions; central
  finite differences at two step sizes with Richardson extrapolation and a
  dtype-aware energy-noise tolerance) and refuses to run otherwise, instead
  of relying on static evidence recorded on another PyTorch/CUDA build.
  Direct-force models are still always refused. The record is stored as
  `gradient_preflight` in `run_context.json`/`neb_results.json`.
  `--allow-unvalidated-neb` is deprecated and has no effect; it can no longer
  bypass a failed check. This is a spot check, not a proof over the whole
  energy surface and not a statement about model accuracy.
- **NEB progress in run.log.** The NEB runner never wrote optimizer progress,
  so `run.log` stayed silent until the end. It now logs the pre-flight, the
  start/finish of every FIRE stage and a progress line (max physical force,
  highest interior image energy) for the first step and every 5th step.
- **Contributing.** The development-environment commands match CI
  (`uv sync` at the repository root installs nothing).
- **Python 3.13** is supported (package, installer, CI).
- **Maintainability.** One backend calculator-option table replaces three
  copies; engine fallbacks read `BUILTIN_DEFAULTS`; unused `mliport/templates`
  removed; root `LICENSE.md` matches the package license.
- **CI.** pytest failures and ruff findings are emitted as GitHub
  annotations.

## 2.0.0b4

Repository freeze / state isolation / compatibility audit beta:

- removes the last implicit UMA defaults from queue task files and the TUI,
  so a run without an explicit backend (or a checkpoint that supplies one)
  fails closed;
- moves persistent job history to the OS state directory, records the state
  schema/application/version, scopes records to their project, adds
  `mliport state path`/`state migrate`, and keeps legacy `~/.mliport` records
  explicitly import-only;
- unifies CUDA device resolution so the CLI isolation re-exec and the queue
  device lease honour an existing `CUDA_VISIBLE_DEVICES` mapping and choose
  the same physical GPU;
- removes the deprecated `mliport/calculator.py` shim and the accidental
  top-level `examples` package from the distribution;
- audits the pinned frameworks at the binary level for Maxwell through
  Blackwell, records wheel hashes and compiled architectures, ships the
  audit to `mliport doctor`, and keeps non-Volta families at
  `binary_theory_verified`/`experimental` instead of claiming hardware
  verification;
- adds a wheel/sdist content gate to CI and makes the generic MD template
  device default CPU-safe.


## 2.0.0b3

Recovery/release-bridge beta polish:

- reconciles the productization tree with the validation-report lineage; Git
  forensics proved `v2.0.0b2`/`0f53dbb` is an ancestor of `main`, so no
  history rewrite or force-push was required;
- locks the productization invariants with behavior-level CI tests
  (backend-neutral selection, strict config, project-scoped citation, docs
  installer matrix, rename hygiene, package metadata);
- preserves the corrected citation/docs/installer contracts and the
  per-backend installation model;
- consolidates current vs historical validation reporting
  (`target_commit_revalidated`, tier scopes, historical section, GO evidence
  queries) and adds the exact-release four-backend bridge smoke with its own
  archive;
- refines the T7 estimator audit with two deterministic OLS windows on the
  same trajectory and an explicit estimator-model difference flag, without
  forcing kinisi and OLS to agree;
- retains the T8 startup/warm benchmark separation and records warmup stop
  reasons plus the actual backend GPU-synchronization method.

## 2.0.0b2

Beta polish after `2.0.0b1`:

- **Backend-neutral core.** No implicit UMA default anywhere; the backend
  must be explicit (`--model-type`/`MODEL_TYPE`, model alias/profile) or the
  run fails closed. `fairchem` remains an alias for UMA.
- **Strict scientific config by default.** Misspelled and cross-backend keys
  are fatal; `--lenient-config` / `strict_config = false` is the explicit
  opt-in to the legacy warning mode.
- **Installer UX.** Environment names, Python ranges and runtime pins come
  from the compatibility registry and are machine-generated into
  `docs/installation.md`; `./bin/mliport-<engine>` launchers only select the
  runtime; backend-incompatible `--python` requests fail closed.
- **Documentation architecture.** New `docs/` tree, rewritten READMEs,
  CONTRIBUTING, troubleshooting (DPA head fails closed) and an mlipx
  migration guide, with link/CLI/generated-block contract tests.
- **Rename hygiene.** Backend-neutral brand tagline and a CI guard against
  `mlipx` residue in current code/docs.
- **Validation narrative.** Explicit per-tier scope
  (current_campaign/historical_reuse/not_run), historical evidence listed
  separately, `target_commit_revalidated` naming and machine-linked GO
  checklist queries.
- **T8 benchmark contract.** Startup/warmup/timed separation, full latency
  spread, anomaly flags, and a V100 rerun (four backends plus GRACE cache
  ON/OFF) that showed the old 32-atom anomalies were cold-start artifacts.
- **T7 transport audit.** Aligned plain-OLS baseline, three-estimator
  comparison, known-answer unit audit (`1 Å²/ps = 1e-8 m²/s`) and the
  `kinisi_to_plain_D_ratio` disagreement warning; trajectories reused.

## 2.0.0b1

Initial public beta: MACE/DPA/GRACE/UMA adapters, SP/OPT/MD/NEB/batch
workflows, queue/TUI/API, trajectory analysis suite and the V100 validation
campaign anchored to software commit `5f8d91d`.
