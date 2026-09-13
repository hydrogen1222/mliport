# Changelog

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
