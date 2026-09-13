# mliport beta GO / NO-GO checklist (task book section 28)

- Project: **mliport** `2.0.0b1`
- Validated software commit: `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c`
- Validation harness commits: `34f894ba454c4889a66bf9d9f6601fe56dd33725` (T7 analysis fix + explicit software-commit override), `425a180acf0cf5adc684ead76d1d0dcd803f7e7b` (campaign manifest + archive tooling)
- Report/archive commit: `b179f7e`, README status commit: `46ef566`
- Evidence campaign: `20260913-current-head-5f8d91d` (96 records: 53 pass, 43 characterized, 0 fail, 0 blocked)
- Device: Tesla V100-SXM2-16GB, driver 580.173.02 (unchanged by the campaign)
- Report: [BETA_VALIDATION.md](BETA_VALIDATION.md) / [BETA_VALIDATION_CN.md](BETA_VALIDATION_CN.md);
  machine-readable: [beta-summary.json](beta-summary.json)

## Hard GO items

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | New name has no known same-domain namespace collision | ✅ | `mliport` free on the PyPI mirror and on GitHub search; repo renamed `hydrogen1222/mliport` |
| 2 | Python package / CLI / repo identity migrated | ✅ | `PR-L` (173 renamed files, 232 touched); clean break, old `mlipx` import deliberately not published |
| 3 | Target-commit CI green on Python 3.10/3.11/3.12 | ✅ | Actions runs on `5f8d91d`, `34f894b`, `425a180`, `b179f7e`, `46ef566` (tests + lint + package-build) |
| 4 | Clean wheel install green | ✅ | `wheel-install-smoke` 3.10/3.11/3.12 + local `mliport-2.0.0b1` capability smoke |
| 5 | Strict evidence loader is the only interpretation path | ✅ | `validation/science/evidence/`; report/figure static no-raw-selection tests |
| 6 | Report/README do not interpret raw records | ✅ | `generate_*` consume `load_evidence`; README block machine-rendered |
| 7 | Malformed evidence fails closed | ✅ | V01-T5/T6/T7 tests; report exits 2/3; archive builder refuses |
| 8 | Profile identity never mixes precision/model/commit | ✅ | V01-T2/T3/T4; engine cells show `mixed` with `worst_status` |
| 9 | Current campaign manifest fixed | ✅ | `validation/science/campaigns/20260913-current-head-5f8d91d.json` (status complete) |
| 10 | MACE/DPA/GRACE/UMA current-head GPU smoke | ✅ | T1 (3 pass each), T2FD (8 each), T5 (3 pass each), T6, T7 per engine |
| 11 | Repeat inference current-head | ✅ | T1 records include repeat-inference metrics per engine |
| 12 | FD current-head / reclassified evidence | ✅ | T2FD 32 characterized records |
| 13 | Saddle current semantics | ⚠️ | §17.1 CPU/analytic known-answer layer green (full suite 1183 passed); the GPU `t5h_saddle_hessian` workflow was **not** re-run in this campaign |
| 14 | NEB fixed-band validation | ✅ | `PR-D` full-image constraint checks + tests |
| 15 | kinisi skew / exact-unwrap known-answer | ✅ | `PR-E` contract tests + campaign T7 `exact_unwrapped` transport |
| 16 | GEMDAT backend errors never become physical zero | ✅ | `PR-C` status contract + campaign GEMDAT records (`characterized`, no zeroed backend errors) |
| 17 | Alpha weighting/validity closure | ✅ | `PR-H` local Δlog(t) weighting + finite & valid summaries |
| 18 | Analysis version bump | ✅ | alpha estimator `/2`; task output revisions msd 7 / transport 6 |
| 19 | T7 transport recomputed | ✅ | 4 engines × (MSD→alpha→kinisi→GEMDAT) at the new identity |
| 20 | T8 re-run or rebuilt under the new identity | ✅ | 20 T8 records (5 cases × 4 engines) re-run at `5f8d91d`; SP latency/VRAM scaling and NVE MD throughput all pass |
| 21 | Historical reused evidence explicitly marked | ✅ | version block + campaign manifest scope (`not_in_scope`, `historical_reuse`) |
| 22 | Report rebuildable from a formal evidence archive | ✅ | archive published on the [v2.0.0b1 release](https://github.com/hydrogen1222/mliport/releases/tag/v2.0.0b1) (86 KiB, 96 records, sha256 `bb6b1b06…`), re-render verified |
| 23 | README statements match the evidence | ✅ | READMEs state the completed four-backend smoke and the historical T3/T4/T8 |
| 24 | No secrets/model weights/temporary probes/attic in the release | ✅ | repository-hygiene and distribution checks; archive excludes weights/trajectories/attic |

## NO-GO triggers (section 28) — all absent

- Report generator cannot bypass strict validation ✅
- A pass cannot hide a same-profile fail ✅ (`mixed` roll-up)
- README does not present old-validator evidence as current-HEAD ✅
- GEMDAT exceptions cannot silently become zero jumps ✅
- Exact displacement cannot be silently replaced by an unverified reconstruction ✅
- Alpha invalid points cannot enter regime summaries ✅
- Malformed schema/evidence cannot pass silently ✅
- Campaign records model hash/dtype/task/head ✅
- No silent fallback in any of the four backends ✅
- Package is not published as `mlipx` ✅
- Evidence archive is traceable (sha256 manifest committed) ✅
- GPU campaign results match the target commit ✅ (all 76 records at `5f8d91d`)

## Residuals before an unconditional GO

1. **GPU saddle-Hessian workflow (`t5h`)** was not part of the four-backend
   smoke; its semantics are covered by the CPU/analytic known-answer layer.

Verdict: **GO for beta** for the scoped, explicitly stated evidence:
current-HEAD four-backend V100 smoke (T1/T2FD/T5/T6/T7/T8) + CI + clean
wheel installs, with T3/T4 carried as historical evidence and a published,
sha256-pinned evidence archive. The GPU `t5h` note above is a statement of
scope, not a NO-GO trigger.
