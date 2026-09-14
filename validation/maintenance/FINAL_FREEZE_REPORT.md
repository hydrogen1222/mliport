# mliport final beta freeze audit

Baseline: `51d2a789e07d881ea349edc0f3db089e1ede9d03` (`v2.0.0b3` is the earlier
tag at `74f8dee`; the tag is not moved).
Tested runtime commit: `28f2a40` (the full suite and all three GitHub Actions
workflows ran on this revision).
Final commit: the commit that contains this report; the annotated tag
`v2.0.0b4` records it (a file cannot contain its own commit SHA).
Final tag: `v2.0.0b4`.

Round commits: `6a6448c` (queue backend neutrality), `3a74943` (state
isolation), `4d7b238` (device resolver), `89585f5` (dead/packaging cleanup),
`7318048` (architecture + package audit), `389d730` (MD template device),
`28f2a40` (documentation/provenance).

## Repository inventory

- 350 tracked files. The largest tracked file is the 844 KB
  `validation/science/data/omat24_subset.extxyz`; figures are 50-150 KB and
  source files at most ~100 KB. No tracked model, trajectory, archive,
  pickle or wheel artifact exists.
- The artifact scan (cache/results/models/runs/venvs, binary suffixes) found
  no tracked hits. The all-history scan still shows large legacy blobs from
  the upstream pre-rename history; they are public upstream content and
  rewriting history was out of scope.
- Secret scan: no API keys, tokens or private identity strings. The absolute
  path scan found one developer path in a test docstring; it is now fixed.
- `AGENTS.md` is the repository development contract, not a transient plan;
  the local task books remain untracked and are ignored by category.

## Legacy and dead code

- Removed now: `mliport/calculator.py` (the UMA-only compatibility shim) and
  the accidental top-level `examples` package. The example scripts moved to
  the repository `examples/` tree.
- Kept compatibility readers (old analysis records, `INCAR.uma`, job-record
  defaults, historical evidence module allowlists) are documented in
  `validation/maintenance/LEGACY_AUDIT.md`; none has an announced removal
  target because they parse data that still exists.
- `vulture` reports no high-confidence dead code; the only finding
  (`logger.py` unused `__exit__` variable) is fixed. The global F841/B017
  ruff ignores are now scoped to `tests/**`.
- `.gitignore` was rewritten around artifact categories (no per-file task
  book names, no root-wide `LGPS*`), and private LGPS structures moved to the
  ignored `local-data/` directory.

## User state

- Old behavior: jobs lived in the hidden `~/.mliport/jobs` directory with no
  inspection command and no schema/provenance fields.
- Same-HOME reinstall: previous records remain, which is the documented
  persistent-user-state policy; `mliport jobs` now prints the state path so a
  reinstall is not confused with a new user.
- Brand-new HOME: the CLI and TUI show no jobs (`No jobs found.`); the
  four-layer state reproduction confirmed no cross-HOME leak.
- New state path:
  `MLIPORT_JOBS_DIR` -> `MLIPORT_STATE_DIR` -> OS app-state directory ->
  `$XDG_STATE_HOME/mliport/jobs` -> `~/.local/state/mliport/jobs`.
- Records carry `schema_version`, `application`, `created_by_version`,
  `submit_cwd` and `project_root`; old records are upgraded in memory only.
- Migration: legacy `~/.mliport/jobs` is never imported silently;
  `mliport state migrate [--dry-run]` imports it additively and leaves the
  legacy files in place. `mliport clean --dry-run` lists terminal records
  without deleting active jobs or result directories.

## Queue backend neutrality

- `build_mliport_command` requires `model_type` as a keyword-only argument;
  task files fail closed without an explicit backend (or a NEB-resume
  checkpoint that supplies one).
- QN-01 (missing backend fails), QN-02/QN-03 (explicit MACE/DPA pass), QN-04
  (`fairchem` canonicalizes to UMA), QN-05 (NEB resume from checkpoint), and
  QN-06 (task/checkpoint conflict fails) are covered by tests.
- The TUI config screen no longer falls back to UMA when the backend select
  is empty, and its task default follows the selected backend.

## Multi-GPU

- MG-01..MG-07 and the MIG rejection tests use a synthetic four-GPU
  inventory: local ordinal resolution honours numeric and UUID
  `CUDA_VISIBLE_DEVICES` lists, rejects malformed/duplicate/out-of-range
  selectors, and both the CLI isolation value and the queue lease resolve the
  same UUID.
- On the single V100, DPA and GRACE CLI runs with `cuda:0` and no existing
  `CUDA_VISIBLE_DEVICES` printed the isolation re-exec note and completed
  with exit code 0.

## Architecture compatibility

| Arch | MACE | DPA | GRACE | UMA | Evidence |
|---|---|---|---|---|---|
| maxwell | experimental | experimental | experimental | experimental | binary audit (no hardware) |
| pascal_p100 | theory | theory | theory | theory | binary audit (no hardware) |
| pascal_p40 | theory | theory | theory | theory | binary audit (no hardware) |
| volta | verified (HW) | verified (HW) | verified (HW) | verified (HW) | real V100 acceptance |
| turing | theory | theory | theory | theory | binary audit (no hardware) |
| ampere_sm80 | theory | theory | theory | theory | binary audit (no hardware) |
| ampere_sm86 | theory | theory | theory | theory | binary audit (no hardware) |
| ada | theory | theory | theory | theory | binary audit (no hardware) |
| hopper | theory | theory | experimental | theory | binary audit (no hardware) |
| blackwell | theory | theory | experimental | theory | binary audit (no hardware) |

Binary evidence: `cuobjdump --list-elf` on the pinned wheels found
`sm_50/60/70/75/80/86/89/90` in torch cu126, `sm_70/75/80/86/89/90/100/120`
in cu128, `sm_60/70/80/89/90` plus compute_90 PTX in TensorFlow 2.20, and
`sm_50..sm_120` (including sm_60/sm_61) in deepmd-kit's
`libdeepmd_op_cuda.so`. MACE/UMA/GRACE ship no device code and inherit their
framework. Wheel hashes for the official torch cu126/cu128 and TF 2.20
artifacts are recorded in `validation/compatibility/`.

Real hardware evidence exists only for Volta: the fresh-install LGPS
acceptance and this round's four-backend SP regression. Everything else is
`binary_theory_verified` or `experimental`; the report and `mliport doctor`
never claim a hardware pass for untested families. Pascal P100 (sm_60) and
P40 (sm_61) are separated because P40 relies on the sm_60 minor-revision
path for torch/TF while DPA has an explicit sm_61 cubin. GRACE stays
experimental on Hopper/Blackwell. PyTorch stops publishing CUDA 12.6 wheels
from 2.15, so the installer pins remain mandatory for legacy GPUs and
Dependabot leaves the frameworks for manual review.

## Packaging

- Wheel `mliport-2.0.0b4-py3-none-any.whl` (c7fb10a96eadd59b7ecdc2c8cf21c3403580876597e65bfdc6c14b0bf37d0393) contains 101
  members: only `mliport/` plus its dist-info, including the packaged
  architecture audit JSON.
- Sdist `mliport-2.0.0b4.tar.gz` (694129041d5621a66e64abe77335be88982e69fde947b23ad45c91dcd1930856) contains 120 members under one
  root plus standard PEP 517 egg-info metadata.
- `twine check` passes for both; `scripts/check_package_contents.py` passes
  and rejects job state, results, caches, native binaries and unexpected
  top-level packages. CI runs the same gate.
- Fresh wheel installs on Python 3.10, 3.11 and 3.12 print
  `mliport 2.0.0b4`, start with an empty job history, and resolve the XDG
  state path correctly.

## V100 regression

Four backends, LGPS primitive, `--device cuda`:

| Backend | Result | Energy (eV/atom) |
|---|---|---|
| MACE | PASS | -4.322228 |
| DPA | PASS | -4.322031 |
| GRACE | PASS | -4.323116 |
| UMA | PASS | -4.326036 |

Queue regression: one MACE job (10/10 queue lifecycle steps PASS), one DPA
job (13/13), and one MACE job against the real HOME; `mliport jobs` shows
`User job history: <state-root>/jobs` and the completed
`lgs-sp` record. The full LGPS matrix was deliberately not re-run because no
workflow-path regression was introduced.

## CPU regression

MACE LGPS single point with `--device cpu` completes with exit code 0 and
`resolved_config.json` records `device: cpu`. The GPU environments were left
installed (all four report `torch 2.8.0+cu126`/`2.10.0+cu126` with CUDA
available; GRACE sees the V100).

## CI

GitHub Actions on tested runtime commit `28f2a40`:

| Workflow | Run | Result |
|---|---|---|
| lint | 34810783982 | success |
| tests | 34810784000 | success |
| package-build | 34810783981 | success |

Local suite on the release tree: 1358 passed, 5 skipped; ruff 0.5.1 check and
format --check pass; `generate_docs.py --check` passes. The annotated
`v2.0.0b4` tag push re-runs the same three workflows; the tag commit and
those run IDs are recorded in the GitHub release metadata rather than
self-referenced here.

## Remaining limitations

1. No Pascal, Turing, Ampere, Ada, Hopper or Blackwell hardware was available;
   their compatibility is binary-level theory, not runtime evidence.
2. The Hugging Face cold-download path was not network-validated on this
   host; `docs/models.md` documents the proxy recipe and the hash-verified
   local paths instead.
3. The all-history scan retains upstream pre-rename large blobs; no history
   rewrite was performed.
4. MIG instances are rejected by the device resolver instead of being
   scheduled as whole GPUs.
5. `pip install .` at the repository root remains intentionally unsupported
   and documented; the installable package is `mliport/`.
6. The beta remains a short-workflow acceptance; long equilibration and
   production MD accuracy are the user's scientific responsibility.

## Release recommendation

**Hard GO for `v2.0.0b4`** ("repository freeze / state isolation /
compatibility audit beta"). None of the hard NO-GO conditions applies:
backend neutrality is enforced, unknown GPU families fail closed, no untested
family is marked hardware-verified, no job state or calculation result ships
in the distribution, the V100 regression passes, CI is green, and the user
machine keeps GPU environments. After this tag, mliport takes only real
calculation bugs, install breakage, upstream breakage, security or
scientific-correctness fixes.
