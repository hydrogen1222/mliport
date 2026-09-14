# mliport LGPS fresh-install acceptance report

Baseline commit: `74f8dee0cfa009b6dc08d888c48a590078f69836` (2.0.0b3)
Acceptance code: `fad5ed6` (fixes + harness + docs in this round)
Acceptance host: Rocky Linux 9.8, dual Xeon E5-2696 v3, NVIDIA
V100-SXM2-16GB (16 GiB, driver 580.173.02), Python 3.12.13 environments.
Report date: 2026-09-14.

This report records the execution of
`mliport_lgps_fresh_install_full_acceptance_plan_20260914.md`. It is written
for a reader who wants to know what a clean installation actually does on a
V100 workstation, which failures were found, and what remains limited.

## Summary

- The documented one-command install (`./scripts/install_mliport.sh --engines
  mace dpa grace uma --device auto`) completes from a clean repository with
  **zero manual pip/site-packages hotfixes**.
- All four GPU backends pass doctor, model load and an LGPS single point.
- The workflow matrix (single point, perturbed single point, relaxation, MD
  NVE/NVT with three thermostats, MD continuation, batch, INCAR run,
  negatives, Python API, TUI, queue, NEB/CI-NEB and analysis core) passes on
  all four backends except the explicitly recorded limitations below.
- The CPU route installs and runs the minimal four-backend LGPS smoke, and a
  pure-CPU analysis environment consumes GPU-produced trajectories.
- Ten product defects and several documentation gaps were found by the
  acceptance run and fixed before the final clean-room pass.

## Git and environment baseline

```text
BASELINE_COMMIT=74f8dee0cfa009b6dc08d888c48a590078f69836
BASELINE_TAG=v2.0.0b3
HEAD_AT_ACCEPTANCE=74f8dee
OS=Rocky Linux 9.8 (kernel 5.14)
GPU=Tesla V100-SXM2-16GB, CC 7.0, driver 580.173.02
CPU=Intel Xeon E5-2696 v3 (2 sockets)
```

Environment names, Python versions and package versions are in
`environment.json` for both phases. Caches (uv/pip) were preserved: this is a
fresh **environment** installation, not a cold-network download test.

## LGPS fixtures

| Item | Value |
|---|---|
| Acceptance structure | `examples/structures/li10gep2s12_primitive.vasp` (50 atoms, Li20Ge2P4S24, min distance 2.02 A) |
| Provenance | derived from a 2x2x2 LGPS supercell (`LGPS.vasp`); tiling reproduces the source within 3e-8 |
| Commit fixture SHA-256 | `0e0702be3a31778839f730430c6d7989118df316c4976a1a98f945c32fd3561a` |
| NEB endpoints | synthetic one-Li displacement away from its nearest Li; workflow-only, no barrier claim |
| Long analysis fixture | local, git-ignored 1 ns LGPS GRACE run (10001 frames, 400 atoms, 800 K) |
| Multi-temperature fixture | local same-engine UMA LGPS runs at 600/700/800 K (400 ps each) |

No Cu or Lennard-Jones substitute was used, and no long trajectory was
regenerated. The local long runs predate this round; their provenance is in
the machine-readable `fixtures/selection.json`.

## Fresh installation

### GPU route

The documented command was executed from a state with only
`.venv-analysis` (the acceptance harness environment) present. Attempt
history:

| Attempt | Command | Outcome |
|---|---|---|
| 1 | README command | Failed: bootstrap planner Python had no `packaging` (fixed) |
| 2 | README command | Failed: space-separated `--engines` rejected (fixed) |
| 3 | README command | Installed all four environments, then the log driver was killed; analysis extras were missing (fixed) |
| 4 | README command (`setsid`) | All 27 steps completed, launchers created, four doctor checks pass |
| DPA | `--engines dpa --clean` after the `mpich` fix | All 8 steps completed |
| Final | README command from a clean state | All 27 steps completed; final log tail: |

```text
[mliport] [launcher] created <repo>/bin/mliport-grace
[mliport] [launcher] created <repo>/bin/mliport-uma
[mliport] All 27 steps completed.
```

### CPU route

`./scripts/install_mliport.sh --engines mace dpa grace uma --device cpu`
completed all 27 steps. The four environments run one LGPS single point and
one repeated single point; the CPU doctor checks pass with `--device cpu`.

### Installation matrix

GPU:

| Platform | Backend | Python | Framework | Device | Install | Doctor | Model load | LGPS SP |
|---|---|---|---|---|---|---|---|---|
| V100 | MACE | 3.12.13 | torch 2.8.0+cu126 | cuda | PASS | PASS | PASS | PASS |
| V100 | DPA | 3.12.13 | torch 2.10.0+cu126 | cuda | PASS | PASS | PASS | PASS |
| V100 | GRACE | 3.12.13 | tensorflow 2.20.0 | cuda | PASS | PASS | PASS | PASS |
| V100 | UMA | 3.12.13 | torch 2.8.0+cu126 | cuda | PASS | PASS | PASS | PASS |

CPU:

| Platform | Backend | Python | Framework | Device | Install | Doctor | Model load | LGPS SP |
|---|---|---|---|---|---|---|---|---|
| CPU | MACE | 3.12.13 | torch 2.8.0+cpu | cpu | PASS | PASS | PASS | PASS |
| CPU | DPA | 3.12.13 | torch 2.10.0+cpu | cpu | PASS | PASS | PASS | PASS |
| CPU | GRACE | 3.12.13 | tensorflow 2.20.0 | cpu | PASS | PASS | PASS | PASS |
| CPU | UMA | 3.12.13 | torch 2.8.0+cpu | cpu | PASS | PASS | PASS | PASS |

### Environment versions

GPU:

| Backend | Python | mliport | Backend package | Framework | Features |
|---|---|---|---|---|---|
| MACE | 3.12.13 | 2.0.0b3 | mace-torch 0.3.16 | torch 2.8.0+cu126 | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |
| DPA | 3.12.13 | 2.0.0b3 | deepmd-kit 3.1.3 | torch 2.10.0+cu126 | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |
| GRACE | 3.12.13 | 2.0.0b3 | tensorpotential 0.6.0 | tensorflow 2.20.0 | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |
| UMA | 3.12.13 | 2.0.0b3 | fairchem-core 2.21.0 | torch 2.8.0+cu126 | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |

CPU:

| Backend | Python | mliport | Backend package | Framework | Features |
|---|---|---|---|---|---|
| MACE | 3.12.13 | 2.0.0b3 | mace-torch 0.3.16 | torch 2.8.0+cpu | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |
| DPA | 3.12.13 | 2.0.0b3 | deepmd-kit 3.1.3 | torch 2.10.0+cpu | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |
| GRACE | 3.12.13 | 2.0.0b3 | tensorpotential 0.6.0 | tensorflow 2.20.0 | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |
| UMA | 3.12.13 | 2.0.0b3 | fairchem-core 2.21.0 | torch 2.8.0+cpu | tui:ok, analysis:ok, transport:ok, electrolyte:ok, plotting:ok |

### Manual hotfix accounting

| Item | Count |
|---|---|
| Manual `pip install` in the documented path | 0 |
| Edits inside `site-packages` | 0 |
| Hidden environment variables required for install | 0 (DPA/GRACE isolation is automatic in the CLI) |
| Model files copied by hand into package dirs | 0 |

## GPU workflow acceptance

| Backend | Tasks | PASS | FAIL | Expected limitations | Not run |
|---|---|---|---|---|---|
| MACE | 42 | 41 | 0 | 1 | 0 |
| DPA | 45 | 44 | 0 | 1 | 0 |
| GRACE | 48 | 47 | 0 | 1 | 0 |
| UMA | 44 | 43 | 0 | 1 | 0 |

Failures and expected limitations:

| Backend | Task | Status | Detail |
|---|---|---|---|
| MACE | `g17_neb_restart_budget_override` | EXPECTED_LIMITATION | beta limitation: the resume fingerprint includes step budgets, so a user-supplied --max-steps override is rejected; resume continues with the recorded budget. |
| DPA | `g17_neb_restart_budget_override` | EXPECTED_LIMITATION | beta limitation: the resume fingerprint includes step budgets, so a user-supplied --max-steps override is rejected; resume continues with the recorded budget. |
| GRACE | `g17_neb_restart_budget_override` | EXPECTED_LIMITATION | beta limitation: the resume fingerprint includes step budgets, so a user-supplied --max-steps override is rejected; resume continues with the recorded budget. |
| UMA | `g17_neb_restart_budget_override` | EXPECTED_LIMITATION | beta limitation: the resume fingerprint includes step budgets, so a user-supplied --max-steps override is rejected; resume continues with the recorded budget. |

The harness records `PASS`, `FAIL`, `EXPECTED_LIMITATION` and `NOT_RUN` with a
reason for every task. NEB runs are recorded as `workflow_smoke_only` with
`saddle_validation=not_performed` and `physical_barrier=not_claimed`.

## Analysis acceptance

Core `validate` / `thermo` / `msd` run on every backend's fresh NVE
trajectory. The representative long-fixture suite and the same-engine
Arrhenius pipeline:

| Analysis task | Status | Key result |
|---|---|---|
| `g19_analysis_validate` | PASS | artifacts: provenance.json, request.json, results.json |
| `g19_analysis_thermo` | PASS | artifacts: provenance.json, request.json, results.json, thermo.csv |
| `g19_analysis_rmsd` | PASS | artifacts: provenance.json, request.json, results.json, rmsd.csv |
| `g19_analysis_vacf` | PASS | artifacts: provenance.json, request.json, results.json, vacf.csv |
| `g19_analysis_spectrum` | PASS | artifacts: provenance.json, request.json, results.json, spectrum.csv |
| `g19_analysis_rdf` | PASS | artifacts: provenance.json, rdf.csv, rdf.png, rdf.svg |
| `g19_analysis_density` | PASS | artifacts: density.npz, provenance.json, request.json, results.json |
| `g19_analysis_transport` | PASS | D=1.885096858330463e-09 m^2/s, 95% CI=[1.4923284163456254e-09, 2.275022374278964e-09], T=797.0238474614208 K |
| `g19_analysis_electrolyte_segment` | PASS | artifacts: density_projection.png, density_projection.svg, detected_sites.cif, diagnostics.json |
| `g19_analysis_electrolyte_full_run` | EXPECTED_LIMITATION | GEMDAT site/jump analysis on the full 1 ns, 400-atom run exceeded the acceptance time budget (>300 s); the 30 ps segment above completed. The full trajectory re |
| `g20_transport_600K` | PASS | D=8.475717920885925e-10 m^2/s, T=601.819571188524 K |
| `g20_transport_700K` | PASS | D=3.479544924221434e-09 m^2/s, T=699.205068855085 K |
| `g20_transport_800K` | PASS | D=4.1032232256356064e-09 m^2/s, T=799.6301902531942 K |
| `g20_arrhenius` | PASS | Ea=0.316231184481365 eV, r2=0.8760299367415956, points=3 |

GEMDAT site analysis completes on a 30 ps segment of the 1 ns fixture; the
full 1 ns, 400-atom site discovery exceeds the short acceptance budget and is
recorded as an expected limitation. Transport on the 1 ns run returns
`D = 1.885e-09 m^2/s` (95% credible interval `[1.492e-09, 2.275e-09] m^2/s`,
fit window 100-200 ps) with an explicit sparse lag grid. The Arrhenius fit
uses three same-engine UMA temperatures produced by this run (not fabricated
values).

CPU analysis of a GPU artifact: `validate`, `thermo` and `msd` succeed in the
pure-CPU `.venv-mace` environment on a copied GPU-produced NVE run.

## Bugs found and fixes

| ID | Severity | Area | Problem | Fix |
|---|---|---|---|---|
| INSTALL-001 | P0 (install impossible) | installer bootstrap | uv-managed Python lacked `packaging`; the documented install failed on a machine with system Python 3.14 | planner runs through `uv run --with packaging` when needed; regression test |
| INSTALL-002 | P1 | installer CLI | README's space-separated `--engines` rejected | `nargs="+"` plus comma/space normalisation; regression test |
| INSTALL-003 | P1 | installer plan | analysis/transport/electrolyte extras were not installed | every editable install now uses `mliport[analysis-all]`; plan test |
| INSTALL-004 | P1 | DPA/GRACE CLI | backends refused to run without process-level CUDA isolation | CLI restarts the command with a single visible GPU; tests |
| INSTALL-005 | P1 | DPA runtime | `deepmd-kit` imports MPICH metadata from its `torch` extra | DPA environment installs `mpich>=5.0,<6`; plan test |
| CLI-001 | P2 | CLI | `mliport --version` did not exist | added; test |
| DOCTOR-001 | P2 | doctor | no machine-readable output or feature inventory | `doctor --json` and core/tui/analysis/transport/electrolyte/plotting rows; tests |
| TUI-001 | P2 | TUI | non-TTY launch waited for a terminal | fails closed with a clear message; pilot smoke; tests |
| ANALYSIS-001 | P1 | arrhenius | shared trajectory-source options crashed the fit | source-only options are ignored and recorded; regression test |
| NEB-001 | P1 | NEB resume | UMA resume failed the fingerprint because the checkpoint materialises implicit charge/spin defaults | resume validates against the recorded electronic state; regression test |

Harness-side fixes (not product defects): API/CLI tolerance for UMA float32
(numeric equivalence at 1e-5 eV total, 4e-8 eV/atom recorded), queue job
identification through the shared job registry, NEB output-directory hygiene,
and a CPU/GPU output split.

## Dependency changes

- `mpich>=5.0,<6` is now installed in the DPA environment. Reason:
  `deepmd-kit==3.1.3` declares `mpich` in its `torch` extra, and
  `deepmd.pt.cxx_op` reads its distribution metadata at model load.
- No other pins changed. The docs add a SOCKS-proxy remedy
  (`httpx[socks]`) for Hugging Face downloads; this is a documentation fix,
  not an installer dependency.

## Documentation rewrite

The README was rewritten around the real acceptance path: direct description,
uv prerequisite, verified GPU and CPU install commands, environment launcher
table, doctor expectations, LGPS first run, four backend examples in
MACE/DPA/GRACE/UMA order, workflow map, analysis map, strict-configuration
notes, short validation status and upstream model links. The Chinese README is
a natural technical adaptation, not a mechanical translation. `docs/` gained
the acceptance-verified options for installation, models, NEB, MD, queue and
the analysis tasks, plus troubleshooting entries for every failure above.

The Humanizer skill is not available in this session (skill search and local
`SKILL.md` lookup found nothing), so the plan's own style rules were applied
directly: no `not-X-but-Y` constructions, staged openers, dramatic closers,
forced triads, bold-label lists or inflated adjectives; commands, model names,
scientific values and citations were kept unchanged. A second technical audit
checked the rewritten pages against the CLI, the compatibility registry, the
model manifest and the acceptance results; the docs contract, README sync and
command-existence tests pass.

## Tests, lint and CI

```text
pytest: 1303 passed, 4 skipped
docs contract: pass
README sync: pass
acceptance harness tests: pass
lint/format: pass
wheel/sdist build: pass
GitHub Actions: pending final push
```

## Remaining limitations

1. The GPU acceptance is a short-workflow smoke by design: MD runs use 5
   steps, NEB uses three images and is explicitly experimental, and the
   transport/Arrhenius numbers come from acceptance-scale trajectories.
2. GEMDAT site discovery on the full 1 ns, 400-atom LGPS run exceeds the
   short acceptance budget; the report uses a 30 ps segment.
3. NEB resume keeps the recorded step budget in this beta; changing
   `--max-steps` on resume is rejected with an explicit fingerprint error.
4. Hugging Face downloads could not be verified from this host because of its
   local SOCKS proxy configuration; the MACE GitHub release download was
   verified (SHA-256 matches the manifest) and all runs used local,
   manifest-verified checkpoints.
5. The preclean `pip freeze` capture failed because uv-managed environments do
   not ship `pip`; the forensic record keeps Python versions, key distribution
   versions, sizes and doctor outputs for the removed environments instead.
6. Cache warmth was preserved. This is a fresh-environment acceptance, not a
   measurement of cold-network install time.
7. UMA, GRACE and DPA checkpoints keep their upstream licenses; mliport does
   not redistribute model files.

## Recommendation

**GO for the user-facing beta**, conditional on the final GitHub Actions run
staying green. The documented install, the four-backend LGPS workflows and the
analysis modules were exercised on the target hardware; every failure found
during the round was fixed in the repository and locked with a regression
test.
