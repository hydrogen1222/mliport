# LGPS fresh-install acceptance harness

This directory drives the user-facing mliport CLI through the LGPS acceptance
matrix from `mliport_lgps_fresh_install_full_acceptance_plan_20260914.md`.
It is test infrastructure, not part of the installed package.

## What it does

`lgps_smoke.py` runs documented CLI commands in each backend environment and
records the exit code, output files and parsed JSON artifacts. It never
re-implements mliport internals: a task passes only when the real command
succeeds and the documented outputs are complete.

Per-backend tasks (GPU):

| Task | Section | What runs |
|---|---|---|
| `g1_sp`, `g1_sp_repeat` | 30 | LGPS single point, then the same structure again |
| `g2_sp_perturbed` | 30 | single point after a 0.01 A Li displacement |
| `g3_opt_fixed`, `g4_opt_cell` | 30 | 2-step fixed-cell and 1-step cell+ionic relaxation |
| `g5_nve` | 30 | 5-step NVE |
| `g6_nvt_langevin`, `g7_nvt_bussi`, `g8_nvt_nhc` | 30 | 5-step NVT with each thermostat |
| `g9_md_restart` | 32 | continue from the saved positions+momenta |
| `g10_batch` | 33 | two-structure batch single points |
| `g11_incar` | 34 | `template sp` + edited INCAR through `mliport run` |
| `g12_negatives` | 35 | missing backend (CLI and INCAR), strict-config typo, wrong head, invalid task |
| `g13_api` | 36 | `mliport.api.calculate_energy` and CLI/API energy equality |
| `g14_tui` | 37 | `tui --help` and a headless launch that exits cleanly |
| `g15_queue` | 38 | `queue submit/start`, `jobs`, `queue status`, `queue stop` |
| `g16_neb` | 39 | 3-image CI-NEB and plain NEB, workflow smoke only |
| `g17_neb_restart` | 41 | stop on the step budget, resume, check the checkpoint sequence continues |
| `g18_analysis_core` | 43-48 | `analyze validate/thermo/msd` on the fresh NVE run |

Representative long-fixture tasks (run once, with the GRACE environment):

| Task | Section | What runs |
|---|---|---|
| `g19_analysis_long` | 43-56 | validate/thermo/rmsd/vacf/spectrum/rdf/density/transport on the local 1 ns LGPS run, plus a 30 ps GEMDAT segment |
| `g20_arrhenius` | 55 | transport at 600/700/800 K on same-engine local runs, then the Arrhenius fit |

CPU tasks (`--device cpu`): `g1_sp` and `g1_sp_repeat` only, as the task book
prescribes for the CPU budget.

## Commands

```bash
# full GPU matrix (four backends + representative analysis)
MLIPORT_ACCEPTANCE_PYTHON=.venv-analysis/bin/python \
    validation/acceptance/run_lgps_gpu.sh

# CPU smoke after the CPU install
MLIPORT_ACCEPTANCE_PYTHON=.venv-analysis/bin/python \
    validation/acceptance/run_lgps_cpu.sh

# one backend / one task while debugging
.venv-analysis/bin/python validation/acceptance/lgps_smoke.py \
    --backend mace --only g1_sp --list
```

Outputs go to `.validation-acceptance/<commit>/` (never into the package or
docs trees):

```text
.validation-acceptance/<commit>/
    environment.json            # collect_environment.py (sanitized)
    INSTALL_MATRIX.md           # installation matrix table
    gpu/<backend>/results.json  # one record per task
    gpu/<backend>/<task>/       # command cwd and raw outputs
    cpu/<backend>/results.json
```

## Status vocabulary

* `PASS` - the command exited as documented and the required files/values were
  present and finite.
* `FAIL` - a real defect or a missing required output.
* `EXPECTED_LIMITATION` - the program failed closed on purpose (for example
  `insufficient_diffusive_window` on a 5-step trajectory) or a documented
  acceptance budget was exceeded. The reason is recorded.
* `NOT_RUN` - the step was skipped, with a reason.

Scientific rules from the task book are preserved by construction: the four
backends never have absolute energies compared, short MD is never judged for
temperature convergence, NEB is marked `workflow_smoke_only` with
`physical_barrier=not_claimed`, and transport never fabricates a diffusion
coefficient.

## Fixtures

* `examples/structures/li10gep2s12_primitive.vasp` - the committed 50-atom
  Li10GeP2S12 primitive cell used by the README quickstart and acceptance.
  Its provenance is documented in `examples/structures/README.md`.
* Synthetic NEB endpoints are generated deterministically from that structure
  (one Li displaced away from its nearest Li, at least 1.8 A from every other
  atom) and are explicitly marked workflow-only.
* The long analysis tasks use local, git-ignored LGPS trajectories under
  `results/` that predate this round (`results/multi-grace-800K-1ns` and the
  600/700/800 K UMA runs). The acceptance logs record their provenance; no
  long trajectory is regenerated.

## CI

GitHub Actions runs `tests/test_acceptance_harness.py`, which checks fixture
integrity, task registration, JSON parsing, INCAR key editing, deterministic
endpoint construction, schema validity and that README/docs commands marked
`tested example` refer to existing CLI subcommands. Real GPU/CPU runs are not
executed in CI.
