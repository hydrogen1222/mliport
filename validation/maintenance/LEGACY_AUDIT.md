# Legacy and dead-code audit

Scope: every tracked file matched by the task-book scan

```bash
git grep -niE 'deprecated|legacy|compatib|pre-rename|old path|old api|remove after|temporary|workaround|hack|todo|fixme|xxx'
```

The raw scan returns 411 lines across source, tests, docs and evidence. Most
are ordinary English uses of "temporary" or descriptions of on-disk formats,
so the findings below group the hits by symbol or file and classify each
group. Nothing was removed without checking source, tests, docs, public
imports and dynamic lookup paths.

## Removed now

| Location | Finding | Evidence | Action |
|---|---|---|---|
| `mliport/mliport/calculator.py` | Deprecated shim exporting `UMACalculator` for pre-rename imports | Only tests and the migration guide imported it; `mliport.calculators` and `CalculatorFactory` are the supported paths; `mliport.__all__` never exposed it | Deleted in 2.0.0b4; tests now import `mliport.calculators.uma`; `docs/migration-from-mlipx.md` records the removal |
| `mliport/examples/` Python package | Top-level `examples` package shipped into `site-packages` | Example scripts were developer-only; the repository already has a user-facing `examples/` directory; packaging included `examples`/`examples.*` | Scripts moved to `examples/`; package and `__init__.py` removed; package discovery now `mliport`, `mliport.*` |

## Kept: format and schema compatibility

These parse older evidence or user files and are covered by tests or
known-answer records. Removing them would break historical runs without a
replacement.

| Location | Reason |
|---|---|
| `mliport/analysis/dataset.py` (`positions` wrapped/unwrapped parsing) | Reads external trajectories and old records; validation knows the convention |
| `mliport/analysis/transport.py` legacy-path note | Documents why reconstructed frames are reported as `adapter=false` |
| `mliport/config/incar.py`, `cli.py` (`INCAR.uma` fallback) | Historical default filename; fail-closed when absent |
| `mliport/jobs.py` record defaults (`display_name`, `run_id`, claim fields, state schema) | Read-only upgrade of pre-2.0.0b4 job records |
| `validation/science/evidence/constants.py` (`mlipx.*`, `mliport.calculator` allowlist) | Historical evidence records reference these module paths; they are strings in JSON, not imports |
| `mliport/calculators/dpa_calc.py` (`.pb` legacy TensorFlow runtime error, `device` key) | Explicit error message for an unsupported backend format |
| `mliport/calculators/mace_calc.py` (`gpu` synonym) | Documented CLI synonym, resolved centrally |
| `mliport/config/schema.py` ("legacy canonical field name") | Renamed INCAR key kept readable for old inputs |
| `docs/migration-from-mlipx.md`, `docs/configuration.md`, `CHANGELOG.md` | Change documents; history belongs there |

## False positives

- `CODE_OF_CONDUCT.md` uses "temporary or permanent repercussions" in its
  standard text.
- Comments that describe atomic-write temporary files (`analysis/runner.py`,
  `analysis/transport.py`) use "temporary" for a real filesystem pattern.
- `docs/workflows/queue.md`, `docs/installation.md` and `docs/models.md`
  describe "legacy wheels" or historical behaviour as documentation, not as
  code paths to remove.
- `mliport/mliport/install/plan.py` and `tests/mliport/mliport/test_install_plan.py`
  match the scan word "plan" because they are the installer planner and its
  tests.

## Deprecated with a deadline

None. No remaining shim has a removal target; the compatibility paths above
are either data-format readers or carry explicit fail-closed errors. The
`mliport.calculator` shim was the only deprecation note and is now removed.

## Static dead-code scan

`uvx vulture mliport/mliport scripts --min-confidence 90` reported one item,
`logger.py:221 unused variable 'exc_value'`; the `__exit__` signature now uses
`_exc_value`. Vulture otherwise produced no high-confidence candidates, which
matches the deliberate use of lazy imports, argparse callbacks, dynamic
backend imports and Textual handlers in this codebase.
