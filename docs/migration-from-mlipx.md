# Migrating from mlipx

The project was renamed from **mlipx** to **mliport**. It is the same lineage of
scientific code; what changed is the names, the way it is installed, and a few
defaults that used to guess.

## What was renamed

| Before | Now |
|---|---|
| package and import `mlipx` | `mliport` |
| command `mlipx ...` | `mliport ...` |
| `mlipx` entry point in `pyproject.toml` | `mliport` |
| evidence schema prefix `mlipx.beta-validation-result/*` | `mliport.beta-validation-result/*` |

Old evidence is not thrown away. The validation loader still reads legacy
`mlipx.*` records and lists them as `legacy_namespace_records` in the campaign
metadata, so earlier results stay auditable while new records use the `mliport.*`
schemas.

## Installation

Before, one working environment could install the CLI and then call into the
backend environments. Now mliport is installed **inside each backend's own
environment**, and you call that environment's command: `.venv-mace/bin/mliport`,
`.venv-dpa/bin/mliport`, `.venv-grace/bin/mliport` or `.venv/bin/mliport` for UMA,
or the generated `./bin/mliport-<engine>` launchers. See
[Installation](installation.md).

## Configuration

1. **The backend is explicit.** There is no implicit UMA default any more. Add
   `--model-type` (or `engine = ...` in a `[model:...]` alias) to every run; leaving
   it out is a fatal configuration error.
2. **Configuration is strict.** Misspelled keys and options meant for another
   backend now fail instead of warning. If you need the old behavior, ask for it
   with `--lenient-config` or `[general] strict_config = false` in `settings.ini`.
3. **Model aliases must declare `engine`.** The old implicit-UMA alias is gone.
4. **Task defaults follow the backend.** UMA uses its own task families
   (`omat`, `omol`, ...); MACE, DPA and GRACE use the explicit `bulk` and
   `molecule` meanings.
5. **Environment variables.** `MLIPORT_SETTINGS` and `MLIPORT_JOBS_DIR` replace
   the `mlipx`-named ones. `mliport config paths` prints where settings are
   searched.

## Python API

```python
# the backend-neutral entry point
from mliport import CalculatorFactory
```

The UMA class now lives in `mliport.calculators.uma.UMACalculator`, next to the
MACE, DPA and GRACE wrappers. The old `mliport.calculator` shim was removed in
2.0.0b4: import from `mliport.calculators`, or use `CalculatorFactory` and choose
the backend explicitly. `mliport.__all__` never singled out UMA.

## Validation records

Old records stay readable. New campaigns use `mliport.beta-validation-result/2`,
`mliport.beta-campaign/1` and `mliport.beta-archive-manifest/1`, and the reports
sit under `validation/science/reports/`; see [Validation](validation.md).

## Checklist

- [ ] Replace `mlipx` with `mliport` in scripts and notebooks.
- [ ] Reinstall per backend (the installer is the supported route).
- [ ] Give every run an explicit backend or alias.
- [ ] Fix any config keys that now fail strict validation.
- [ ] Run `mliport doctor` again in each backend environment.
