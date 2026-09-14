# Migrating from `mlipx`

The project and package were renamed from **`mlipx`** to **`mliport`**. The
scientific code is the same lineage; the names, installation model and a few
defaults changed.

## Name changes

| Old | New |
|---|---|
| package / import `mlipx` | `mliport` |
| CLI `mlipx ...` | `mliport ...` |
| `mlipx` entry point in `pyproject.toml` | `mliport` |
| evidence schema prefix `mlipx.beta-validation-result/*` | `mliport.beta-validation-result/*` |

Historical evidence is **not deleted**: the validation evidence loader keeps
migrating legacy `mlipx.*` records and reports them as
`legacy_namespace_records` in the campaign metadata. Old validation results
therefore remain auditable; new records use the `mliport.*` schemas.

## Installation model

- Old: a single working environment could install the CLI and then call
  backend environments.
- New: install mliport **inside each backend's own environment**. Use
  `.venv-mace/bin/mliport`, `.venv-dpa/bin/mliport`,
  `.venv-grace/bin/mliport` or `.venv/bin/mliport` (UMA), or the generated
  `./bin/mliport-<engine>` launchers. See
  [installation.md](installation.md).

## Configuration changes

1. **Explicit backend.** There is no implicit UMA default any more. Add
   `--model-type` (or `engine = ...` in a `[model:...]` alias) to every run.
   Omitting it is a fatal configuration error.
2. **Strict configuration by default.** Misspelled keys and
   cross-backend options now fail instead of warning. If you truly need the
   legacy behaviour, opt in explicitly with `--lenient-config` or
   `[general] strict_config = false` in `settings.ini`.
3. **Model aliases must declare `engine`.** The old implicit-UMA alias no
   longer exists.
4. **Task defaults follow the backend.** UMA uses its task families
   (`omat`/`omol`/...); MACE/DPA/GRACE use the explicit `bulk`/`molecule`
   semantics.
5. **Environment variables:** `MLIPORT_SETTINGS` and `MLIPORT_JOBS_DIR`
   replace the old `mlipx`-named variables; the settings search paths are
   printed by `mliport config paths`.

## Python API

```python
# public entry point (backend-neutral)
from mliport import CalculatorFactory
```

The UMA class moved to `mliport.calculators.uma.UMACalculator`, next to the
MACE/DPA/GRACE wrappers. The old `mliport.calculator` shim was removed in
2.0.0b4; import from `mliport.calculators` or use `CalculatorFactory` and
select the backend explicitly. `mliport.__all__` never privileged UMA.

## Validation records

- Old records remain readable (legacy namespace migration).
- New campaigns use `mliport.beta-validation-result/2`,
  `mliport.beta-campaign/1` and `mliport.beta-archive-manifest/1`.
- Report paths are under `validation/science/reports/`; see
  [validation.md](validation.md).

## Checklist

- [ ] Replace `mlipx` with `mliport` in scripts/notebooks.
- [ ] Reinstall per backend (the installer is the supported path).
- [ ] Add an explicit backend/alias to every run.
- [ ] Fix any config keys that now fail strict validation.
- [ ] Re-run `mliport doctor` in each backend environment.
