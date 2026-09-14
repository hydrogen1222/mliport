# Development

## Two kinds of environments

mliport has a **core development environment** and separate **backend runtime
environments**. They are not interchangeable.

### Core development environment

Used for unit/integration tests, lint, docs, and synthetic-calculator tests:

```bash
uv sync                     # core dev environment (pytest, ruff, build tools)
.venv/bin/python -m pytest -q
.venv/bin/ruff check mliport/mliport tests scripts
.venv/bin/ruff format --check mliport/mliport tests scripts
```

`uv sync` does **not** install UMA, MACE, DPA or GRACE runtimes, and it does
not prove that any backend works. Core tests use synthetic/fake calculators
for those paths.

### Backend runtime environments

Real backend smoke tests need the installer-built environments:

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
.venv-mace/bin/mliport doctor --engine mace --device auto
```

Full GPU campaigns live under `validation/` and use those environments; see
[validation.md](validation.md) and `validation/science/README.md`.

## Repository layout

```text
mliport/mliport/           package source
  calculators/             backend adapters + factory
  config/                  resolver, schema, defaults, settings
  runners/                 SP/OPT/MD/NEB/batch execution
  analysis/                trajectory analysis
  tui/                     Textual UI
  install/                 installer plan, compatibility registry, docs, launchers
tests/mliport/             unit/integration tests
validation/science/        campaigns, evidence, reports, model manifest
docs/                      user documentation (generated + curated)
scripts/                   repo tooling (installer, docs generation, hygiene)
```

The installable package is `mliport/` (with `mliport/pyproject.toml`). The
repository-root `pyproject.toml` is developer tooling only (ruff, pytest,
uv), so `pip install .` at the repository root is intentionally unsupported:
setuptools sees the repository layout and stops with a package-discovery
error. Use `./scripts/install_mliport.sh`; if you manage an environment by
hand, install the `mliport/` subdirectory explicitly rather than the
repository root.

## Generated artifacts

Never hand-edit generated files; regenerate and commit:

```bash
python scripts/generate_docs.py          # docs/installation.md, docs/backends/*, mliport/README.md
python scripts/generate_docs.py --check  # CI check
```

The compatibility registry (`mliport/install/compatibility.py`) is the single
source for environment names, Python ranges, framework pins and GPU routes.
Model profiles come from `validation/science/model_manifest.json`.

## Tests

- Unit/integration: `.venv/bin/python -m pytest -q` (all green before push).
- Supported core Pythons: 3.10, 3.11, 3.12 (backend constraints differ; see
  [installation.md](installation.md)).
- Docs contract: link checking, CLI smoke tests and generated-block sync are
  part of the test suite.
- Every change ships with a red test first, then the fix.
- Repository hygiene: `python scripts/check_repository_hygiene.py`.

## Package build

```bash
uv build mliport              # sdist + wheel
```

Check the metadata (README, CITATION, package name) and that no `mlipx`
namespace or transient plan documents enter the wheel.

## Commit / PR expectations

- One logical change per PR, with a red test first.
- Commit message states RED / FIX / GREEN and a **DATA IMPACT** section
  (e.g. "no scientific record invalidated", or which evidence was
  regenerated and why).
- CI must be green (tests on 3.10/3.11/3.12, lint, package build).
- Do not loosen thresholds or delete honest `not_run` records to make a
  report look better.
