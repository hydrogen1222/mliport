# Development

This page is for people changing mliport itself. If you only want to run it, the
[installation guide](installation.md) is what you need.

## Two kinds of environment

mliport keeps a **core development environment** and separate **backend runtime
environments**, and they are not interchangeable.

The core environment runs unit and integration tests, lint and the docs build. The
repository root is tooling configuration only, so `uv sync` there installs nothing
useful (no numpy, no pytest) and certainly no MLIP runtime. Build the environment
from the package metadata, the way CI does:

```bash
uv venv .venv-dev
uv pip install --python .venv-dev/bin/python -e './mliport[dev]'
uv pip install --python .venv-dev/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/ruff check mliport/mliport tests scripts
.venv-dev/bin/ruff format --check mliport/mliport tests scripts
python scripts/check_repository_hygiene.py
```

This environment proves nothing about UMA, MACE, DPA or GRACE. The core tests use
synthetic calculators for those paths. Checks against a real backend need the
environments the installer builds:

```bash
./scripts/install_mliport.sh --engines mace dpa grace uma --device auto
.venv-mace/bin/mliport doctor --engine mace --device auto
```

The full GPU campaigns live under `validation/` and use those environments; see
[Validation](validation.md).

## Where things are

```text
mliport/mliport/           package source
  calculators/             backend adapters and the factory
  config/                  resolver, schema, defaults, settings
  runners/                 SP, optimization, MD, NEB and batch execution
  confsearch/              MD-based configuration search (adsorbate mode)
  analysis/                trajectory analysis
  tui/                     Textual UI
  install/                 installer plan, compatibility registry, launchers
tests/mliport/             unit and integration tests
validation/science/        campaigns, evidence, reports, model manifest
docs/                      this documentation (English and Chinese)
scripts/                   repository tooling: installer, docs generation, hygiene
```

The installable package is `mliport/`, with its own `mliport/pyproject.toml`. The
`pyproject.toml` at the repository root is developer tooling (ruff, pytest, uv), so
installing the repository root is deliberately unsupported: setuptools sees the
repository layout and stops with a package-discovery error. Use
`./scripts/install_mliport.sh`, or point a hand-managed environment at the
`mliport/` subdirectory.

## Writing the documentation

The site is built with MkDocs Material and published to GitHub Pages from `main`.
Every page exists twice: `page.md` is English and `page.zh.md` is its Chinese twin.
A tool will not translate for you, so write both by hand, in the same structure.

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

A few rules keep the two languages honest, and the tests enforce them:

- A translated page has the same number of `##` sections, code blocks and links as
  its English page, and the same generated blocks.
- Links between pages name the English file (`installation.md`, never
  `installation.zh.md`); the site resolves the language. A link that leaves `docs/`
  has to be a full GitHub URL.
- New pages go into the `nav` of `mkdocs.yml` and into the page list in the docs
  contract test.

Some facts live in code, and the pages show them in generated blocks:

```markdown
<!-- BEGIN GENERATED: backend-matrix -->
...never edited by hand...
<!-- END GENERATED -->
```

The compatibility registry (`mliport/install/compatibility.py`) owns environment
names, Python ranges, framework pins and GPU routes; the model manifest owns the
pinned profiles. To change one, change the source and regenerate:

```bash
python scripts/generate_docs.py          # refresh blocks and mliport/README.md
python scripts/generate_docs.py --check  # what CI runs
```

## Tests

- Run `.venv-dev/bin/python -m pytest -q` before you push; everything should be
  green.
- Core code supports Python 3.10 to 3.13. Backends are narrower; see
  [Installation](installation.md).
- The docs contract tests check links and anchors, run the documented CLI commands
  with `--help`, compile the Python snippets and compare generated blocks with their
  sources.
- Every change starts with a failing test, then the fix.

## Building the package

```bash
uv build mliport              # sdist and wheel
```

Check the metadata (README, citation, package name), and that no `mlipx` namespace
or transient planning document ended up in the wheel.

## Commits and pull requests

- One logical change per pull request.
- The commit message says what was failing, what fixed it, and the **data impact**:
  either "no scientific record invalidated", or which evidence was regenerated and
  why.
- CI must be green: tests on Python 3.10 to 3.13, lint, the package build and the
  docs build.
- Do not loosen a threshold or delete an honest `not_run` record to make a report
  look better.
