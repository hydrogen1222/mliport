# Contributing to mliport

Thanks for helping improve mliport. Please open an issue or a pull request
against `main`; this document describes the real development commands and the
environment model.

## Two kinds of environments

mliport has a **core development environment** and separate **backend
runtime environments** (MACE, DPA, GRACE, UMA). They are not
interchangeable:

- The repository root is tooling configuration only: `uv sync` there installs
  nothing (no numpy, no pytest) and does not install any MLIP runtime, in
  particular not UMA. The **core development environment** for unit tests,
  lint, docs and synthetic calculators is built from the package metadata, as
  CI does.
- Real backend checks need the installer-built environments; see
  [docs/installation.md](docs/installation.md).

```bash
# core development environment (what CI runs)
uv venv .venv-dev
uv pip install --python .venv-dev/bin/python -e './mliport[dev]'
uv pip install --python .venv-dev/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv-dev/bin/python -m pytest -q
.venv-dev/bin/ruff check mliport/mliport tests scripts
.venv-dev/bin/ruff format --check mliport/mliport tests scripts
python scripts/check_repository_hygiene.py

# alternative on a machine that already has a backend environment:
# .venv-mace/bin/python -m pip install -e './mliport[dev]'  (or uv pip install --python ...)
# .venv-mace/bin/python -m pytest tests/mliport/mliport/neb -q

# package build
uv build mliport

# backend runtime (one per engine, installer-managed)
./scripts/install_mliport.sh --engines mace --device auto
.venv-mace/bin/mliport doctor --engine mace --device auto
```

## Package layout

The installable package lives in `mliport/`; the repository-root
`pyproject.toml` is tooling only (ruff, pytest, uv). `pip install .` at the
repository root therefore fails fast with a setuptools package-discovery
error; install through `./scripts/install_mliport.sh`, or point a manual
environment manager at the `mliport/` subdirectory explicitly.

## Pull requests

1. Fork and branch from `main`.
2. Add a **red test first**, then the smallest fix that makes it pass.
3. Update the docs in both languages (see Documentation above).
4. Keep the test suite green and run the lint/format commands above.
5. State a **DATA IMPACT** section in the commit message: which scientific
   records are invalidated/regenerated, or "no scientific record
   invalidated".
6. Do not loosen validation thresholds, delete honest `not_run` records, or
   route around fail-closed checks to make output look better.

## Documentation

The documentation site (MkDocs Material, published to GitHub Pages from `main`) is
bilingual. Every page in `docs/` exists twice: `page.md` is the English original and
`page.zh.md` its Chinese twin. Write both by hand, with the same sections, code blocks
and links; the docs contract tests compare their structure.

```bash
pip install -r requirements-docs.txt
mkdocs serve                       # live preview at http://127.0.0.1:8000
python scripts/generate_docs.py    # refresh the generated blocks
```

- Link to the English file (`installation.md`) from both languages; a link that leaves
  `docs/` must be a full GitHub URL.
- A new page goes into `nav` in `mkdocs.yml` and into the page list in
  `tests/mliport/mliport/test_docs_contract.py`.
- Never hand-edit a generated block (`<!-- BEGIN GENERATED: ... -->`). The blocks in
  `docs/installation*.md`, `docs/backends/*.md` and `docs/workflows/neb*.md`, and the
  whole of `mliport/README.md`, come from `scripts/generate_docs.py`.
- The compatibility registry (`mliport/mliport/install/compatibility.py`) owns
  environment names, Python ranges, framework pins and GPU routes;
  `validation/science/model_manifest.json` owns pinned model profiles. Changing a pin
  means updating the registry and regenerating, not editing prose.

More background is in [docs/development.md](docs/development.md).

## Issues

Use GitHub issues for bugs and feature requests. Include the mliport commit,
the command, the resolved configuration (`resolved_config.json`), the backend
and framework versions, the GPU/driver, and the complete error output.

## License

By contributing you agree that your contributions are licensed under the MIT
License (see `LICENSE.md`). mliport is derived from FAIRChem (MIT; the Meta
notices are retained in the derived source files) and drives MACE,
DeePMD-kit, tensorpotential and FAIRChem as separately installed packages under
their own licenses. `licenses/` keeps the notices (CGCNN, MMF) inherited from
the FAIRChem code base; citations are in `docs/citations.md`.
