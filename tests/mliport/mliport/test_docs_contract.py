"""Documentation contract tests (task book section 29-31, 42, 45).

Facts that must not be hand-copied are byte-compared against the generators;
internal links and anchors are checked; documented CLI commands are smoke
tested; Python snippets are compiled; the DOC-04 DPA-head fix and the
"VASP-shaped, not DFT" boundary are asserted.

The site is bilingual: ``page.md`` is the English default and ``page.zh.md``
its Chinese twin (mkdocs-static-i18n suffix layout). Both languages are held to
the same structure, so a translation cannot silently drop a command, a link or
a section.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"

PAGES = (
    "index",
    "quickstart",
    "test-systems",
    "installation",
    "configuration",
    "models",
    "backends/index",
    "backends/mace",
    "backends/dpa",
    "backends/grace",
    "backends/uma",
    "workflows/index",
    "workflows/single-point",
    "workflows/optimization",
    "workflows/md",
    "workflows/neb",
    "workflows/confsearch",
    "workflows/batch",
    "workflows/queue",
    "analysis/index",
    "analysis/overview",
    "analysis/msd",
    "analysis/transport",
    "analysis/electrolyte-gemdat",
    "analysis/rdf",
    "analysis/vacf-spectrum",
    "analysis/arrhenius",
    "outputs",
    "validation",
    "troubleshooting",
    "migration-from-mlipx",
    "citations",
    "development",
)

REQUIRED_TREE = tuple(f"docs/{page}.md" for page in PAGES)


def _zh(page: str) -> Path:
    return DOCS / f"{page}.zh.md"


def _safe_parse(block: str) -> str | None:
    try:
        ast.parse(block)
    except SyntaxError as exc:
        return f"{exc}"
    return None


def _load_generator():
    path = REPO / "scripts" / "generate_docs.py"
    spec = importlib.util.spec_from_file_location("mliport_generate_docs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _markdown_files() -> list[Path]:
    files = sorted(DOCS.rglob("*.md"))
    files += [REPO / "README.md", REPO / "README_CN.md", REPO / "CONTRIBUTING.md"]
    return [path for path in files if path.is_file()]


# ------------------------------------------------------------- structure
def test_required_docs_tree_exists():
    missing = [rel for rel in REQUIRED_TREE if not (REPO / rel).is_file()]
    assert not missing, f"missing docs: {missing}"


def test_every_page_has_a_chinese_twin_and_no_chinese_page_is_orphaned():
    missing = [page for page in PAGES if not _zh(page).is_file()]
    assert not missing, f"no Chinese page: {missing}"
    strays = sorted(
        str(path.relative_to(REPO))
        for path in DOCS.rglob("*.zh.md")
        if str(path.relative_to(DOCS))[: -len(".zh.md")] not in PAGES
    )
    assert not strays, f"Chinese page without an English page: {strays}"


def _structure(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    fences = re.findall(r"^```(\w*)", text, flags=re.MULTILINE)
    return {
        "h2": len(re.findall(r"^## ", text, flags=re.MULTILINE)),
        "code_blocks": len(fences) // 2,
        "link_files": sorted(
            {
                target.partition("#")[0]
                for target in _LINK_RE.findall(text)
                if not target.startswith(("http://", "https://", "mailto:", "#"))
            }
        ),
        "generated_blocks": re.findall(r"<!-- BEGIN GENERATED: ([a-z0-9-]+) -->", text),
    }


@pytest.mark.parametrize("page", PAGES)
def test_translation_has_the_same_structure_as_the_english_page(page):
    english, chinese = _structure(DOCS / f"{page}.md"), _structure(_zh(page))
    assert chinese == english, f"{page}: zh and en structure differ"


def test_links_name_the_english_file_and_stay_inside_the_docs_tree():
    """mkdocs resolves ``page.md`` to the right language itself; a link that
    leaves ``docs/`` cannot be built by mkdocs, so it must be a GitHub URL."""
    problems: list[str] = []
    for path in DOCS.rglob("*.md"):
        for target in _LINK_RE.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            file_part = target.partition("#")[0]
            if file_part.endswith(".zh.md"):
                problems.append(f"{path.relative_to(REPO)} -> {target} (use .md)")
            resolved = (path.parent / file_part).resolve()
            if file_part and DOCS.resolve() not in (resolved, *resolved.parents):
                problems.append(f"{path.relative_to(REPO)} -> {target} leaves docs/")
    assert not problems, "\n".join(problems)


# ------------------------------------------------------------- links
_LINK_RE = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _slug(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if match:
            anchors.add(_slug(match.group(1)))
    return anchors


def test_internal_markdown_links_and_anchors_resolve():
    problems: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for target in _LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue
            file_part, _, anchor = target.partition("#")
            current = path
            if file_part:
                resolved = (path.parent / file_part).resolve()
                if resolved.is_dir():
                    resolved = resolved / "index.md"
                if not resolved.is_file():
                    problems.append(f"{path.relative_to(REPO)} -> missing {target}")
                    continue
                current = resolved
            if anchor and _slug(anchor) not in _anchors(current):
                problems.append(
                    f"{path.relative_to(REPO)} -> missing anchor {target} "
                    f"in {current.relative_to(REPO)}"
                )
    assert not problems, "\n".join(problems)


# --------------------------------------------------------- CLI smoke
DOCUMENTED_COMMANDS = (
    "--help",
    "run --help",
    "sp --help",
    "opt --help",
    "md --help",
    "neb --help",
    "confsearch --help",
    "batch --help",
    "analyze --help",
    "template --help",
    "config --help",
    "doctor --help",
    "queue --help",
)


def _mliport_executable() -> list[str]:
    candidate = Path(sys.executable).parent / "mliport"
    if candidate.is_file():
        return [str(candidate)]
    return [
        sys.executable,
        "-c",
        "from mliport.cli import main; raise SystemExit(main())",
    ]


@pytest.mark.parametrize("command", DOCUMENTED_COMMANDS)
def test_documented_cli_commands_smoke(command):
    argv = _mliport_executable() + command.split()
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


# ------------------------------------------------------ python snippets
def test_documented_python_snippets_compile():
    problems: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
        parses = [(block, _safe_parse(block)) for block in blocks]
        for block, error in parses:
            if error is not None:
                problems.append(f"{path.relative_to(REPO)}: {error}")
    assert not problems, "\n".join(problems)


def test_documented_api_names_exist():
    import mliport.api

    snippets = "\n".join(
        block
        for path in _markdown_files()
        for block in re.findall(
            r"```python\n(.*?)```",
            path.read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
    )
    documented = set(re.findall(r"from mliport(?:\.api)? import ([^\n]+)", snippets))
    names = {
        name.strip()
        for group in documented
        for name in group.split(",")
        if name.strip() and not name.strip().startswith("(")
    }
    assert {"run_single_point", "run_optimization", "run_md", "run_neb"} <= names | set(
        mliport.api.__all__
    )
    # top-level exports live in mliport/__init__.py; load it directly so the
    # repository-root namespace package cannot shadow it.
    spec = importlib.util.spec_from_file_location(
        "mliport_docs_probe", REPO / "mliport" / "mliport" / "__init__.py"
    )
    top_level_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(top_level_module)
    known = set(mliport.api.__all__) | set(top_level_module.__all__)
    missing = {name for name in names if name not in known}
    assert not missing, f"documented API names not exported: {missing}"


# ----------------------------------------------------- generated sync
def test_generated_docs_are_in_sync():
    generator = _load_generator()
    stale: list[str] = []
    for path, render in generator._generated_files().items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != render():
            stale.append(str(path.relative_to(REPO)))
    assert not stale, f"stale generated docs: {stale}"


def test_backend_pages_match_the_registry():
    import sys as _sys

    _sys.path.insert(0, str(REPO / "mliport"))
    from mliport.install.compatibility import (
        BACKENDS,
        python_versions_for,
    )

    for engine, backend in BACKENDS.items():
        for path in (
            DOCS / "backends" / f"{engine}.md",
            DOCS / "backends" / f"{engine}.zh.md",
        ):
            text = path.read_text(encoding="utf-8")
            assert backend.venv_name in text, path
            assert backend.requirement in text, path
            assert ", ".join(python_versions_for(engine)) in text, path


# ------------------------------------------------------------- language
def test_docs_keep_the_vasp_shaped_not_dft_boundary():
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    assert "not" in index and "DFT" in index
    assert "DFT" in _zh("index").read_text(encoding="utf-8")
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8").lower()
        assert "vasp-compatible" not in text, path
        assert "dft-like" not in text, path


def test_docs_do_not_claim_dpa_silently_picks_a_head():
    troubleshooting = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
    assert "fails closed" in troubleshooting
    assert "never silently picks" in troubleshooting
    for path in (
        REPO / "README.md",
        REPO / "README_CN.md",
        DOCS / "troubleshooting.md",
    ):
        text = path.read_text(encoding="utf-8")
        assert "silent 使用错误 head" not in text
        assert "silently uses the wrong head" not in text
