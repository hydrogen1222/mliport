"""The documentation site configuration (mkdocs.yml and its Pages workflow).

mkdocs itself is only installed in the docs workflow, so these tests read the
configuration as data: they cannot prove the site builds (``mkdocs build
--strict`` in CI does that), but they catch the mistakes that break it most
often -- a nav entry without a page, a page missing from the nav, English not
being the default language, or the workflow not building strictly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / "docs"


class _Loader(yaml.SafeLoader):
    """SafeLoader that tolerates mkdocs' ``!!python/...`` tags as plain data."""


def _ignore_python_tag(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_Loader.add_multi_constructor("tag:yaml.org,2002:python/", _ignore_python_tag)


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.load((REPO / "mkdocs.yml").read_text(encoding="utf-8"), _Loader)


def _nav_files(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [name for item in node for name in _nav_files(item)]
    if isinstance(node, dict):
        return [name for value in node.values() for name in _nav_files(value)]
    return []


def _i18n(config: dict) -> dict:
    for plugin in config["plugins"]:
        if isinstance(plugin, dict) and "i18n" in plugin:
            return plugin["i18n"]
    raise AssertionError("mkdocs.yml has no i18n plugin")


def test_nav_lists_exactly_the_english_pages(config):
    in_nav = sorted(_nav_files(config["nav"]))
    on_disk = sorted(
        str(path.relative_to(DOCS)).replace("\\", "/")
        for path in DOCS.rglob("*.md")
        if not path.name.endswith(".zh.md")
    )
    assert in_nav == on_disk


def test_english_is_the_default_and_chinese_is_switchable(config):
    i18n = _i18n(config)
    assert i18n["docs_structure"] == "suffix"
    assert i18n["fallback_to_default"] is True
    languages = {entry["locale"]: entry for entry in i18n["languages"]}
    assert set(languages) == {"en", "zh"}
    assert languages["en"].get("default") is True
    assert not languages["zh"].get("default")
    assert languages["zh"]["name"] == "中文"
    assert i18n.get("reconfigure_material") is True


def test_chinese_nav_sections_are_translated(config):
    sections = [next(iter(item)) for item in config["nav"] if isinstance(item, dict)]
    zh = next(e for e in _i18n(config)["languages"] if e["locale"] == "zh")
    assert set(sections) <= set(zh["nav_translations"]), sections


def test_site_does_not_depend_on_google_fonts(config):
    assert config["theme"]["font"] is False


def test_docs_workflow_builds_strictly_and_deploys_from_main_only():
    workflow = yaml.safe_load(
        (REPO / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["build"]["steps"]
    runs = "\n".join(step.get("run", "") for step in steps)
    assert "mkdocs build --strict" in runs
    assert "scripts/generate_docs.py --check" in runs
    deploy = workflow["jobs"]["deploy"]
    assert "refs/heads/main" in deploy["if"]
    assert "pull_request" in deploy["if"]
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}


def test_docs_requirements_name_the_site_toolchain():
    names = {
        re.split(r"[<>=!~ ]", line, maxsplit=1)[0]
        for line in (REPO / "requirements-docs.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert names == {"mkdocs", "mkdocs-material", "mkdocs-static-i18n"}
