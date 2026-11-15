"""Regenerate the machine-derived parts of the documentation.

The documentation pages are written by hand. Only facts that live in code are
generated, and only inside marker-delimited blocks::

    <!-- BEGIN GENERATED: backend-matrix -->
    ...never edit by hand...
    <!-- END GENERATED -->

Blocks (each page exists in English as ``page.md`` and in Chinese as
``page.zh.md``; both are filled, with translated table headers):

- ``docs/installation`` -- ``backend-matrix`` (environment and Python
  versions per backend, from the compatibility registry) and ``gpu-routes``
  (install-route state per GPU family);
- ``docs/backends/<engine>`` -- ``environment`` (environment, Python range
  and runtime stack) and ``profiles`` (pinned model profiles from
  ``validation/science/model_manifest.json``);
- ``docs/incar`` -- ``incar-common`` and one ``incar-<calc type>`` block per
  calculation type (the INCAR key reference, generated from the option table in
  ``mliport/config/schema.py``);
- ``docs/workflows/neb`` -- ``neb-capability`` (which entry point can express
  which NEB option, from ``validation/science/capability_matrix.json``).

``mliport/README.md`` (the PyPI/package README) is generated from the root
``README.md`` with repository-relative links rewritten to GitHub URLs.

Usage::

    python scripts/generate_docs.py [--check]

``--check`` exits non-zero when a generated block or file is out of date
instead of rewriting it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mliport"))

from mliport.config.defaults import (  # noqa: E402
    BUILTIN_DEFAULTS,
    DEFAULT_DEVICE_BY_CALC_TYPE,
)
from mliport.config.schema import (  # noqa: E402
    TEMPLATE_GROUP_ORDER,
    get_schema,
)
from mliport.install.compatibility import (  # noqa: E402
    BACKENDS,
    python_versions_for,
    runtime_stack_text,
)
from mliport.install.docs import (  # noqa: E402
    LANGUAGES,
    render_backend_matrix,
    render_gpu_route_table,
)

MANIFEST_PATH = REPO / "validation" / "science" / "model_manifest.json"
CAPABILITY_MATRIX_PATH = REPO / "validation" / "science" / "capability_matrix.json"
GITHUB_URL = "https://github.com/hydrogen1222/mliport"

#: ``page.md`` is English, ``page.zh.md`` is Chinese.
_SUFFIX = {"en": "", "zh": ".zh"}

_BLOCK_RE = re.compile(
    r"(?P<begin><!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->)\n"
    r".*?"
    r"(?P<end><!-- END GENERATED -->)",
    flags=re.DOTALL,
)

_PROFILE_HEADER = {
    "en": ("Profile", "Identity", "dtype", "task", "head", "model SHA-256"),
    "zh": ("Profile", "标识", "dtype", "task", "head", "模型 SHA-256"),
}
_ENVIRONMENT_HEADER = {
    "en": ("Environment", "Python", "Runtime stack"),
    "zh": ("环境", "Python", "运行时"),
}
_NEB_HEADER = {
    "en": ("Feature", "API", "direct CLI", "INCAR", "TUI"),
    "zh": ("功能", "API", "直接 CLI", "INCAR", "TUI"),
}


def _row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def _profiles_for(engine: str) -> dict[str, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        key: profile
        for key, profile in manifest["profiles"].items()
        if profile.get("engine") == engine
    }


def _sha_short(sha: object) -> str:
    text = str(sha or "")
    return text[:12] + "..." if len(text) > 12 else text


def render_backend_environment(engine: str, lang: str) -> str:
    """Environment, Python range and runtime stack of one backend."""
    backend = BACKENDS[engine]
    return "\n".join(
        [
            _row(_ENVIRONMENT_HEADER[lang]),
            "|---|---|---|",
            _row(
                [
                    f"`{backend.venv_name}`",
                    ", ".join(python_versions_for(engine)),
                    runtime_stack_text(engine, lang),
                ]
            ),
            "",
            "```bash",
            f"./scripts/install_mliport.sh --engines {engine} --device auto",
            "```",
        ]
    )


def render_backend_profiles(engine: str, lang: str) -> str:
    """Pinned model profiles of one backend, from the validation manifest."""
    lines = [_row(_PROFILE_HEADER[lang]), "|---|---|---|---|---|---|"]
    for key, profile in sorted(_profiles_for(engine).items()):
        lines.append(
            _row(
                [
                    f"`{key}`",
                    str(profile.get("identity")),
                    str(profile.get("dtype")),
                    str(profile.get("task")),
                    str(profile.get("head")),
                    f"`{_sha_short(profile.get('model_sha256'))}`",
                ]
            )
        )
    return "\n".join(lines)


def render_neb_capability_table(lang: str) -> str:
    """Interface expressibility table generated from the capability matrix."""
    matrix = json.loads(CAPABILITY_MATRIX_PATH.read_text(encoding="utf-8"))
    lines = [_row(_NEB_HEADER[lang]), "|---|---|---|---|---|"]
    for feature in matrix["features"]:
        if not feature["feature"].startswith("neb_"):
            continue
        name = feature["feature"].removeprefix("neb_").replace("_", " ")
        lines.append(
            _row(
                [
                    name,
                    feature["api"],
                    feature["direct_cli"],
                    feature["incar"],
                    feature["tui"],
                ]
            )
        )
    return "\n".join(lines)


#: Calculation types that get their own key-reference section.
INCAAR_CALC_TYPES = ("sp", "opt", "md", "neb", "confsearch")

#: Scopes that are valid for every calculation type.
_COMMON_SCOPES = frozenset({"general", "output", "meta", "safety", "resources"})

_INCAR_HEADER = {
    "en": (
        "Key",
        "Type",
        "Default",
        "Allowed values",
        "Meaning",
        "Applies when",
        "Aliases",
    ),
    "zh": ("键", "类型", "默认值", "取值范围", "含义", "适用条件", "别名"),
}


def _incar_key_row(spec, calc_type: str | None) -> list[str]:
    """One row of the generated INCAR key reference."""
    default = _incar_default_text(spec, calc_type)
    allowed = _incar_allowed_text(spec)
    applies = ""
    if spec.applies_when:
        applies = ", ".join(
            f"`{key.upper()} = {value}`" for key, value in spec.applies_when
        )
    if spec.condition_calc_types and calc_type not in spec.condition_calc_types:
        applies = ""
    # Only INCAR spellings belong in this column; lowercase aliases are the
    # CLI/API names and would only confuse a user editing an INCAR file.
    aliases = sorted(
        alias for alias in spec.aliases if alias != spec.incar_key and alias.isupper()
    )
    return [
        f"`{spec.incar_key}`",
        spec.type.__name__,
        default,
        allowed,
        spec.description or "-",
        applies or "-",
        ", ".join(f"`{alias}`" for alias in aliases) or "-",
    ]


def _incar_default_text(spec, calc_type: str | None) -> str:
    if spec.placeholder:
        return f"required (`{spec.placeholder}`)"
    if spec.template_value is not None:
        return f"`{_incar_value_text(spec.template_value)}`"
    if spec.name == "device" and calc_type is not None:
        return f"`{DEFAULT_DEVICE_BY_CALC_TYPE.get(calc_type, 'cpu')}`"
    for scope in ([calc_type] if calc_type else []) + ["general", "calculator"]:
        block = BUILTIN_DEFAULTS.get(scope) or {}
        if spec.name in block:
            return f"`{_incar_value_text(block[spec.name])}`"
    if spec.default is not None:
        return f"`{_incar_value_text(spec.default)}`"
    return "-"


def _incar_value_text(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return ".TRUE." if value else ".FALSE."
    return str(value)


def _incar_allowed_text(spec) -> str:
    parts: list[str] = []
    if spec.choices:
        parts.append(", ".join(f"`{choice}`" for choice in spec.choices))
    if spec.minimum is not None or spec.maximum is not None:
        low = "" if spec.minimum is None else f"{spec.minimum}"
        high = "" if spec.maximum is None else f"{spec.maximum}"
        parts.append(f"`{low}..{high}`")
    return "; ".join(parts) or "-"


def _incar_specs(calc_type: str | None):
    """Keys valid for ``calc_type`` (``None`` = the keys common to all)."""
    specs = []
    for spec in get_schema().specs:
        scopes = set(spec.scopes)
        is_calculator = any(scope.startswith("calculator") for scope in scopes)
        if calc_type is None:
            if is_calculator or scopes & _COMMON_SCOPES:
                specs.append(spec)
            continue
        if calc_type in scopes or is_calculator or scopes & _COMMON_SCOPES:
            specs.append(spec)
        elif scopes & {"sp", "opt", "md", "neb", "batch", "confsearch"}:
            continue  # a key of another calculation type
    return specs


def render_incar_keys(lang: str, calc_type: str | None = None) -> str:
    """The generated key-reference tables of ``docs/incar`` (#61 R4).

    One table per section of the option table, in ``TEMPLATE_GROUP_ORDER``, so
    the page, the templates and ``mliport config schema`` cannot disagree.
    """
    specs = _incar_specs(calc_type)
    lines: list[str] = []
    for group in TEMPLATE_GROUP_ORDER:
        group_specs = [spec for spec in specs if spec.group == group]
        if not group_specs:
            continue
        heading = _INCAR_GROUP_HEADING[lang].format(group=group)
        lines.append(f"**{heading}**")
        lines.append("")
        lines.append(_row(_INCAR_HEADER[lang]))
        lines.append("|" + "---|" * len(_INCAR_HEADER[lang]))
        for spec in sorted(group_specs, key=lambda item: item.order):
            lines.append(_row(_incar_key_row(spec, calc_type)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_INCAR_GROUP_HEADING = {
    "en": "{group}",
    "zh": "{group}",
}


def _blocks_for(page: str, lang: str) -> dict[str, str]:
    """The generated blocks a page must contain (empty for hand-only pages)."""
    if page == "incar":
        blocks = {"incar-common": render_incar_keys(lang)}
        for calc_type in INCAAR_CALC_TYPES:
            blocks[f"incar-{calc_type}"] = render_incar_keys(lang, calc_type)
        return blocks
    if page == "installation":
        return {
            "backend-matrix": render_backend_matrix(lang),
            "gpu-routes": render_gpu_route_table(lang),
        }
    if page == "workflows/neb":
        return {"neb-capability": render_neb_capability_table(lang)}
    if page.startswith("backends/") and page.removeprefix("backends/") in BACKENDS:
        engine = page.removeprefix("backends/")
        return {
            "environment": render_backend_environment(engine, lang),
            "profiles": render_backend_profiles(engine, lang),
        }
    return {}


def splice_blocks(text: str, blocks: dict[str, str], *, where: str) -> str:
    """Replace the content of each named marker block; every block must exist."""
    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name not in blocks:
            raise SystemExit(f"{where}: unknown generated block {name!r}")
        seen.add(name)
        return f"{match.group('begin')}\n{blocks[name]}\n{match.group('end')}"

    result = _BLOCK_RE.sub(replace, text)
    missing = set(blocks) - seen
    if missing:
        raise SystemExit(f"{where}: missing generated block(s) {sorted(missing)}")
    return result


#: Pages that carry generated blocks.
GENERATED_PAGES = (
    "incar",
    "installation",
    "workflows/neb",
    *(f"backends/{engine}" for engine in BACKENDS),
)


def render_page(page: str, lang: str) -> str:
    """The committed page with its generated blocks refreshed."""
    path = REPO / "docs" / f"{page}{_SUFFIX[lang]}.md"
    return splice_blocks(
        path.read_text(encoding="utf-8"),
        _blocks_for(page, lang),
        where=str(path.relative_to(REPO)),
    )


def render_package_readme(root_readme: str) -> str:
    """PyPI/package README generated from the root README.

    Repository-relative links are rewritten to absolute GitHub URLs so the
    rendered PyPI page keeps working outside a checkout.
    """
    body = re.sub(
        r"\]\((docs/[^)]+|CITATION\.cff|LICENSE\.md|README_CN\.md|validation/[^)]+)\)",
        lambda match: f"]({GITHUB_URL}/blob/main/{match.group(1)})",
        root_readme,
    )
    header = (
        "<!-- Generated from the repository root README.md by "
        "scripts/generate_docs.py; DO NOT EDIT. -->\n\n"
    )
    return header + body


def _generated_files() -> dict[Path, callable]:
    package_readme = REPO / "mliport" / "README.md"
    files: dict[Path, callable] = {
        package_readme: lambda: render_package_readme(
            (REPO / "README.md").read_text(encoding="utf-8")
        ),
    }
    for page in GENERATED_PAGES:
        for lang in LANGUAGES:
            path = REPO / "docs" / f"{page}{_SUFFIX[lang]}.md"
            files[path] = lambda page=page, lang=lang: render_page(page, lang)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting when a generated file is stale.",
    )
    args = parser.parse_args()
    failures = 0
    for path, render in _generated_files().items():
        content = render()
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == content:
            print(f"[docs] up to date: {path.relative_to(REPO)}")
            continue
        if args.check:
            print(
                f"[docs] STALE: {path.relative_to(REPO)} "
                "(run python scripts/generate_docs.py)",
                file=sys.stderr,
            )
            failures += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[docs] wrote {path.relative_to(REPO)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
