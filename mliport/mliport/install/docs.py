"""Generated tables for the documentation (DOC-01/DOC-03).

The backend/environment/Python matrix and the GPU install-route table are
rendered from :mod:`mliport.install.compatibility`, so the pages never hand-copy
environment names or Python ranges. ``scripts/generate_docs.py`` splices these
tables into marker-delimited blocks of the hand-written pages
(``docs/installation.md`` and its Chinese twin ``docs/installation.zh.md``);
the docs contract test compares the committed pages with the renderer.
"""

from __future__ import annotations

from .compatibility import BACKENDS, backend_environment_rows

LANGUAGES = ("en", "zh")

_MATRIX_HEADER = {
    "en": ("Backend", "Environment", "Python", "Runtime stack"),
    "zh": ("后端", "环境", "Python", "运行时"),
}

_ENGINE_HEADER = {"en": "Engine", "zh": "引擎"}

GPU_ROUTE_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Maxwell", ("maxwell",)),
    ("Pascal", ("pascal",)),
    ("Volta / V100", ("volta",)),
    ("Ada / RTX 4090", ("ada",)),
    ("Other Turing+", ("turing", "ampere")),
    ("Hopper / Blackwell", ("hopper", "blackwell")),
)

_ROUTE_STATES = {
    "en": ("experimental", "needs runtime smoke test"),
    "zh": ("实验性", "需运行时冒烟测试"),
}


def render_backend_matrix(lang: str = "en") -> str:
    """Markdown table: which environment and Python versions each backend uses."""
    header = _MATRIX_HEADER[lang]
    lines = [
        "| " + " | ".join(header) + " |",
        "|---|---|---|---|",
    ]
    for row in backend_environment_rows():
        lines.append(
            f"| {row['label']} | `{row['environment']}` | {row['python']} "
            f"| {row['runtime']} |"
        )
    return "\n".join(lines)


def render_gpu_route_table(lang: str = "en") -> str:
    """Install-route states per engine and GPU family, from the registry.

    States are the two honest install-route states only: ``experimental``
    when any profile in the family is not upstream-supported, otherwise
    ``needs runtime smoke test``. Neither is a "verified" claim.
    """
    experimental_text, smoke_text = _ROUTE_STATES[lang]
    header = (
        f"| {_ENGINE_HEADER[lang]} | "
        + " | ".join(name for name, _ in GPU_ROUTE_COLUMNS)
        + " |"
    )
    lines = [header, "|---" * (len(GPU_ROUTE_COLUMNS) + 1) + "|"]
    for engine, backend in BACKENDS.items():
        cells = []
        for _, arch_keys in GPU_ROUTE_COLUMNS:
            profiles = [backend.arch_profiles[key] for key in arch_keys]
            experimental = any(not p.upstream_supported for p in profiles)
            cells.append(experimental_text if experimental else smoke_text)
        lines.append(f"| {engine.upper()} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = [
    "GPU_ROUTE_COLUMNS",
    "LANGUAGES",
    "backend_environment_rows",
    "render_backend_matrix",
    "render_gpu_route_table",
]
