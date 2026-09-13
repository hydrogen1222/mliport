# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Python-native mliport installer entry point.

This is the thin-but-complete CLI behind ``scripts/install_mliport.sh``.  It is
responsible for argument parsing, GPU detection, plan generation, dry-run
rendering, and (optionally) execution — with ``shell=False`` throughout.

The shell wrapper only ensures ``uv`` exists and selects a Python 3.10–3.12
interpreter for this module (so it does not depend on the system ``python3``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from mliport.install.hardware import detect_gpus
from mliport.install.plan import (
    InstallPlanError,
    generate_plan,
    render_plan_shell,
)
from mliport.install.sources import (
    CHINA_SOURCE_CHOICES,
    SOURCE_PROFILES,
    resolve_source,
    source_environment,
    source_environment_summary,
)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mliport-install",
        description="Install mliport engines into isolated environments.",
    )
    p.add_argument(
        "--engines",
        default="uma,mace,dpa,grace",
        help="Comma-separated engines to install (default: all four).",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device (default: auto).",
    )
    p.add_argument(
        "--source",
        default="auto",
        choices=list(SOURCE_PROFILES),
        help=(
            "Package source profile. In a terminal, 'china' opens a numbered "
            "mirror menu (default: auto -> official)."
        ),
    )
    p.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Never prompt for source selection; 'china' uses TUNA PyPI + "
            "Aliyun PyTorch. Non-terminal input is always non-interactive."
        ),
    )
    p.add_argument(
        "--python",
        default="3.12",
        help=(
            "Python version for the isolated venvs. It must satisfy every "
            "requested backend's requires-python (for example UMA needs "
            ">=3.11); an incompatible combination fails closed before "
            "anything is installed."
        ),
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove each target venv before recreating it.",
    )
    p.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Do not append doctor verify steps to the plan.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without executing anything.",
    )
    return p


def _can_prompt_for_source() -> bool:
    """Return whether stdin is an interactive terminal.

    Checking only stdin is intentional: users may redirect the dry-run output
    to a file while still selecting a source from their terminal. Pipelines,
    CI jobs, and closed stdin must never block waiting for an answer.
    """
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, OSError):
        return False


def _prompt_china_source(
    input_fn: Callable[[str], str] | None = None,
) -> str:
    """Prompt for one of the numbered China mirror profiles.

    An empty response selects the historical TUNA + Aliyun combination.
    EOF and Ctrl-C abort instead of silently choosing a different source.
    """
    if input_fn is None:
        input_fn = input

    print("[mliport] Select download source / 请选择下载源:", file=sys.stderr)
    for index, name in enumerate(CHINA_SOURCE_CHOICES, start=1):
        profile = resolve_source(name)
        default = " [default / 默认]" if index == 1 else ""
        print(f"  {index}) {profile.label}{default}", file=sys.stderr)
    print(
        "[mliport] Choices 1-4 use Aliyun for PyTorch wheels and select the "
        "PyPI mirror; choice 5 uses the official PyTorch index.",
        file=sys.stderr,
    )

    while True:
        try:
            print(
                "[mliport] Choice / 选择 [1-5] (default 1): ",
                end="",
                file=sys.stderr,
                flush=True,
            )
            answer = input_fn("").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise InstallPlanError("Source selection cancelled.") from exc
        if answer == "":
            return CHINA_SOURCE_CHOICES[0]
        try:
            index = int(answer)
        except ValueError:
            index = 0
        if 1 <= index <= len(CHINA_SOURCE_CHOICES):
            return CHINA_SOURCE_CHOICES[index - 1]
        print(f"[mliport] Invalid choice {answer!r}; enter 1-5.", file=sys.stderr)


def _existing_venv_python_mismatch(venv: str, requested: str) -> bool:
    """Return True if an existing venv uses a different Python version."""
    py = Path(venv) / "bin" / "python"
    if not py.is_file():
        return False
    try:
        out = subprocess.run(
            [
                str(py),
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode != 0:
        return False
    return out.stdout.strip() != requested


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # GPU detection is done in Python so multi-GPU is handled correctly.
    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            print(
                f"[mliport] GPU: {g.name} (CC {g.compute_capability}, {g.vram_mib} MiB)"
            )
    else:
        print("[mliport] No NVIDIA GPU detected (nvidia-smi unavailable).")

    try:
        source_name = args.source
        if (
            source_name == "china"
            and not args.non_interactive
            and _can_prompt_for_source()
        ):
            source_name = _prompt_china_source()
            print(f"[mliport] Selected source: {resolve_source(source_name).label}")

        src = resolve_source(source_name)
        plan = generate_plan(
            gpus=gpus,
            engines=args.engines.split(","),
            source=source_name,
            python_version=args.python,
            device=args.device,
            clean=args.clean,
            verify=not args.skip_doctor,
        )
    except InstallPlanError as e:
        print(f"[mliport] ERROR: {e}", file=sys.stderr)
        return 2

    for w in plan.warnings:
        print(f"[mliport] WARNING: {w}", file=sys.stderr)

    env = source_environment(src, os.environ)
    for setting in source_environment_summary(src, env):
        print(f"[mliport] Effective custom source environment: {setting}")

    # Existing-venv Python mismatch check (fail unless --clean).
    if not args.clean:
        for step in plan.steps:
            if step.stage != "venv":
                continue
            # The venv path is the last argument of `uv venv --python X <path>`.
            try:
                venv = step.argv[-1]
            except IndexError:
                continue
            if _existing_venv_python_mismatch(venv, args.python):
                print(
                    f"[mliport] ERROR: existing {venv} uses a different Python than "
                    f"{args.python}. Re-run with --clean to recreate it.",
                    file=sys.stderr,
                )
                return 2

    engine_list = [engine for engine in args.engines.split(",") if engine.strip()]

    if args.dry_run:
        print()
        print(render_plan_shell(plan))
        for engine in engine_list:
            print(f"[mliport] [launcher] would create bin/mliport-{engine.strip()}")
        return 0

    # Execute
    cwd = Path.cwd()

    failures = 0
    for step in plan.steps:
        print(f"[mliport] [{step.stage}] {step.description}")
        step_env = env.copy()
        step_env.update(step.env)
        r = subprocess.run(
            step.argv,
            cwd=cwd,
            env=step_env,
            shell=False,
        )
        if r.returncode != 0:
            print(
                f"[mliport] FAILED: {step.description} (exit {r.returncode})",
                file=sys.stderr,
            )
            failures += 1
            break

    if failures:
        return 1
    from mliport.install.launcher import create_launchers  # noqa: PLC0415

    for launcher in create_launchers(engine_list, cwd):
        print(f"[mliport] [launcher] created {launcher}")
    print(f"[mliport] All {len(plan.steps)} steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
