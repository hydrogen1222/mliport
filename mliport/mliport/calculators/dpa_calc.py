# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mliport project: DPA (DeepMD-kit) engine wrapper.

"""
DPA (DeepMD-kit) engine wrapper.

Wraps ``deepmd.calculator.DP`` behind the ``BaseMLIPCalculator`` contract.
``deepmd-kit`` is imported lazily so mliport works without it installed; users
only need it when selecting ``--model-type dpa``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mliport.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class DPACalculatorWrapper(BaseMLIPCalculator):
    """Wrapper for DeepMD-kit DPA ASE calculators."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        task: str = "bulk",
        head: str | None = None,
    ):
        """
        Initialize DPA calculator wrapper.

        Args:
            model_path: Path to a DPA model (``.pth``/``.pt`` for PyTorch
                backend, or ``.pb`` for the legacy TensorFlow backend).
            device: Device for calculation (``cpu`` or ``cuda``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by DPA.
            head: Optional branch/head for a multi-task DeepMD model.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._task = str(task).strip().lower()
        self._head = str(head).strip() if head is not None else None
        self._active_head: str | None = None
        self._available_heads: list[str] = []
        self._model_precision: list[str] = []
        self._calculator: Calculator | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self._task not in {"bulk", "molecule"}:
            raise ValueError("DPA task must be 'bulk' or 'molecule'.")
        if head is not None and not str(head).strip():
            raise ValueError("DPA head must be a non-empty branch name.")
        dev = str(device).lower()
        if dev not in {"cpu", "cuda", "gpu"} and not (
            dev.startswith("cuda:") and dev[5:].isdigit()
        ):
            raise ValueError(
                f"Invalid DPA device {device!r}. Use cpu, cuda, gpu, or cuda:N."
            )

    def get_calculator(self) -> Calculator:
        """Return the cached DPA ASE calculator (lazy import)."""
        if self._calculator is None:
            self._apply_device_env()
            if self.model_path.suffix.lower() == ".pb":
                try:
                    import tensorflow  # noqa: F401, PLC0415
                except ImportError as exc:
                    raise ImportError(
                        "DPA .pb requires the dpa-tensorflow-legacy runtime. "
                        "The dpa-pytorch installer profile only certifies .pt/.pth; "
                        "install a compatible DeepMD TensorFlow environment separately."
                    ) from exc
            self._preload_packaged_cuda_runtime()
            try:
                from deepmd.calculator import DP  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "DPA support requires the 'deepmd-kit' package.\n"
                    "Install with: pip install deepmd-kit>=3.0.0"
                ) from e
            kwargs = {
                "model": str(self.model_path),
                "type_dict": None,
            }
            if self._head is not None:
                kwargs["head"] = self._head
            self._calculator = DP(**kwargs)
            self._inspect_loaded_model()
        return self._calculator

    def _inspect_loaded_model(self) -> None:
        """Resolve the active DeepMD branch and fail closed for multi-task PESs."""
        import json  # noqa: PLC0415

        evaluator = getattr(self._calculator, "dp", None)
        getter = getattr(evaluator, "get_model_def_script", None)
        if not callable(getter):
            # Older single-task backends do not expose model metadata.  A
            # requested head was already validated by DP construction; with no
            # head there is no evidence that this is a multi-task model.
            self._active_head = self._head
            return
        model_def = getter()
        if isinstance(model_def, str):
            model_def = json.loads(model_def)
        if not isinstance(model_def, dict):
            raise RuntimeError(
                "DeepMD returned an unreadable model definition; refusing to "
                "guess the active task/head."
            )
        branches = model_def.get("model_dict")
        if not isinstance(branches, dict) or not branches:
            self._active_head = None
            self._model_precision = _deepmd_precisions(model_def)
            return

        self._available_heads = list(branches)
        if self._head is None:
            preview = ", ".join(self._available_heads[:12])
            suffix = " ..." if len(self._available_heads) > 12 else ""
            self._calculator = None
            raise ValueError(
                "The DPA/DeepMD model is multi-task, so --head/HEAD is "
                f"required. Available canonical branches: {preview}{suffix}. "
                "Use `dp show MODEL model-branch` for canonical names and aliases."
            )

        requested = str(self._head)
        matches: list[str] = []
        for canonical, branch in branches.items():
            aliases = (
                branch.get("model_branch_alias", []) if isinstance(branch, dict) else []
            )
            if isinstance(aliases, str):
                aliases = [aliases]
            elif not isinstance(aliases, (list, tuple, set)):
                aliases = []
            if requested == canonical or requested in aliases:
                matches.append(str(canonical))
        if len(matches) != 1:
            self._calculator = None
            if not matches:
                raise ValueError(
                    f"DPA head {requested!r} is not present in this model. "
                    f"Available canonical branches: {', '.join(self._available_heads)}"
                )
            raise ValueError(
                f"DPA head alias {requested!r} is ambiguous across branches: "
                f"{', '.join(matches)}"
            )
        self._active_head = matches[0]
        selected = branches[self._active_head]
        self._model_precision = _deepmd_precisions(selected)

    def _apply_device_env(self) -> None:
        """Validate process isolation; never reinterpret a visible ordinal."""
        from mliport.devices import require_isolated_visibility

        require_isolated_visibility(self._device, "DPA")

    @staticmethod
    def _preload_packaged_cuda_runtime() -> None:
        """Preload pip's optional CUDA runtime before importing DeepMD.

        The DeepMD wheel probes ``libcudart.so.12`` even for a CPU inference
        run. When ``nvidia-cuda-runtime-cu12`` is installed inside a venv its
        library directory is not necessarily on the dynamic-loader search
        path, producing a long but non-fatal error banner. Loading that exact
        packaged library globally avoids the false alarm. Absence of the
        optional package remains harmless.
        """
        try:
            import ctypes  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            import nvidia.cuda_runtime  # type: ignore[import-not-found]  # noqa: PLC0415

            for root in nvidia.cuda_runtime.__path__:
                candidate = Path(root) / "lib" / "libcudart.so.12"
                if candidate.is_file():
                    ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                    break
        except (ImportError, OSError):
            pass

    def _actual_device(self) -> str:
        """Best-effort actual device from the loaded DeepMD model, else 'unknown'.

        Per plan section 6.2 we never *guess* the device: if it cannot be read
        reliably from the backend we report ``'unknown'`` rather than echoing
        the requested value.
        """
        calc = self._calculator
        if calc is None:
            return "unknown"
        dp = getattr(calc, "dp", None)
        for attr in ("device", "_device"):
            d = getattr(dp, attr, None)
            if d:
                return str(d)
        return "unknown"

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); DPA itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Return model metadata.

        Distinguishes ``requested_device`` (what the user asked for) from
        ``actual_device`` (what the backend reports, or ``'unknown'``). The
        legacy ``device`` key is kept (== requested) for backward-compatible
        output writers (plan section 6.2).
        """
        from mliport.devices import torch_model_identity

        self.get_calculator()
        evaluator = getattr(getattr(self._calculator, "dp", None), "deep_eval", None)
        identity = torch_model_identity(getattr(evaluator, "dp", None), self._device)
        return {
            **identity,
            "model_type": "dpa",
            "backend_profile": (
                "dpa-tensorflow-legacy"
                if self.model_path.suffix.lower() == ".pb"
                else "dpa-pytorch"
            ),
            "model_path": str(self.model_path),
            "requested_device": self._device,
            "actual_device": self._actual_device(),
            "device": self._device,
            "task": self._task,
            "head": self._head,
            "requested_head": self._head,
            "active_head": self._active_head,
            "available_heads": list(self._available_heads),
            "model_precision": list(self._model_precision),
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }


def _deepmd_precisions(model_def: dict) -> list[str]:
    """Collect explicitly declared DeepMD descriptor/fitting precisions."""
    values: set[str] = set()

    def visit(value) -> None:
        if isinstance(value, dict):
            precision = value.get("precision")
            if isinstance(precision, str):
                values.add(precision)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(model_def)
    return sorted(values)
