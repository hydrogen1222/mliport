"""Grimme DFT-D3 dispersion layered on top of any mliport backend.

A DFT-D3 correction is not part of an MLIP checkpoint: a model trained on plain
PBE (MACE ``omat`` heads, UMA ``omat``, most DPA/GRACE materials models) has
learned the PBE potential energy surface, which has no long-range dispersion.
"Model + D3" means adding the D3 energy, forces and stress of the same geometry
to the model's own.  This module does exactly that with ASE's
:class:`~ase.calculators.mixing.SumCalculator` and the ``torch-dftd``
implementation of D3, independent of the backend that produced the model
energies.

Correctness guards (fail closed):

* D3 must only be added to a model whose reference data contain no dispersion.
  mliport cannot read that from an arbitrary MACE/DPA/GRACE checkpoint, so the
  user states it by enabling D3 and naming the reference functional
  (``d3_xc``).  For UMA the reference of each task is known, and only the tasks
  without dispersion are accepted, each with its own functional.
* D3 options that would not execute (``d3_xc``/``d3_damping`` without ``d3``)
  are rejected rather than recorded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mliport.backend_selection import DISPERSION_OPTION_KEYS as D3_OPTION_KEYS
from mliport.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

#: Damping functions implemented by torch-dftd.
D3_DAMPINGS: tuple[str, ...] = ("zero", "bj", "zerom", "bjm")
#: Defaults: PBE reference with Becke-Johnson damping, D3(BJ).
DEFAULT_D3_XC = "pbe"
DEFAULT_D3_DAMPING = "bj"

#: UMA tasks whose training reference contains no dispersion, mapped to the
#: reference functional whose D3 parameters must be used.  ``omol``
#: (wB97M-V, includes VV10), ``omc`` and ``odac`` (PBE-D3) already contain
#: dispersion; tasks not listed here are rejected rather than guessed.
UMA_D3_REFERENCE_XC: dict[str, str] = {"omat": "pbe", "oc20": "rpbe"}

_INSTALL_HINT = (
    "DFT-D3 (--d3 / D3 = .TRUE.) needs the 'torch-dftd' package in this "
    "backend environment, for example:\n"
    "  uv pip install --python .venv-mace/bin/python torch-dftd\n"
    "It requires PyTorch, which the MACE/DPA/UMA environments already have."
)


def normalize_d3_options(
    model_type: str,
    task: str,
    *,
    d3: Any = False,
    d3_xc: Any = None,
    d3_damping: Any = None,
) -> dict[str, str] | None:
    """Validate the D3 request and return ``{"xc", "damping"}`` or ``None``.

    Args:
        model_type: Canonical backend name (``uma`` for the fairchem alias).
        task: Normalized task (UMA task family or ``bulk``/``molecule``).
        d3: Whether the correction is requested.
        d3_xc: Reference functional of the model's training data.
        d3_damping: D3 damping function.

    Returns:
        The effective settings when D3 is enabled, else ``None``.

    Raises:
        ValueError: For settings that would not execute, unknown damping
            functions, or UMA tasks that already include dispersion.
    """
    if not isinstance(d3, bool):
        from mliport.config.schema import _coerce_bool  # noqa: PLC0415

        d3 = False if d3 is None else _coerce_bool(d3)
    if not d3:
        given = [
            name
            for name, value in (("d3_xc", d3_xc), ("d3_damping", d3_damping))
            if value is not None
        ]
        if given:
            raise ValueError(
                f"{', '.join(given)} was set but DFT-D3 is not enabled; add "
                "--d3 (D3 = .TRUE.) or remove the option instead of recording "
                "a setting that does not execute."
            )
        return None

    xc = DEFAULT_D3_XC if d3_xc is None else str(d3_xc).strip().lower()
    damping = (
        DEFAULT_D3_DAMPING if d3_damping is None else str(d3_damping).strip().lower()
    )
    if not xc:
        raise ValueError("d3_xc must name the model's reference functional.")
    if damping not in D3_DAMPINGS:
        raise ValueError(
            f"Invalid d3_damping {damping!r}. Choose one of: {', '.join(D3_DAMPINGS)}"
        )
    if model_type == "uma":
        reference = UMA_D3_REFERENCE_XC.get(task)
        if reference is None:
            raise ValueError(
                f"DFT-D3 is not allowed for UMA task {task!r}: its reference "
                "already contains dispersion or is not verified to be "
                "dispersion-free. D3 is accepted for UMA tasks: "
                + ", ".join(
                    f"{name} ({ref})" for name, ref in UMA_D3_REFERENCE_XC.items()
                )
            )
        if xc != reference:
            raise ValueError(
                f"UMA task {task!r} is trained on {reference.upper()} data; "
                f"d3_xc={xc!r} would use D3 parameters of another functional. "
                f"Use d3_xc={reference!r}."
            )
    return {"xc": xc, "damping": damping}


def d3_device(device: str, *, process_isolated: bool) -> str:
    """Torch device string for the D3 term of a backend running on ``device``.

    Process-isolated backends (DPA/GRACE) run with ``CUDA_VISIBLE_DEVICES``
    reduced to the one selected GPU, which torch then sees as ``cuda:0``.
    """
    dev = str(device).strip().lower()
    if dev == "cpu":
        return "cpu"
    if process_isolated or dev == "gpu":
        return "cuda"
    return dev


class D3CorrectedCalculator(BaseMLIPCalculator):
    """A backend wrapper whose ASE calculator is ``model + DFT-D3``."""

    def __init__(
        self,
        base: BaseMLIPCalculator,
        *,
        device: str,
        xc: str = DEFAULT_D3_XC,
        damping: str = DEFAULT_D3_DAMPING,
    ):
        """
        Args:
            base: The backend wrapper that provides the model energies.
            device: Torch device for the D3 term (see :func:`d3_device`).
            xc: Reference functional whose D3 parameters are used.
            damping: Damping function (``zero``, ``bj``, ``zerom``, ``bjm``).
        """
        if damping not in D3_DAMPINGS:
            raise ValueError(
                f"Invalid D3 damping {damping!r}. "
                f"Choose one of: {', '.join(D3_DAMPINGS)}"
            )
        self._base = base
        self._device = device
        self._xc = xc
        self._damping = damping
        self._calculator: Calculator | None = None
        self._d3_version: str | None = None

    @property
    def base(self) -> BaseMLIPCalculator:
        """The wrapped backend calculator wrapper."""
        return self._base

    def _make_d3_calculator(self) -> Calculator:
        try:
            import torch  # noqa: PLC0415
            from torch_dftd.torch_dftd3_calculator import (  # noqa: PLC0415
                TorchDFTD3Calculator,
            )
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        try:
            from importlib.metadata import version  # noqa: PLC0415

            self._d3_version = version("torch-dftd")
        except Exception:
            self._d3_version = None
        # float64: the correction is small next to the model energy and must
        # not add single-precision noise to forces near convergence.
        return TorchDFTD3Calculator(
            device=self._device,
            xc=self._xc,
            damping=self._damping,
            dtype=torch.float64,
        )

    def get_calculator(self) -> Calculator:
        """Return the cached ``SumCalculator([model, D3])``."""
        if self._calculator is None:
            from ase.calculators.mixing import SumCalculator  # noqa: PLC0415

            model = self._base.get_calculator()
            dispersion = self._make_d3_calculator()
            self._calculator = SumCalculator([model, dispersion])
        return self._calculator

    @property
    def task(self) -> str:
        return self._base.task

    @property
    def has_stress(self) -> bool:
        # D3 provides stress for periodic cells; the sum has stress exactly
        # when the model does.
        return self._base.has_stress

    @property
    def inference_mode(self) -> str:
        return self._base.inference_mode

    def dispersion_info(self) -> dict[str, Any]:
        """What was added on top of the model, for run provenance."""
        return {
            "method": "DFT-D3",
            "xc": self._xc,
            "damping": self._damping,
            "implementation": "torch-dftd",
            "version": self._d3_version,
            "device": self._device,
            "dtype": "float64",
        }

    def info(self) -> dict:
        info = dict(self._base.info())
        info["dispersion"] = self.dispersion_info()
        if self._calculator is not None:
            info["implemented_properties"] = self.implemented_properties
        return info


__all__ = [
    "D3_DAMPINGS",
    "D3_OPTION_KEYS",
    "DEFAULT_D3_DAMPING",
    "DEFAULT_D3_XC",
    "UMA_D3_REFERENCE_XC",
    "D3CorrectedCalculator",
    "d3_device",
    "normalize_d3_options",
]
