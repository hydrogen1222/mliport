# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modified for the mliport project: multi-engine MLIP support (UMA/MACE/DPA/GRACE).
"""
Calculator wrapper for UMA models.

Provides a simplified interface for loading and using FAIRChem calculators
with local model files.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

from mliport.base_calculator import BaseMLIPCalculator
from mliport.gpu_compat import arch_supports_device

if TYPE_CHECKING:
    from typing import ClassVar

    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import MLIPPredictUnit


class UMACalculator(BaseMLIPCalculator):
    """Wrapper for FAIRChem UMA calculators.

    Simplifies model loading and calculator creation from local checkpoint files.

    Example:
        >>> calc_wrapper = UMACalculator("uma-s-1.pt", task="omat", device="cuda")
        >>> calculator = calc_wrapper.get_calculator()
        >>> atoms.calc = calculator
        >>> energy = atoms.get_potential_energy()
    """

    # Keep the UI/wrapper boundary tied to the installed fairchem-core API.
    # This version's UMATask enum does not expose OC22; individual checkpoints
    # are subsequently validated by FAIRChemCalculator against their datasets.
    VALID_DEVICES: ClassVar[set[str]] = {"cpu", "cuda", "gpu"}
    VALID_INFERENCE_MODES: ClassVar[set[str]] = {"default", "turbo"}

    # Known UMA tasks (used when fairchem-core is not importable, e.g. in
    # lightweight parameter-validation tests).  When fairchem-core is installed
    # its UMATask enum is authoritative.
    _FALLBACK_TASKS: ClassVar[frozenset[str]] = frozenset(
        {"omat", "omol", "oc20", "oc25", "odac", "omc"}
    )

    @property
    def VALID_TASKS(self) -> set[str]:
        """Valid UMA tasks, derived lazily from the installed fairchem-core."""
        try:
            from fairchem.core.units.mlip_unit.api.inference import (  # noqa: PLC0415
                UMATask,
            )
        except ImportError:
            return set(self._FALLBACK_TASKS)
        return {task.value for task in UMATask}

    def __init__(
        self,
        model_path: str | Path,
        task: str = "omat",
        device: str | None = None,
        inference_mode: str = "default",
        torch_num_threads: int | None = None,
        activation_checkpointing: bool | None = None,
    ):
        """Initialize UMA calculator wrapper.

        Args:
            model_path: Path to model checkpoint (.pt file)
            task: Task type (omat, omol, oc20, odac, omc)
            device: Device for calculation (cpu or cuda)
            inference_mode: Inference mode (default or turbo)
            torch_num_threads: Number of threads for PyTorch inference
            activation_checkpointing: Enable activation checkpointing

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If invalid task or inference mode
        """
        self.model_path = Path(model_path)
        self._task = task.lower()
        self.device = device or "cpu"
        self._inference_mode = inference_mode.lower()
        self.torch_num_threads = torch_num_threads
        self.activation_checkpointing = activation_checkpointing

        # Validate inputs
        self._validate()

        # Load predictor
        self._predictor = None
        self._calculator = None

    @property
    def task(self) -> str:
        """Current UMA task name (omat/omol/oc20/oc25/odac/omc)."""
        return self._task

    @property
    def inference_mode(self) -> str:
        """Inference mode: 'default' or 'turbo' (UMA only)."""
        return self._inference_mode

    def _validate(self) -> None:
        """Validate initialization parameters."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.task not in self.VALID_TASKS:
            raise ValueError(
                f"Invalid task '{self.task}'. "
                f"Must be one of: {', '.join(self.VALID_TASKS)}"
            )

        if self.inference_mode not in self.VALID_INFERENCE_MODES:
            raise ValueError(
                f"Invalid inference mode '{self.inference_mode}'. "
                f"Must be one of: {', '.join(self.VALID_INFERENCE_MODES)}"
            )
        dev = str(self.device).lower()
        if dev not in self.VALID_DEVICES and not (
            dev.startswith("cuda:") and dev[5:].isdigit()
        ):
            raise ValueError(
                f"Invalid device {self.device!r}. Use cpu, cuda, gpu, or cuda:N."
            )

    def _gpu_index(self) -> int:
        """Logical GPU index requested via ``self.device`` (plan section 3.2).

        ``cuda`` / ``gpu`` / ``cuda:0`` -> 0; ``cuda:N`` -> N. Uses the same
        logical-index convention as ``load_predict_unit(device=...)`` so the
        compatibility probe inspects the GPU the model will actually run on,
        not always physical device 0.
        """
        dev = str(self.device).lower()
        if dev.startswith("cuda:") and dev[5:].isdigit():
            return int(dev[5:])
        return 0

    def _backend_device(self) -> str:
        """Translate mliport's indexed device syntax to FAIRChem's API.

        ``load_predict_unit`` currently accepts only ``"cpu"`` or ``"cuda"``.
        For ``cuda:N`` select N as PyTorch's current device first, then pass
        the supported ``"cuda"`` token.  Passing ``cuda:N`` through directly
        reaches an assertion inside FAIRChem before model loading.
        """
        dev = str(self.device).lower()
        if dev == "cpu":
            return "cpu"
        if dev == "gpu":
            dev = "cuda"
        if dev.startswith("cuda"):
            import torch  # noqa: PLC0415

            torch.cuda.set_device(self._gpu_index())
            return "cuda"
        # _validate() makes this unreachable; keep a defensive error for
        # instances constructed under a mocked validation method.
        raise ValueError(f"Unsupported UMA device: {self.device!r}")

    def _check_gpu_compatibility(self) -> None:
        """Check if this PyTorch build includes CUDA kernels for this GPU.

        PyTorch cu128+ wheels dropped ``sm_50``/``sm_60``, so Maxwell/Pascal
        GPUs (compute capability 5.x/6.x: GTX 9xx/10xx, P104-100, P100) fail
        with ``no kernel image is available for execution on the device``.
        ``sm_60`` kernels are binary-compatible with ``sm_61`` devices. The
        install advice emitted below is generated from the central install
        compatibility matrix (``mliport.install.compatibility`` via
        ``mliport.gpu_setup.recommend_torch``), never a hardcoded pin.
        """
        # Treat ``cuda`` and ``cuda:N`` (and the ``gpu`` alias) as GPU devices.
        if not str(self.device).lower().startswith("cuda") and self.device != "gpu":
            return

        try:
            import torch

            if not torch.cuda.is_available():
                return

            idx = self._gpu_index()
            major, minor = torch.cuda.get_device_capability(idx)
            gpu_cc = f"sm_{major}{minor}"

            # Check if this PyTorch build actually includes kernels for this GPU.
            # sm_60 kernels run on sm_61 (minor-revision binary compatibility),
            # so arch_supports_device accepts sm_60 for an sm_61 device.
            arch_list = torch.cuda.get_arch_list()
            if not arch_supports_device(gpu_cc, arch_list):
                gpu_name = torch.cuda.get_device_name(idx)

                # RC-07: install advice is generated from the central
                # compatibility matrix, never a hardcoded version pin.
                from mliport.gpu_setup import recommend_torch  # noqa: PLC0415

                recommendation = recommend_torch(major, minor, engine="uma")
                if recommendation.supported:
                    matrix_advice = (
                        "Install a PyTorch build that still ships sm_60 "
                        "(sm_60 kernels are binary-compatible with sm_61):"
                        f"\n       uv pip install torch=={recommendation.version} "
                        f"--index-url {recommendation.index_url}"
                    )
                else:
                    matrix_advice = recommendation.rationale
                raise RuntimeError(
                    f"\n{'=' * 68}\n"
                    f" GPU NOT SUPPORTED BY THIS PYTORCH BUILD: {gpu_name}\n"
                    f"{'=' * 68}\n\n"
                    f"  Architecture: {gpu_cc} (CC {major}.{minor})\n"
                    f"  PyTorch kernels: {arch_list}\n\n"
                    f"  This PyTorch build has no kernel compatible with {gpu_cc}.\n"
                    "  PyTorch cu128+ wheels dropped sm_50/sm_60; the install\n"
                    "  compatibility matrix routes Maxwell/Pascal/Volta through\n"
                    "  the cu126 legacy channel.\n\n"
                    "  Options:\n"
                    f"    1. {matrix_advice}\n"
                    "    2. Build PyTorch from source with Pascal kernels:\n"
                    '       TORCH_CUDA_ARCH_LIST="6.0;6.1" '
                    "python setup.py develop\n"
                    "    3. Use a GPU with sm_70+ (Volta or newer)\n"
                    "    4. Fall back to CPU: --device cpu\n"
                    f"{'=' * 68}\n"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            # We could not run the GPU-compat probe (e.g. CUDA not
            # initialised, driver mismatch, unexpected torch error).
            # Surface this as a warning instead of silently swallowing it:
            # a silent skip lets the run proceed and only fail much later
            # inside the MD loop with a confusing "no kernel image" error.
            import warnings

            warnings.warn(
                "GPU compatibility check could not be performed and was "
                f"skipped: {exc!r}. The calculation may fail later if the "
                "PyTorch build lacks kernels for this GPU.",
                stacklevel=2,
            )

    def load_predictor(self) -> MLIPPredictUnit:
        """Load the prediction unit from checkpoint.

        Returns:
            Loaded MLIPPredictUnit
        """
        if self._predictor is None:
            self._check_gpu_compatibility()
            # Start from the named preset ("default" / "turbo"), which sets
            # external_graph_gen / internal_graph_gen_version to values that
            # actually build a neighbor graph. Building a partial
            # InferenceSettings by hand leaves those as None (checkpoint
            # default) and breaks graph generation for some models.
            from fairchem.core.units.mlip_unit import (  # noqa: PLC0415
                load_predict_unit,
            )
            from fairchem.core.units.mlip_unit.api.inference import (  # noqa: PLC0415
                guess_inference_settings,
            )

            settings = copy.deepcopy(guess_inference_settings(self.inference_mode))
            if self.activation_checkpointing is not None:
                settings.activation_checkpointing = self.activation_checkpointing
            if self.torch_num_threads is not None:
                settings.torch_num_threads = self.torch_num_threads
            # The "turbo" preset enables torch.compile (inductor/triton backend).
            # Triton only supports CUDA Capability >= 7.0, so on older GPUs
            # (e.g. Pascal sm_61: GTX 10xx, P104-100) compilation raises
            # "too old to be supported by the triton GPU compiler". Disable
            # compile there and fall back to eager — the other turbo speedups
            # (tf32, merge_mole) still apply.
            if settings.compile and not self._compile_supported():
                settings.compile = False
            self._predictor = load_predict_unit(
                path=self.model_path,
                device=self._backend_device(),
                inference_settings=settings,
            )

        return self._predictor

    def _compile_supported(self) -> bool:
        """Whether torch.compile (triton/inductor) can run on the current GPU.

        Triton requires CUDA Capability >= 7.0. Returns True when there is no
        CUDA device or the capability can't be determined (let torch.compile
        fail naturally in that case), and False only for known-unsupported
        older architectures.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return True
            major, _ = torch.cuda.get_device_capability(self._gpu_index())
            return major >= 7
        except Exception:
            return True

    def get_calculator(self) -> FAIRChemCalculator:
        """Get ASE calculator instance.

        Returns:
            FAIRChemCalculator instance
        """
        if self._calculator is None:
            from fairchem.core import FAIRChemCalculator  # noqa: PLC0415

            predictor = self.load_predictor()
            self._calculator = FAIRChemCalculator(
                predict_unit=predictor,
                task_name=self.task,
            )

        return self._calculator

    @property
    def implemented_properties(self) -> list[str]:
        """Get list of implemented properties.

        Returns:
            List of property names
        """
        calc = self.get_calculator()
        return list(calc.implemented_properties)

    @property
    def has_stress(self) -> bool:
        """Check if stress calculation is supported.

        Returns:
            True if stress is supported
        """
        return "stress" in self.implemented_properties

    def info(self) -> dict:
        """Get calculator information.

        Returns:
            Dictionary with calculator details
        """
        from mliport.devices import torch_model_identity

        self.get_calculator()
        model = getattr(self._predictor, "model", None)
        if getattr(self._predictor, "lazy_model_intialized", None) is False:
            model = None  # Upstream has not moved/executed this model yet.
        identity = torch_model_identity(model, self.device)
        return {
            **identity,
            "model_type": "uma",
            "model_path": str(self.model_path),
            "task": self.task,
            "device": self.device,
            "inference_mode": self.inference_mode,
            "direct_forces": getattr(self._predictor, "direct_forces", None),
            "compile_enabled": getattr(
                getattr(self._predictor, "inference_settings", None), "compile", None
            ),
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
