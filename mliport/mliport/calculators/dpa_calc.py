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
        self._checkpoint_tensors_verified = False
        self._checkpoint_migrations: list[str] = []
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
            calculator = DP(**kwargs)
            try:
                self._inspect_loaded_model(calculator)
                self._validate_pt_checkpoint(calculator)
            except BaseException:
                self._active_head = None
                self._available_heads = []
                self._model_precision = []
                self._checkpoint_tensors_verified = False
                self._checkpoint_migrations = []
                raise
            # Publish only after inspection succeeds. A failed load must not
            # turn a retry into access to an unvalidated task/head.
            self._calculator = calculator
        return self._calculator

    def _inspect_loaded_model(self, calculator: Calculator) -> None:
        """Resolve the active DeepMD branch and fail closed for multi-task PESs."""
        import json  # noqa: PLC0415

        evaluator = getattr(calculator, "dp", None)
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
        self._inspect_model_definition(model_def)

    def _inspect_model_definition(self, model_def: dict) -> None:
        """Validate the full checkpoint definition, including exportable heads."""
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

    @staticmethod
    def _torch_model(calculator):
        """Find tensors in both upstream PyTorch evaluator implementations.

        pt_expt's get_model() can return a callable eager runner; its actual
        weights live on _dpmodel. The regular pt evaluator owns a ModelWrapper.
        """
        evaluator = getattr(getattr(calculator, "dp", None), "deep_eval", None)
        model = getattr(evaluator, "_dpmodel", None)
        return model if model is not None else getattr(evaluator, "dp", None)

    def _validate_pt_checkpoint(self, calculator) -> None:
        """Reject partial loads or implicit dtype conversions before inference.

        Upstream pt 3.2 loads state_dict with strict=False. A missing learned
        tensor would otherwise retain an initialization value. Read only with
        the restricted unpickler and compare the selected head's tensor state.
        """
        if self.model_path.suffix.lower() != ".pt":
            return
        model = self._torch_model(calculator)
        if model is None:
            return
        import torch  # noqa: PLC0415

        if not isinstance(model, torch.nn.Module):
            return
        payload = torch.load(self.model_path, map_location="cpu", weights_only=True)
        state = payload.get("model", payload)
        self._inspect_model_definition(state["_extra_state"]["model_params"])
        head = self._active_head or "Default"
        definition = state["_extra_state"]["model_params"]
        selected = definition.get("model_dict", {}).get(head, definition)
        descriptor = selected.get("descriptor", {})
        prefix = f"model.{head}."
        expected = {}
        for key, value in state.items():
            if not key.startswith(prefix) or not isinstance(value, torch.Tensor):
                continue
            # Upstream pt_expt strips these compiled-runner duplicate tensors
            # and unwraps original_model before loading its eager network.
            if ".compiled_forward_lower." in key or "._task_" in key:
                continue
            name = key[len(prefix) :].replace("original_model.", "", 1)
            if name in expected:
                raise RuntimeError(f"Ambiguous DPA checkpoint tensor: {name}")
            expected[name] = value
        actual = {}
        for key, value in model.state_dict().items():
            if isinstance(value, torch.Tensor):
                name = key.removeprefix("model.Default.")
                actual[name] = value
        # The pinned upstream loader deliberately rebuilds these two integer
        # gather caches by broadcasting (pt_expt/descriptor/dpa4c.py). They are
        # not learned arrays. Record their removal, never ignore other keys.
        if descriptor.get("type") == "dpa4c":
            for suffix in ("angular_channel_index", "angular_harmonic_index"):
                name = f"atomic_model.descriptor.{suffix}"
                if name in expected and name not in actual:
                    if expected[name].dtype != torch.int64:
                        raise RuntimeError(f"Invalid DPA4C integer cache: {name}")
                    del expected[name]
                    self._checkpoint_migrations.append(f"upstream rebuilt {name}")
        if not expected or expected.keys() != actual.keys():
            missing = sorted(actual.keys() - expected.keys())[:8]
            unused = sorted(expected.keys() - actual.keys())[:8]
            raise RuntimeError(
                "DPA checkpoint does not match the runtime network; refusing a "
                f"partial load. Missing tensors: {missing}; unused tensors: {unused}. "
                "Use the upstream implementation that produced this checkpoint."
            )
        for name, source in expected.items():
            loaded = actual[name]
            # Upstream SeZM's documented unbridged, non-spin 1.1/1.2 -> 1.3
            # migration changes only the schema marker. All other tensors
            # must still compare exactly. Do not extend this to spin/bridging.
            if (
                name == "atomic_model.descriptor.version_tensor"
                and descriptor.get("type") in {"dpa4", "sezm"}
                and descriptor.get("use_spin") is None
                and not descriptor.get("add_chg_spin_ebd", False)
                and selected.get("bridging_method") in {None, "None"}
                and descriptor.get("inner_clamp_f_inner") is None
                and source.numel() == loaded.numel() == 1
                and source.dtype == loaded.dtype == torch.float64
                and float(source) in {1.1, 1.2}
                and float(loaded) == 1.3
            ):
                self._checkpoint_migrations.append(
                    f"upstream unbridged descriptor version {float(source)} -> 1.3"
                )
                continue
            if (
                source.dtype != loaded.dtype
                or source.shape != loaded.shape
                or not torch.equal(source, loaded.detach().cpu())
            ):
                raise RuntimeError(
                    f"DPA checkpoint tensor changed during loading: {name}; "
                    "refusing changed values, shapes or dtype."
                )
        self._checkpoint_tensors_verified = True

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
        identity = torch_model_identity(
            self._torch_model(self._calculator), self._device
        )
        actual = identity.get("actual_device_type", "unknown")
        if actual == "cuda":
            actual = f"cuda:{identity['actual_device_logical_index']}"
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
            "actual_device": actual if actual != "unknown" else self._actual_device(),
            "device": self._device,
            "task": self._task,
            "head": self._head,
            "requested_head": self._head,
            "active_head": self._active_head,
            "available_heads": list(self._available_heads),
            "model_precision": identity.get(
                "model_precision", list(self._model_precision)
            ),
            "declared_model_precision": list(self._model_precision),
            "checkpoint_tensors_verified": self._checkpoint_tensors_verified,
            "checkpoint_migrations": list(self._checkpoint_migrations),
            "deepmd_evaluator": type(evaluator).__module__,
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
