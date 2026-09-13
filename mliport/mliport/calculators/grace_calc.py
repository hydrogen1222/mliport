# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for the mliport project: GRACE (tensorpotential) engine wrapper.

"""
GRACE engine wrapper.

Wraps ``tensorpotential.calculator.TPCalculator`` behind the
``BaseMLIPCalculator`` contract. ``tensorpotential`` is imported lazily so
mliport works without it installed; users only need it when selecting
``--model-type grace``.

Note: GRACE uses a TensorFlow/XLA backend. mliport configures TensorFlow before
the first graph is built so it does not reserve the whole visible GPU by
default. A hard per-process limit can additionally be supplied when a GPU is
shared with another calculation.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mliport.base_calculator import BaseMLIPCalculator

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


def _canonicalize_model_input(data: dict[str, Any], ase_atoms: Atoms) -> dict[str, Any]:
    """Order bonds deterministically by atom indices and lattice image.

    TensorPotential's segment reductions are mathematically permutation
    invariant but finite-precision sums depend slightly on pair order.  A
    single canonical order makes cached and fresh searches numerically
    reproducible without re-running the neighbour search on cached steps.
    """
    from tensorpotential import constants  # noqa: PLC0415

    ind_i = np.asarray(data[constants.BOND_IND_I])
    ind_j = np.asarray(data[constants.BOND_IND_J])
    vectors = np.asarray(data[constants.BOND_VECTOR], dtype=float)
    if len(ind_i) < 2:
        return data
    if bool(np.asarray(ase_atoms.pbc, dtype=bool).all()) and ase_atoms.cell.rank == 3:
        raw = (
            np.asarray(ase_atoms.positions, dtype=float)[ind_j]
            - np.asarray(ase_atoms.positions, dtype=float)[ind_i]
        )
        shifts = np.rint(
            (vectors - raw) @ np.linalg.inv(np.asarray(ase_atoms.cell[:], dtype=float))
        ).astype(np.int64)
        order = np.lexsort((shifts[:, 2], shifts[:, 1], shifts[:, 0], ind_j, ind_i))
    else:
        order = np.lexsort(
            (
                np.round(vectors[:, 2], 12),
                np.round(vectors[:, 1], 12),
                np.round(vectors[:, 0], 12),
                ind_j,
                ind_i,
            )
        )
    for key in (
        constants.BOND_VECTOR,
        constants.BOND_MU_I,
        constants.BOND_MU_J,
        constants.BOND_IND_I,
        constants.BOND_IND_J,
    ):
        data[key] = np.asarray(data[key])[order]
    return data


class _PBCPreservingExtractor:
    """Prevent TensorPotential's nonperiodic setup from mutating caller atoms."""

    def __init__(self, upstream_extract: Any) -> None:
        self._upstream_extract = upstream_extract

    def __call__(self, ase_atoms: Atoms, **kwargs: Any) -> dict[str, Any]:
        if bool(np.asarray(ase_atoms.pbc, dtype=bool).all()):
            return _canonicalize_model_input(
                self._upstream_extract(ase_atoms, **kwargs), ase_atoms
            )
        work = ase_atoms.copy()
        data = self._upstream_extract(work, **kwargs)
        return _canonicalize_model_input(data, work)


class _NeighborListCache:
    """Verlet-list style neighbor cache for sequential single-structure calls.

    The expensive part of a GRACE inference on CPU is the neighbor search
    (ASE ``neighbor_list``), not the GPU kernels. In MD / relaxation the
    structure moves only slightly per step, so the full search only needs to
    be repeated every few steps.

    Mathematical guarantee (standard Verlet-list argument): an extended list
    is built at ``cutoff + skin``.  Reuse is allowed only while every atom is
    within ``skin/2`` of its rebuild position under ASE's general minimum-image
    convention.  Therefore a pair can move by at most ``skin`` and no image
    that enters the exact cutoff can be absent from the extended list.

    Every periodic image returned by matscipy is retained separately.  Its
    current vector is the rebuild vector plus the minimum-image displacements
    of atoms ``j`` and ``i``.  This works for triclinic cells and avoids the
    invalid component-wise fractional-coordinate rounding shortcut.

    The cache is deliberately restricted to fixed, fully periodic cells.
    Other PBC modes are delegated to TensorPotential's original extractor on
    a copy of the atoms, preserving both its molecule/vacuum policy and the
    caller's structure.  Atom count, species, cell, or PBC changes fail closed
    to a rebuild/fallback.
    """

    def __init__(self, builder: Any, skin: float) -> None:
        if (
            isinstance(skin, bool)
            or not isinstance(skin, (int, float))
            or float(skin) <= 0
        ):
            raise ValueError("neighbor_skin must be a positive length (Å).")
        self.builder = builder
        self._upstream_extract = builder.extract_from_ase_atoms
        self.skin = float(skin)
        # Original (exact) cutoff state.
        self.cutoff = float(builder.cutoff)
        self.cutoff_dict = builder.cutoff_dict
        self.max_cutoff = float(builder.max_cutoff)
        self.float_dtype = builder.float_dtype
        self.element_map = builder.elements_map
        # Cached extended neighbor list.
        self._ref_positions: np.ndarray | None = None
        self._ref_cell: np.ndarray | None = None
        self._ref_numbers: np.ndarray | None = None
        self._ref_pbc: np.ndarray | None = None
        # Per-image cached arrays (extended cutoff).
        self._ind_i: np.ndarray | None = None
        self._ind_j: np.ndarray | None = None
        self._shift: np.ndarray | None = None
        self._ref_bond_vectors: np.ndarray | None = None
        self._per_pair_cut: np.ndarray | None = None
        self._atomic_mu_i: np.ndarray | None = None

    def _clear(self) -> None:
        """Invalidate all geometry-dependent state."""
        self._ref_positions = None
        self._ref_cell = None
        self._ref_numbers = None
        self._ref_pbc = None
        self._ind_i = None
        self._ind_j = None
        self._shift = None
        self._ref_bond_vectors = None
        self._per_pair_cut = None
        self._atomic_mu_i = None

    def _cutoff_map(
        self, symbols: list[str], *, extended: bool
    ) -> float | dict[tuple[str, str], float]:
        """Return TensorPotential's element-pair cutoff policy."""
        extra = self.skin if extended else 0.0
        if self.cutoff_dict is None:
            return self.cutoff + extra
        return {
            (e1, e2): float(self.cutoff_dict.get((e1, e2), self.cutoff)) + extra
            for e1, e2 in combinations_with_replacement(set(symbols), 2)
        }

    def _search_extended(self, ase_atoms: Atoms) -> dict[str, Any]:
        """Build the real extended list with matscipy, retaining all images."""
        from matscipy.neighbours import neighbour_list  # noqa: PLC0415

        symbols = ase_atoms.get_chemical_symbols()
        atomic_mu_i = np.array([self.element_map[s] for s in symbols], dtype=np.int32)
        ind_i, ind_j, shift, bond_vector = neighbour_list(
            "ijSD",
            ase_atoms,
            cutoff=self._cutoff_map(symbols, extended=True),
        )
        order = np.lexsort((shift[:, 2], shift[:, 1], shift[:, 0], ind_j, ind_i))
        ind_i = ind_i[order]
        ind_j = ind_j[order]
        shift = shift[order]
        bond_vector = bond_vector[order]
        return {
            "ind_i": ind_i.astype(np.int32),
            "ind_j": ind_j.astype(np.int32),
            "shift": shift.astype(np.int32),
            "bond_vector": bond_vector.astype(self.float_dtype),
            "atomic_mu_i": atomic_mu_i,
            "symbols": symbols,
        }

    def _pair_cutoffs(
        self, symbols: list[str], ind_i: np.ndarray, ind_j: np.ndarray
    ) -> np.ndarray:
        """Return the exact cutoff for every cached periodic image."""
        if self.cutoff_dict is None:
            return np.full(len(ind_i), self.cutoff, dtype=float)
        return np.asarray(
            [
                self.cutoff_dict.get((symbols[i], symbols[j]), self.cutoff)
                for i, j in zip(ind_i, ind_j, strict=True)
            ],
            dtype=float,
        )

    def _build_extended(self, ase_atoms: Atoms) -> dict[str, Any]:
        """Run a fresh extended-cutoff search and cache the result."""
        d = self._search_extended(ase_atoms)
        self._ref_positions = np.array(ase_atoms.positions, dtype=float, copy=True)
        self._ref_cell = np.array(ase_atoms.cell[:], dtype=float, copy=True)
        self._ref_numbers = np.array(ase_atoms.numbers, copy=True)
        self._ref_pbc = np.array(ase_atoms.pbc, copy=True)
        self._ref_bond_vectors = np.array(d["bond_vector"], dtype=float, copy=True)
        self._ind_i = d["ind_i"]
        self._ind_j = d["ind_j"]
        self._shift = d["shift"]
        self._per_pair_cut = self._pair_cutoffs(d["symbols"], d["ind_i"], d["ind_j"])
        self._atomic_mu_i = d["atomic_mu_i"]
        return self._filter(ase_atoms, np.zeros_like(self._ref_positions))

    def _reuse_displacements(self, ase_atoms: Atoms) -> np.ndarray | None:
        """Return MIC displacements when reuse is safe, otherwise ``None``."""
        from ase.geometry import find_mic  # noqa: PLC0415

        if self._ref_positions is None:
            return None
        if len(ase_atoms) != len(self._ref_numbers):
            return None
        if not np.array_equal(ase_atoms.numbers, self._ref_numbers):
            return None
        if not np.array_equal(np.asarray(ase_atoms.pbc), self._ref_pbc):
            return None
        if not np.array_equal(np.asarray(ase_atoms.cell[:]), self._ref_cell):
            return None
        delta = np.asarray(ase_atoms.positions, dtype=float) - self._ref_positions
        try:
            displacement, lengths = find_mic(
                delta,
                np.asarray(ase_atoms.cell[:], dtype=float),
                pbc=np.asarray(ase_atoms.pbc, dtype=bool),
            )
        except (AssertionError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return None
        if lengths.size and float(np.max(lengths)) > self.skin / 2.0:
            return None
        return np.asarray(displacement, dtype=float)

    def _assemble(
        self, ase_atoms: Atoms, bond_vector: np.ndarray, keep: np.ndarray
    ) -> dict[str, Any]:
        """Assemble exact real neighbours plus upstream-compatible dummies."""
        from tensorpotential import constants  # noqa: PLC0415

        n = len(ase_atoms)
        ind_i = np.asarray(self._ind_i[keep], dtype=np.int32)
        ind_j = np.asarray(self._ind_j[keep], dtype=np.int32)
        vectors = np.asarray(bond_vector[keep], dtype=self.float_dtype)

        # TensorPotential requires at least one (zero-envelope) entry for each
        # atom with no exact-cutoff neighbour.  Recreate its exact-cutoff dummy
        # vector here; an extended-cutoff dummy would have the wrong magnitude.
        present = np.unique(ind_i)
        missing = np.arange(n, dtype=np.int32)[
            ~np.isin(np.arange(n, dtype=np.int32), present)
        ]
        if missing.size:
            dummy_j = np.zeros(len(missing), dtype=np.int32)
            dummy_vectors = (
                np.dot(
                    np.asarray(ase_atoms.cell[:], dtype=float),
                    np.ones((3, len(missing)), dtype=float),
                ).reshape(-1, 3)
                + self.max_cutoff
            ).astype(self.float_dtype)
            ind_i = np.concatenate((ind_i, missing))
            ind_j = np.concatenate((ind_j, dummy_j))
            vectors = np.concatenate((vectors, dummy_vectors), axis=0)
            order = np.argsort(ind_i, kind="stable")
            ind_i = ind_i[order]
            ind_j = ind_j[order]
            vectors = vectors[order]

        mu_i = self._atomic_mu_i[ind_i].astype(np.int32, copy=False)
        mu_j = self._atomic_mu_i[ind_j].astype(np.int32, copy=False)
        data = {
            constants.ATOMIC_MU_I: self._atomic_mu_i.astype(np.int32),
            constants.BOND_VECTOR: vectors,
            constants.BOND_MU_I: mu_i,
            constants.BOND_MU_J: mu_j,
            constants.BOND_IND_I: ind_i,
            constants.BOND_IND_J: ind_j,
            constants.N_ATOMS_BATCH_REAL: np.array(n, dtype=np.int32),
            constants.N_STRUCTURES_BATCH_REAL: np.array(1, dtype=np.int32),
            constants.N_NEIGHBORS_REAL: np.array(len(ind_i), dtype=np.int32),
        }
        return data

    def _filter(self, ase_atoms: Atoms, displacement: np.ndarray) -> dict[str, Any]:
        """Filter cached periodic images at the exact model cutoff."""
        cell = np.asarray(ase_atoms.cell[:], dtype=float)
        raw_displacement = (
            np.asarray(ase_atoms.positions, dtype=float) - self._ref_positions
        )
        # If a caller wraps coordinates between evaluations, raw and MIC
        # displacements differ by an integer lattice translation.  Transfer
        # that translation to each cached pair image, then reconstruct the
        # vector from current coordinates in the same algebraic form used by
        # a fresh neighbour search.  This avoids amplifying ~1e-14 Å rounding
        # from repeated ``reference_vector + delta_j - delta_i`` updates in a
        # high-order model while retaining boundary-crossing correctness.
        atom_images = np.rint(
            (raw_displacement - displacement) @ np.linalg.inv(cell)
        ).astype(np.int64)
        pair_images = self._shift - atom_images[self._ind_j] + atom_images[self._ind_i]
        bv = (
            np.asarray(ase_atoms.positions, dtype=float)[self._ind_j]
            - np.asarray(ase_atoms.positions, dtype=float)[self._ind_i]
            + pair_images @ cell
        )
        dist = np.sqrt(np.sum(bv * bv, axis=1))
        keep = dist < self._per_pair_cut
        return self._assemble(ase_atoms, bv, keep)

    def __call__(self, ase_atoms: Atoms, **kwargs: Any) -> dict[str, Any]:
        if not bool(np.asarray(ase_atoms.pbc, dtype=bool).all()):
            self._clear()
            work = ase_atoms.copy()
            data = self._upstream_extract(work, **kwargs)
            return _canonicalize_model_input(data, work)
        displacement = self._reuse_displacements(ase_atoms)
        if displacement is None:
            return self._build_extended(ase_atoms)
        return self._filter(ase_atoms, displacement)


class GRACECalculatorWrapper(BaseMLIPCalculator):
    """Wrapper for GRACE (tensorpotential) ASE calculators."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        task: str = "bulk",
        cpu_threads: int | None = None,
        gpu_memory_growth: bool = True,
        gpu_memory_limit_mb: int | None = None,
        neighbor_cache: bool = True,
        neighbor_skin: float = 1.5,
    ):
        """
        Initialize GRACE calculator wrapper.

        Args:
            model_path: Path to an exported GRACE SavedModel directory.
            device: Device for calculation (``cpu`` or ``cuda``).
            task: PBC hint (``bulk`` or ``molecule``); not consumed by GRACE.
            cpu_threads: TensorFlow intra-op CPU thread count. ``None`` keeps
                TensorFlow's default.
            gpu_memory_growth: Let TensorFlow grow its allocator on demand
                instead of reserving all visible GPU memory at startup.
            gpu_memory_limit_mb: Optional hard TensorFlow logical-device limit
                in MiB. When set it takes precedence over memory growth.
            neighbor_cache: Cache the neighbor list between calls with an
                extended cutoff (verlet-list style) and re-filter it exactly
                each call. The periodic-image multiset is equivalent to a
                fresh search; pair ordering may differ. Ideal for MD and
                relaxation where atoms move little per step.
            neighbor_skin: Extra cutoff (Å) for the cached neighbor list.
                Larger skin -> fewer rebuilds, larger filter lists. The
                rebuild happens when any atom moves more than ``skin/2``.
        """
        self.model_path = Path(model_path)
        self._device = device
        self._task = str(task).strip().lower()
        self._cpu_threads = cpu_threads
        self._gpu_memory_growth = gpu_memory_growth
        self._gpu_memory_limit_mb = gpu_memory_limit_mb
        self._neighbor_cache = neighbor_cache
        self._neighbor_skin = neighbor_skin
        self._calculator: Calculator | None = None
        self._neighbor_cache_inst: _NeighborListCache | None = None
        self._inference_failed = False

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.model_path.is_dir():
            raise ValueError(
                "GRACE model_path must be an exported SavedModel directory, "
                f"got: {self.model_path}"
            )
        if self._task not in {"bulk", "molecule"}:
            raise ValueError("GRACE task must be 'bulk' or 'molecule'.")
        dev = str(device).lower()
        if dev not in {"cpu", "cuda", "gpu"} and not (
            dev.startswith("cuda:") and dev[5:].isdigit()
        ):
            raise ValueError(
                f"Invalid GRACE device {device!r}. Use cpu, cuda, gpu, or cuda:N."
            )
        if cpu_threads is not None and (
            isinstance(cpu_threads, bool)
            or not isinstance(cpu_threads, int)
            or cpu_threads < 1
        ):
            raise ValueError("GRACE cpu_threads must be a positive integer.")
        if not isinstance(gpu_memory_growth, bool):
            raise ValueError("GRACE gpu_memory_growth must be a boolean.")
        if not isinstance(neighbor_cache, bool):
            raise ValueError("GRACE neighbor_cache must be a boolean.")
        if (
            isinstance(neighbor_skin, bool)
            or not isinstance(neighbor_skin, (int, float))
            or neighbor_skin <= 0
        ):
            raise ValueError("GRACE neighbor_skin must be a positive length (Å).")
        if gpu_memory_limit_mb is not None and (
            isinstance(gpu_memory_limit_mb, bool)
            or not isinstance(gpu_memory_limit_mb, int)
            or gpu_memory_limit_mb < 1
        ):
            raise ValueError("GRACE gpu_memory_limit_mb must be a positive integer.")
        if dev == "cpu" and gpu_memory_limit_mb is not None:
            raise ValueError(
                "GRACE gpu_memory_limit_mb applies only to a CUDA/GPU device."
            )

    def get_calculator(self) -> Calculator:
        """Return the cached GRACE ASE calculator (lazy import)."""
        if self._inference_failed:
            raise RuntimeError(
                "GRACE inference failed; create a fresh wrapper instead of "
                "reusing potentially incomplete runtime/cache state."
            )
        if self._calculator is None:
            self._apply_device_env()
            try:
                from tensorpotential.calculator import TPCalculator  # noqa: PLC0415
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "GRACE support requires the 'tensorpotential' package.\n"
                    "Install with: pip install tensorpotential"
                ) from e
            tf = None
            if self._cpu_threads is not None or self._uses_gpu:
                # Import tensorpotential first: its package initializer must set
                # TF_USE_LEGACY_KERAS before TensorFlow is imported. Threading
                # and GPU allocation are still configured before TPCalculator
                # builds/executes a TensorFlow graph.
                try:
                    import tensorflow as tf  # noqa: PLC0415
                except ImportError as e:  # pragma: no cover
                    raise ImportError(
                        "GRACE support requires TensorFlow via tensorpotential."
                    ) from e
            if self._cpu_threads is not None:
                try:
                    tf.config.threading.set_intra_op_parallelism_threads(
                        self._cpu_threads
                    )
                except RuntimeError as e:
                    raise RuntimeError(
                        "Could not set GRACE CPU threads because TensorFlow was "
                        "already initialized. Start mliport in a fresh process or "
                        "omit --cpu-threads."
                    ) from e
            if self._uses_gpu:
                assert tf is not None
                self._configure_tensorflow_gpu(tf)
            # UQ-capable GRACE exports otherwise select the full UQ signature,
            # including dsigma/dr, for an ordinary ASE energy/force request.
            # mliport does not expose those UQ tensors, so retaining them wastes
            # substantial host/GPU memory without changing the requested
            # physical observables.
            candidate = TPCalculator(
                model=str(self.model_path),
                enable_uq_if_available=False,
            )
            self._guard_inference(candidate)
            builders = getattr(candidate, "data_builders", None)
            builder = builders[0] if builders else None
            if builder is None or not hasattr(builder, "extract_from_ase_atoms"):
                if self._neighbor_cache or self._task == "molecule":
                    raise RuntimeError(
                        "This TensorPotential calculator does not expose a "
                        "compatible geometry data builder; mliport cannot install "
                        "the requested cache or preserve molecular PBC semantics."
                    )
                self._calculator = candidate
                return candidate
            if self._neighbor_cache:
                required = (
                    "cutoff",
                    "cutoff_dict",
                    "max_cutoff",
                    "float_dtype",
                    "elements_map",
                )
                missing = [name for name in required if not hasattr(builder, name)]
                if missing:
                    raise RuntimeError(
                        "GRACE neighbor cache was requested, but the installed "
                        "TensorPotential geometry builder is incompatible "
                        f"(missing: {', '.join(missing)}). Disable it explicitly "
                        "with --no-neighbor-cache or install a supported version."
                    )
                cache = _NeighborListCache(builder, self._neighbor_skin)
                builder.extract_from_ase_atoms = cache  # type: ignore[method-assign]
                self._neighbor_cache_inst = cache
            else:
                builder.extract_from_ase_atoms = _PBCPreservingExtractor(
                    builder.extract_from_ase_atoms
                )  # type: ignore[method-assign]
            self._calculator = candidate
        return self._calculator

    def _guard_inference(self, candidate) -> None:
        """Invalidate failed inference state, including a failing first call."""
        calculate = getattr(candidate, "calculate", None)
        if not callable(calculate):
            return

        def guarded(*args, **kwargs):
            if self._inference_failed:
                raise RuntimeError("GRACE calculator is in a failed inference state")
            try:
                return calculate(*args, **kwargs)
            except BaseException:
                self._inference_failed = True
                self._calculator = None
                self._neighbor_cache_inst = None
                candidate.results = {}
                raise

        candidate.calculate = guarded

    @property
    def _uses_gpu(self) -> bool:
        return str(self._device).lower() != "cpu"

    def _configure_tensorflow_gpu(self, tf) -> None:
        """Apply a bounded TensorFlow GPU policy before runtime initialisation."""
        physical_gpus = list(tf.config.list_physical_devices("GPU"))
        if not physical_gpus:
            raise RuntimeError(
                f"GRACE device {self._device!r} requested a GPU, but TensorFlow "
                "reports no visible physical GPU."
            )
        try:
            if self._gpu_memory_limit_mb is not None:
                logical_config = tf.config.LogicalDeviceConfiguration(
                    memory_limit=self._gpu_memory_limit_mb
                )
                for gpu in physical_gpus:
                    tf.config.set_logical_device_configuration(gpu, [logical_config])
            elif self._gpu_memory_growth:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            policy = (
                f"a {self._gpu_memory_limit_mb} MiB hard limit"
                if self._gpu_memory_limit_mb is not None
                else "memory growth"
            )
            raise RuntimeError(
                f"Could not configure GRACE TensorFlow GPU {policy} because "
                "the TensorFlow runtime was already initialized. Start mliport "
                "in a fresh process; refusing to run with an unbounded policy."
            ) from e

    def _apply_device_env(self) -> None:
        """Validate isolation; allocation policy uses the TensorFlow API."""
        from mliport.devices import require_isolated_visibility

        require_isolated_visibility(self._device, "GRACE")

    def _actual_device(self) -> str:
        """Best-effort actual device, else 'unknown' (plan section 7.5).

        TensorFlow device placement is not reliably queryable from the ASE
        calculator, so we report ``'unknown'`` rather than guessing.
        """
        calc = self._calculator
        if calc is None:
            return "unknown"
        for attr in ("device", "_device"):
            d = getattr(calc, attr, None)
            if d:
                return str(d)
        return "unknown"

    @property
    def task(self) -> str:
        """PBC hint (``bulk``/``molecule``); GRACE itself is task-agnostic."""
        return self._task

    @property
    def has_stress(self) -> bool:
        """Whether stress is supported."""
        return "stress" in self.implemented_properties

    def _checkpoint_precision(self) -> list[str] | None:
        """Report the dtype of the stored weights, not of cast output tensors.

        TensorPotential builds its SavedModel input/output signatures and output
        tensors in float64 regardless of the stored parameter dtype, so reading
        the output placement would misreport a float32 model as float64.
        The checkpoint variable dtype map is the authoritative evidence.
        """
        try:
            import tensorflow as tf

            checkpoint = self.model_path / "variables" / "variables"
            reader = tf.train.load_checkpoint(str(checkpoint))
            dtypes = {
                dtype.name for dtype in reader.get_variable_to_dtype_map().values()
            }
        except Exception:
            return None
        return sorted(name for name in dtypes if name.startswith("float")) or None

    def info(self) -> dict:
        """Return model metadata.

        Distinguishes ``requested_device`` from ``actual_device`` (``'unknown'``
        when TensorFlow placement cannot be read). The legacy ``device`` key is
        kept (== requested) for backward-compatible output writers (plan 7.5).
        """
        from mliport.devices import tensorflow_output_identity

        self.get_calculator()
        identity = tensorflow_output_identity(
            getattr(self._calculator, "outputs", None), self._device
        )
        return {
            **identity,
            # The checkpoint weight dtype is authoritative for precision:
            # TensorPotential casts output tensors to float64 regardless of it.
            "model_precision": (
                self._checkpoint_precision() or identity.get("model_precision")
            ),
            "model_type": "grace",
            "model_path": str(self.model_path),
            "requested_device": self._device,
            "actual_device": self._actual_device(),
            "device": self._device,
            "task": self._task,
            "cpu_threads": self._cpu_threads,
            "gpu_memory_growth": (
                self._gpu_memory_growth if self._gpu_memory_limit_mb is None else False
            ),
            "gpu_memory_limit_mb": self._gpu_memory_limit_mb,
            "neighbor_cache_requested": self._neighbor_cache,
            "neighbor_cache_status": (
                "enabled"
                if self._neighbor_cache_inst is not None
                else "not_initialized"
                if self._calculator is None
                else "disabled"
            ),
            "neighbor_skin_angstrom": (
                self._neighbor_skin if self._neighbor_cache else None
            ),
            "uq_enabled": False,
            "implemented_properties": self.implemented_properties,
            "has_stress": self.has_stress,
        }
