"""AUD-007: what a NEB geometry-resume actually preserves.

The tracker asks whether resume truly keeps the structure, the random state and
the model identity.  For NEB:

* **structure** — every checkpoint stores the complete band as an ASE
  trajectory; a resume must rebuild the *exact* band from it;
* **random state** — NEB has no stochastic optimizer.  The only randomness is
  the gradient pre-flight, which is seeded from a module constant, so it is
  reproducible without checkpointing.  There is no generator state to restore;
* **model identity** — enforced by the resume fingerprint (covered by
  ``test_neb_workflow.py``).

What is deliberately *not* preserved is the optimizer's internal state: the
checkpoint declares ``geometry_resume_only: True``, so a resume restarts the NEB
optimisation from the checkpointed geometry.  These tests pin the exact part
(band restored verbatim, prior evidence kept) and the loose part (the resumed
run converges to the same saddle, but its path — and therefore its final
energy — may differ slightly from an uninterrupted run).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms

from mliport.neb.io import NEBCheckpointStore, load_checkpoint
from mliport.neb.workflow import run_neb_workflow

if TYPE_CHECKING:
    from pathlib import Path

#: Six images and a curved minimum-energy path: the band genuinely has to move,
#: unlike a straight 1-D path where the projected force is already zero.
_CLI = {"n_intermediate_images": 6, "max_steps": 100, "checkpoint_interval": 5}


class _Interrupted(RuntimeError):
    """Test-only stand-in for a process that dies mid-run."""


class _CurvedMEP(Calculator):
    """Curved 2-D well for the moving atom; stationary at ``x_rel = +-1``."""

    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]
    arch = 0.6
    k = 5.0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        forces = np.zeros_like(positions)
        x = positions[1, 0] - 2.0
        y = positions[1, 1] - 1.5
        z = positions[1, 2] - 1.5
        residual = y - self.arch * (1.0 - x * x)
        energy = (x * x - 1.0) ** 2 + self.k * residual**2 + z * z
        forces[1, 0] = -(
            4.0 * x * (x * x - 1.0) + 4.0 * self.k * self.arch * x * residual
        )
        forces[1, 1] = -2.0 * self.k * residual
        forces[1, 2] = -2.0 * z
        self.results = {"energy": float(energy), "forces": forces}


def _endpoints() -> tuple[Atoms, Atoms]:
    initial = Atoms(
        "HHe",
        positions=[[0.5, 0.5, 0.5], [1.0, 1.5, 1.5]],
        cell=[4.0, 4.0, 4.0],
        pbc=True,
    )
    final = initial.copy()
    final.positions[1, 0] = 3.0
    initial.set_constraint(FixAtoms(indices=[0]))
    final.set_constraint(FixAtoms(indices=[0]))
    return initial, final


def _wrapper():
    from tests.mliport.mliport.neb.test_neb_workflow import FakeMACEWrapper

    return FakeMACEWrapper(_CurvedMEP())


def _resolved(model: Path, **overrides):
    from tests.mliport.mliport.neb.test_neb_workflow import _resolved as resolve

    return resolve(model, **{**_CLI, **overrides})


def _output_for(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "model.pt"
    model.write_bytes(b"model identity")
    return tmp_path / "run"


def _run(
    tmp_path: Path,
    *,
    interrupt_after_checkpoints: int | None = None,
    climb: bool = False,
) -> dict:
    output = _output_for(tmp_path)
    resolved = (
        _resolved(tmp_path / "model.pt", climb=True)
        if climb
        else _resolved(tmp_path / "model.pt")
    )
    initial, final = _endpoints()

    if interrupt_after_checkpoints is None:
        return run_neb_workflow(
            _wrapper(),
            resolved,
            output_dir=output,
            initial=initial,
            final=final,
            verbose=False,
        )

    original = NEBCheckpointStore.write
    calls = {"n": 0}

    def interrupt_after_write(self, images, **kwargs):
        calls["n"] += 1
        if calls["n"] > interrupt_after_checkpoints:
            raise _Interrupted("simulated process death")
        return original(self, images, **kwargs)

    NEBCheckpointStore.write = interrupt_after_write
    try:
        with pytest.raises(_Interrupted):
            run_neb_workflow(
                _wrapper(),
                resolved,
                output_dir=output,
                initial=initial,
                final=final,
                verbose=False,
            )
    finally:
        NEBCheckpointStore.write = original
    return {}


def _resume(tmp_path: Path) -> dict:
    output = tmp_path / "run"
    return run_neb_workflow(
        _wrapper(),
        _resolved(tmp_path / "model.pt"),
        output_dir=output,
        resume=output / "checkpoints" / "latest",
        verbose=False,
    )


def _band_of(checkpoint: Path) -> list[list[list[float]]]:
    _meta, images = load_checkpoint(checkpoint)
    return [
        np.asarray(image.get_positions(), dtype=float).round(12).tolist()
        for image in images
    ]


def _checkpoints(output: Path) -> list[Path]:
    return sorted((output / "checkpoints").glob("step_*"))


def test_resume_restores_the_checkpointed_band_exactly(tmp_path: Path) -> None:
    """The resumed attempt must start from the checkpointed band, verbatim."""

    _run(tmp_path, interrupt_after_checkpoints=3)
    output = tmp_path / "run"
    before = _checkpoints(output)
    assert len(before) == 3
    interrupted_band = _band_of(output / "checkpoints" / "latest")

    _resume(tmp_path)

    after = _checkpoints(output)
    assert len(after) > len(before), "resume must add checkpoints, not replace them"
    # NEBCheckpointStore numbers checkpoints monotonically, so the first one
    # after the interruption is the resumed attempt's own "prepared" snapshot.
    first_of_resume = after[len(before)]
    meta, _images = load_checkpoint(first_of_resume)
    assert meta["stage"] == "prepared"
    assert _band_of(first_of_resume) == interrupted_band


def test_resume_keeps_the_interrupted_attempts_evidence(tmp_path: Path) -> None:
    """Checkpoints are immutable: the interrupted attempt's files survive."""

    _run(tmp_path, interrupt_after_checkpoints=3)
    output = tmp_path / "run"
    names = [path.name for path in _checkpoints(output)]

    _resume(tmp_path)

    for name in names:
        assert (output / "checkpoints" / name).is_dir()
    exports = sorted(path.name for path in output.glob("vasp_path*"))
    assert exports and all(name.startswith("vasp_path") for name in exports)


def test_resume_converges_to_the_same_saddle(tmp_path: Path) -> None:
    """A geometry-resume re-optimises the band; the saddle must still match."""

    reference = _run(tmp_path / "reference")
    assert reference["status"] == "completed"
    assert reference["converged"] is True

    _run(tmp_path / "resumed", interrupt_after_checkpoints=3)
    resumed = _resume(tmp_path / "resumed")
    assert resumed["status"] == "completed"
    assert resumed["converged"] is True

    # Same reaction channel and same saddle, not the same path: the checkpoint
    # is geometry-only, so the resumed optimiser restarts and lands on the same
    # stationary point up to its own convergence tolerance.  The observed
    # difference for this fixture is ~2e-4 eV; 1e-3 eV is a loose guard against
    # a gross divergence, not a precision claim.
    assert resumed["climbing_image_index"] == reference["climbing_image_index"]
    assert resumed["barrier_forward_sampled_eV"] == pytest.approx(
        reference["barrier_forward_sampled_eV"], abs=1.0e-3
    )


def test_geometry_resume_only_is_recorded_not_implicit(tmp_path: Path) -> None:
    """The limitation is explicit in the checkpoint and the run context."""

    _run(tmp_path, interrupt_after_checkpoints=2)
    _resume(tmp_path)

    output = tmp_path / "run"
    metadata, _images = load_checkpoint(output / "checkpoints" / "latest")
    assert metadata["geometry_resume_only"] is True

    context = json.loads((output / "run_context.json").read_text(encoding="utf-8"))
    assert context["geometry_resume_only"] is True


# --- R10-NEB-RESUME-01: explicit option changes are refused, and named --------
#
# A resume inherits every option the user did not pass (the checkpoint layer
# supplies them), accepts an explicit value that matches the checkpoint, and
# refuses an explicit value that differs.  The refusal must name the fields that
# differ, not only say "incompatible".


def _resume_with(tmp_path: Path, **overrides) -> dict:
    """Resume the way the CLI does: checkpoint layer + explicit CLI overrides.

    ``mliport neb --resume`` loads the checkpoint's resolved configuration as the
    INCAR/job layer and lets explicit CLI arguments win.  Reproducing that here
    is what makes "omitted" mean "inherit" and "explicit and different" mean
    "conflict".
    """
    from mliport.config import resolve_config
    from mliport.neb.workflow import checkpoint_resolved_layer

    output = tmp_path / "run"
    layer = checkpoint_resolved_layer(output / "checkpoints" / "latest")
    cli = {"model_path": str(tmp_path / "model.pt")}
    cli.update(overrides)
    return run_neb_workflow(
        _wrapper(),
        resolve_config(calc_type="neb", incar=layer, cli=cli),
        output_dir=output,
        resume=output / "checkpoints" / "latest",
        verbose=False,
    )


def test_resume_inherits_options_the_user_omitted(tmp_path: Path) -> None:
    """Not passing max_steps/climb means "keep the checkpointed values"."""

    _run(tmp_path, interrupt_after_checkpoints=3)
    resumed = _resume_with(tmp_path)
    assert resumed["status"] == "completed"
    assert resumed["converged"] is True


def test_resume_accepts_an_explicit_value_that_matches(tmp_path: Path) -> None:
    """Explicitly repeating the checkpointed value is allowed."""

    _run(tmp_path, interrupt_after_checkpoints=3)
    resumed = _resume_with(tmp_path, max_steps=100, n_intermediate_images=6)
    assert resumed["status"] == "completed"


def test_resume_rejects_a_changed_step_budget_and_names_it(tmp_path: Path) -> None:
    _run(tmp_path, interrupt_after_checkpoints=3)
    with pytest.raises(Exception) as excinfo:
        _resume_with(tmp_path, max_steps=17, n_intermediate_images=6)
    message = str(excinfo.value)
    assert "Resume fingerprint is incompatible" in message
    assert "neb_options.max_steps" in message
    assert "17" in message and "100" in message


def test_resume_accepts_explicit_climb_when_the_band_climbs(tmp_path: Path) -> None:
    """Explicit --climb identical to the recorded value is allowed."""

    _run(tmp_path, interrupt_after_checkpoints=3, climb=True)
    resumed = _resume_with(tmp_path, climb=True)
    assert resumed["status"] == "completed"


def test_resume_rejects_no_climb_on_a_climbing_image_band(tmp_path: Path) -> None:
    """Explicit --no-climb against a recorded climbing band is a conflict."""

    _run(tmp_path, interrupt_after_checkpoints=3, climb=True)
    with pytest.raises(Exception) as excinfo:
        _resume_with(tmp_path, climb=False)
    message = str(excinfo.value)
    assert "Resume fingerprint is incompatible" in message
    assert "neb_options.climb" in message


def test_resume_refusal_leaves_the_checkpoint_untouched(tmp_path: Path) -> None:
    _run(tmp_path, interrupt_after_checkpoints=3)
    output = tmp_path / "run"
    before = {p.name: sorted(x.name for x in p.iterdir()) for p in _checkpoints(output)}
    with pytest.raises(Exception):
        _resume_with(tmp_path, max_steps=17, n_intermediate_images=6)
    after = {p.name: sorted(x.name for x in p.iterdir()) for p in _checkpoints(output)}
    assert after == before


def test_resume_accepts_no_climb_when_the_band_does_not_climb(tmp_path: Path) -> None:
    """Explicit --no-climb matching a plain band is allowed."""

    _run(tmp_path, interrupt_after_checkpoints=3)
    resumed = _resume_with(tmp_path, climb=False)
    assert resumed["status"] == "completed"


def test_fingerprint_differences_lists_nested_paths_and_limits_output() -> None:
    from mliport.neb.workflow import _fingerprint_differences

    recorded = {
        "model": {"model_sha256": "aa", "task": "bulk"},
        "neb_options": {"max_steps": 100, "climb": True, "fmax_eV_A": 0.05},
        "band": {"image_count": 7},
    }
    current = {
        "model": {"model_sha256": "aa", "task": "bulk"},
        "neb_options": {"max_steps": 17, "climb": False, "fmax_eV_A": 0.05},
        "band": {"image_count": 7},
    }
    differences = _fingerprint_differences(recorded, current)
    assert len(differences) == 2
    assert any(
        "neb_options.max_steps" in d and "100" in d and "17" in d for d in differences
    )
    assert any("neb_options.climb" in d for d in differences)

    # identical fingerprints produce no noise
    assert _fingerprint_differences(recorded, recorded) == []

    # keys present on only one side are reported too, and the list is bounded
    only_one_side = {**recorded, "extra": 1}
    one = _fingerprint_differences(recorded, only_one_side)
    assert one and "extra" in one[0]
    many = {"root": {f"k{i}": i for i in range(20)}}
    assert len(_fingerprint_differences(many, {"root": {}})) == 6
