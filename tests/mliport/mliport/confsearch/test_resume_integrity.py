"""AUD-007: confsearch checkpoint/resume integrity.

Two contracts the spec makes explicit and the existing suites do not cover:

* 13.2 — "interrupt after each possible unit, resume, compare
  ``search_results.json``, registry and raw files to an uninterrupted run
  (bitwise on CPU)";
* 11.2 — the checkpoint carries the unit index, every replica state and **all
  RNG states**; a damaged checkpoint must be refused, not reinterpreted.

All runs use the synthetic ``EggCrateCalculator`` on disposable CPU
directories; no GPU or real model is involved.

Schema 2 stores every member's own geometry and classification. Earlier
schema-1 checkpoints cannot safely restore per-member coordinates and must
fail closed; previously resumed artifacts require provenance review.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import pytest

from mliport.confsearch.checkpoint import ConfsearchCheckpointStore
from mliport.confsearch.engine import SearchEngine, run_confsearch
from mliport.confsearch.errors import ConfsearchPreparationError
from tests.mliport.mliport.confsearch.calculators import EggCrateCalculator
from tests.mliport.mliport.confsearch.test_engine import (
    SITE_ENERGIES,
    _options,
    _prepared,
)

if TYPE_CHECKING:
    from pathlib import Path

SEED = 23
_MODEL = {"model_type": "synthetic", "task": "bulk"}
_PREFLIGHT = {"status": "passed"}

#: Artifacts the current design can reproduce bitwise across a resume.
_COMPARED = (
    "search_results.json",
    "minima/registry.json",
    "diagnostics/discovery.csv",
    "minima/all_minima.extxyz",
)


def _call(options, starts, output: Path, *, resume=None, fingerprint=None) -> dict:
    return run_confsearch(
        EggCrateCalculator(),
        options,
        starts=starts,
        output_dir=output,
        seed=SEED,
        model_record=_MODEL,
        gradient_preflight=_PREFLIGHT,
        resume=resume,
        resume_fingerprint=fingerprint,
    )


def _inputs():
    options = _options()
    return options, _prepared(list(SITE_ENERGIES), options=options)


def _interrupted(tmp_path: Path, monkeypatch, *, stop_after: int) -> Path:
    """Run a search and kill it right after the ``stop_after``-th checkpoint."""

    options, starts = _inputs()
    output = tmp_path / "interrupted"
    original = SearchEngine._write_checkpoint
    counter = {"n": 0}

    def crash_after_write(self):
        original(self)
        counter["n"] += 1
        if counter["n"] >= stop_after:
            raise KeyboardInterrupt("simulated crash")

    monkeypatch.setattr(SearchEngine, "_write_checkpoint", crash_after_write)
    with pytest.raises(KeyboardInterrupt):
        _call(options, starts, output)
    monkeypatch.undo()
    return output


def _copy(tmp_path: Path, source: Path, name: str) -> tuple[Path, object, dict]:
    target = tmp_path / name
    shutil.copytree(source, target)
    data, fingerprint = ConfsearchCheckpointStore(target).load()
    return target, data, fingerprint


def _mutate_checkpoint(target: Path, mutate) -> None:
    path = target / "checkpoints" / "confsearch.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resume_reproduces_the_reference_artifacts_bitwise(tmp_path, monkeypatch):
    """Spec 13.2: the reproducible artifacts must be bitwise-identical."""

    options, starts = _inputs()
    reference = tmp_path / "reference"
    _call(options, starts, reference)

    source = _interrupted(tmp_path, monkeypatch, stop_after=5)
    target, data, fingerprint = _copy(tmp_path, source, "resumed")
    _call(options, starts, target, resume=data, fingerprint=fingerprint)

    for relative in _COMPARED:
        assert (target / relative).read_bytes() == (
            reference / relative
        ).read_bytes(), f"{relative} differs after an interrupt/resume"


def test_resume_keeps_the_fixed_atom_mask_on_every_minimum(tmp_path, monkeypatch):
    """The extxyz ``move_mask`` must not depend on when a minimum was found.

    A checkpoint round trip rebuilds registry geometries from bare
    numbers/positions/cell/PBC, so without deriving the mask from the run's own
    regions the minima registered before the checkpoint would be written
    without it while the later ones kept it.
    """

    from ase.io import read

    options, starts = _inputs()
    reference = tmp_path / "reference"
    _call(options, starts, reference)

    source = _interrupted(tmp_path, monkeypatch, stop_after=5)
    target, data, fingerprint = _copy(tmp_path, source, "resumed")
    _call(options, starts, target, resume=data, fingerprint=fingerprint)

    def constraints(path: Path) -> list[list[str]]:
        return [
            [type(constraint).__name__ for constraint in atoms.constraints]
            for atoms in read(path, index=":")
        ]

    reference_masks = constraints(reference / "minima" / "all_minima.extxyz")
    resumed_masks = constraints(target / "minima" / "all_minima.extxyz")
    assert reference_masks, "the reference must register minima"
    assert resumed_masks == reference_masks
    assert all(mask == ["FixAtoms"] for mask in resumed_masks)


def test_resume_refuses_a_unit_index_beyond_the_plan(tmp_path, monkeypatch):
    """An out-of-range unit index silently skips every unit and "completes"."""

    options, starts = _inputs()
    source = _interrupted(tmp_path, monkeypatch, stop_after=5)
    target, _data, _fingerprint = _copy(tmp_path, source, "damaged")
    _mutate_checkpoint(target, lambda payload: payload.update(unit_index=10**6))
    data, fingerprint = ConfsearchCheckpointStore(target).load()

    with pytest.raises(ConfsearchPreparationError, match="unit index"):
        _call(options, starts, target, resume=data, fingerprint=fingerprint)

    # The damaged checkpoint is refused, never rewritten or deleted.
    assert (
        json.loads(
            (target / "checkpoints" / "confsearch.json").read_text(encoding="utf-8")
        )["unit_index"]
        == 10**6
    )


def test_resume_refuses_missing_rng_states(tmp_path, monkeypatch):
    """Without the RNG states a resume silently re-seeds and diverges."""

    options, starts = _inputs()
    source = _interrupted(tmp_path, monkeypatch, stop_after=5)
    target, _data, _fingerprint = _copy(tmp_path, source, "no-rng")
    _mutate_checkpoint(target, lambda payload: payload.update(rng_states={}))
    data, fingerprint = ConfsearchCheckpointStore(target).load()

    with pytest.raises(ConfsearchPreparationError, match="RNG"):
        _call(options, starts, target, resume=data, fingerprint=fingerprint)


def test_resume_refuses_an_rng_state_that_is_missing_a_stream(tmp_path, monkeypatch):
    """A partially written RNG state must fail closed, not raise a KeyError."""

    options, starts = _inputs()
    source = _interrupted(tmp_path, monkeypatch, stop_after=5)
    target, _data, _fingerprint = _copy(tmp_path, source, "partial-rng")

    def drop_mc(payload):
        for entry in payload["rng_states"]["replicas"]:
            entry.pop("mc", None)

    _mutate_checkpoint(target, drop_mc)
    data, fingerprint = ConfsearchCheckpointStore(target).load()

    with pytest.raises(ConfsearchPreparationError, match="RNG"):
        _call(options, starts, target, resume=data, fingerprint=fingerprint)


def test_load_refuses_a_truncated_checkpoint(tmp_path, monkeypatch):
    """A corrupt checkpoint file is a domain refusal, and is left on disk."""

    source = _interrupted(tmp_path, monkeypatch, stop_after=2)
    target = tmp_path / "truncated"
    shutil.copytree(source, target)
    path = target / "checkpoints" / "confsearch.json"
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: len(text) // 2], encoding="utf-8")

    with pytest.raises(ConfsearchPreparationError, match="checkpoint"):
        ConfsearchCheckpointStore(target).load()
    assert path.is_file()


def test_resume_preserves_every_member_geometry(tmp_path, monkeypatch):
    """``all_minima.extxyz`` must hold the real geometry of every quench."""

    import numpy as np
    from ase.io import read

    options, starts = _inputs()
    reference = tmp_path / "reference"
    _call(options, starts, reference)

    source = _interrupted(tmp_path, monkeypatch, stop_after=5)
    target, data, fingerprint = _copy(tmp_path, source, "resumed")
    _call(options, starts, target, resume=data, fingerprint=fingerprint)

    before = read(reference / "minima" / "all_minima.extxyz", index=":")
    after = read(target / "minima" / "all_minima.extxyz", index=":")
    assert len(before) == len(after)
    for expected, actual in zip(before, after, strict=True):
        assert np.allclose(expected.positions, actual.positions, atol=1e-12)


def test_old_checkpoint_schema_is_refused_without_overwrite(tmp_path, monkeypatch):
    """Schema v1 lost member geometries and may not be resumed by v2."""
    source = _interrupted(tmp_path, monkeypatch, stop_after=2)
    target = tmp_path / "legacy"
    shutil.copytree(source, target)
    path = target / "checkpoints" / "confsearch.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] = "mliport.confsearch-checkpoint/1"
    payload["checkpoint_schema_revision"] = 1
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ConfsearchPreparationError, match="schema"):
        ConfsearchCheckpointStore(target).load()
    assert path.read_text(encoding="utf-8") == original
