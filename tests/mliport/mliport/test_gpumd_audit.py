"""Known-answer tests for the native GPUMD MD completion audit (no ASE needed).

Exit status 0 only says that ``gpumd`` ended.  These tests pin down what makes a
dump the *complete* product of ``run.in``: the frame count GPUMD writes for a
requested number of steps (``steps // dump_interval``, evidenced by two real
V100 runs), the recorded time axis, and the alignment with ``thermo.out``.
"""

from __future__ import annotations

import json
import tracemalloc

import numpy as np
import pytest

from mliport.gpumd.audit import MDDumpAudit, expected_dump_frames
from mliport.gpumd.parse import GPUMDIncompleteRunError, GPUMDParseError

SYMBOLS = ("O", "O")
CELL = np.eye(3) * 10.0
PROPS_UNWRAPPED = ("species", "pos", "forces", "mass", "unwrapped_position")
PROPS_WRAPPED = ("species", "pos", "forces", "mass")


def _audit(**overrides):
    kwargs = {
        "steps": 20,
        "dump_interval": 10,
        "dt_fs": 1.0,
        "symbols": SYMBOLS,
        "cell": CELL,
    }
    kwargs.update(overrides)
    return MDDumpAudit(**kwargs)


def _frame(**overrides):
    values = {
        "symbols": SYMBOLS,
        "pbc": (True, True, True),
        "cell": CELL,
        "time_fs": 10.0,
        "property_names": PROPS_UNWRAPPED,
    }
    values.update(overrides)
    return values


def _feed(audit, times, **overrides):
    for time_fs in times:
        audit.observe(**_frame(time_fs=time_fs, **overrides))
    return audit


# ---------------------------------------------------------------------------
# expected frame count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("steps", "interval", "expected"),
    [
        (20, 10, 2),  # real V100 run
        (200, 20, 10),  # real V100 run
        (25, 10, 2),  # non-divisible: the trailing partial interval writes nothing
        (30, 10, 3),
        (5, 10, 0),
        (1, 1, 1),
    ],
)
def test_expected_dump_frames_known_answers(steps, interval, expected):
    assert expected_dump_frames(steps, interval) == expected


@pytest.mark.parametrize(
    ("steps", "interval"),
    [(0, 10), (10, 0), (-5, 10), (10.0, 5), (10, 2.5), (True, 1), (10, "5")],
)
def test_expected_dump_frames_rejects_invalid_arguments(steps, interval):
    with pytest.raises(ValueError, match="integer|>= 1"):
        expected_dump_frames(steps, interval)


# ---------------------------------------------------------------------------
# a complete run is accepted and fully described
# ---------------------------------------------------------------------------


def test_complete_run_is_verified_and_reported():
    audit = _feed(_audit(thermo_rows=2), [10.0, 20.0])
    report = audit.verify()
    assert report.nframes == report.expected_nframes == 2
    assert report.steps.tolist() == [10, 20]
    assert report.times_fs.tolist() == [10.0, 20.0]
    assert (report.first_step, report.last_step) == (10, 20)
    assert report.frame_interval_fs == pytest.approx(10.0)
    assert report.positions_convention == "unwrapped"
    assert report.thermo_rows == 2
    summary = report.to_dict()
    assert summary["schedule_verified"] is True
    assert "steps" not in summary
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_non_divisible_request_is_accepted_with_floor_frames():
    """25 steps at interval 10 write 2 frames (steps 10, 20); step 25 has none."""
    report = _feed(_audit(steps=25), [10.0, 20.0]).verify()
    assert report.nframes == 2
    assert report.last_step == 20


def test_first_frame_offset_inside_the_interval_is_not_asserted():
    """The repository holds no real dump pinning the first ``Time=`` offset, so a
    series that starts one step early is accepted; count, spacing and coverage
    of the requested window still bind the run length."""
    report = _feed(_audit(), [9.0, 19.0]).verify()
    assert (report.first_step, report.last_step) == (9, 19)


def test_time_step_other_than_one_femtosecond():
    audit = _audit(steps=40, dump_interval=20, dt_fs=0.5)
    report = _feed(audit, [10.0, 20.0]).verify()
    assert report.steps.tolist() == [20, 40]
    assert report.frame_interval_fs == pytest.approx(10.0)


def test_wrapped_dump_is_never_reported_as_unwrapped():
    report = _feed(_audit(), [10.0, 20.0], property_names=PROPS_WRAPPED).verify()
    assert report.positions_convention == "wrapped"


# ---------------------------------------------------------------------------
# exit status 0 but the output is not the whole run
# ---------------------------------------------------------------------------


def test_exit_zero_partial_dump_is_refused():
    """A 10000-step run that left a single well-formed frame must not complete."""
    audit = _feed(_audit(steps=10000, dump_interval=10), [10.0])
    with pytest.raises(GPUMDIncompleteRunError, match="holds 1 frame"):
        audit.verify()


def test_dump_longer_than_the_run_is_refused():
    """Three frames for a 20-step run: the file mixes two runs (dump_xyz appends)."""
    audit = _feed(_audit(), [10.0, 20.0, 30.0])
    with pytest.raises(GPUMDIncompleteRunError, match="holds 3 frame"):
        audit.verify()


def test_empty_dump_is_refused():
    with pytest.raises(GPUMDIncompleteRunError, match="no frames"):
        _audit().verify()


def test_run_shorter_than_one_dump_interval_has_nothing_to_analyse():
    audit = _feed(_audit(steps=5, dump_interval=10), [])
    with pytest.raises(GPUMDIncompleteRunError, match="no frame before"):
        audit.verify()


def test_incomplete_error_is_a_parse_error():
    assert issubclass(GPUMDIncompleteRunError, GPUMDParseError)


def test_non_uniform_time_axis_is_refused():
    audit = _feed(_audit(steps=30), [10.0, 21.0, 30.0])
    with pytest.raises(GPUMDIncompleteRunError, match="uniformly spaced"):
        audit.verify()


def test_time_axis_off_the_step_grid_is_refused():
    audit = _feed(_audit(dt_fs=1.0), [10.4, 20.4])
    with pytest.raises(GPUMDIncompleteRunError, match="step grid"):
        audit.verify()


def test_time_axis_disagreeing_with_requested_window_is_refused():
    """Two correctly spaced frames at steps 10, 20 cannot be a 100-step run."""
    audit = _feed(_audit(steps=100, dump_interval=50), [10.0, 20.0])
    # spacing 10 fs != 50 * 1 fs
    with pytest.raises(GPUMDIncompleteRunError, match="uniformly spaced"):
        audit.verify()
    audit = _feed(_audit(steps=200, dump_interval=10), [10.0, 20.0])
    with pytest.raises(GPUMDIncompleteRunError, match="holds 2 frame"):
        audit.verify()


def test_last_frame_outside_final_interval_is_refused():
    # 2 frames spaced by 10 fs but ending at step 40 for a 20-step run
    audit = _feed(_audit(), [30.0, 40.0])
    with pytest.raises(GPUMDIncompleteRunError, match="outside the final dump"):
        audit.verify()


def test_missing_or_non_finite_time_is_never_inferred():
    audit = _audit()
    with pytest.raises(GPUMDIncompleteRunError, match="never inferred"):
        audit.observe(**_frame(time_fs=None))
    with pytest.raises(GPUMDIncompleteRunError, match="never inferred"):
        audit.observe(**_frame(time_fs=float("nan")))


def test_stated_step_must_agree_with_time():
    audit = _audit()
    audit.observe(**_frame(time_fs=10.0, step=10))
    with pytest.raises(GPUMDIncompleteRunError, match="states step=11"):
        audit.observe(**_frame(time_fs=20.0, step=11))


# ---------------------------------------------------------------------------
# frame identity: species, PBC, cell, columns
# ---------------------------------------------------------------------------


def test_reordered_species_are_refused():
    audit = _audit(symbols=("O", "H"))
    with pytest.raises(GPUMDIncompleteRunError, match="atom 0 is 'H'"):
        audit.observe(**_frame(symbols=("H", "O")))


def test_changed_atom_count_is_refused():
    with pytest.raises(GPUMDIncompleteRunError, match="3 atoms, expected 2"):
        _audit().observe(**_frame(symbols=("O", "O", "O")))


@pytest.mark.parametrize("pbc", [(True, True, False), (False, False, False)])
def test_partial_or_missing_periodicity_is_refused(pbc):
    with pytest.raises(GPUMDIncompleteRunError, match="full 3D periodicity"):
        _audit().observe(**_frame(pbc=pbc))


def test_cell_change_is_refused():
    with pytest.raises(GPUMDIncompleteRunError, match="changed the cell"):
        _audit().observe(**_frame(cell=np.eye(3) * 10.5))


def test_singular_cell_is_refused():
    with pytest.raises(GPUMDIncompleteRunError, match="singular"):
        _audit(cell=None).observe(**_frame(cell=np.zeros((3, 3))))


def test_mixed_column_layouts_are_refused():
    audit = _audit()
    audit.observe(**_frame(time_fs=10.0))
    with pytest.raises(GPUMDIncompleteRunError, match="mixed layouts"):
        audit.observe(**_frame(time_fs=20.0, property_names=PROPS_WRAPPED))


# ---------------------------------------------------------------------------
# thermo.out alignment
# ---------------------------------------------------------------------------


def test_thermo_row_count_must_match_frames():
    audit = _feed(_audit(thermo_rows=3), [10.0, 20.0])
    with pytest.raises(GPUMDIncompleteRunError, match="3 row"):
        audit.verify()


def test_thermo_header_must_agree_with_run_in():
    audit = _feed(_audit(thermo_dump_interval=5), [10.0, 20.0])
    with pytest.raises(GPUMDIncompleteRunError, match="dump_thermo 5"):
        audit.verify()
    audit = _feed(_audit(thermo_dt_output_fs=5.0), [10.0, 20.0])
    with pytest.raises(GPUMDIncompleteRunError, match="dt_output"):
        audit.verify()
    # a header that agrees is fine, and an absent field is not guessed
    agreeing = _audit(thermo_dump_interval=10, thermo_dt_output_fs=10.0)
    assert _feed(agreeing, [10.0, 20.0]).verify().nframes == 2


def test_invalid_audit_arguments_fail_loudly():
    with pytest.raises(ValueError, match="dt_fs"):
        _audit(dt_fs=0.0)
    with pytest.raises(ValueError, match="dt_fs"):
        _audit(dt_fs=float("inf"))
    with pytest.raises(ValueError, match="steps"):
        _audit(steps=0)


# ---------------------------------------------------------------------------
# memory: the audit must not grow with anything but one float per frame
# ---------------------------------------------------------------------------


def test_audit_memory_is_one_float_per_frame():
    """50 000 frames must cost roughly 8 bytes each, not a frame object each."""
    nframes = 50_000
    audit = _audit(steps=nframes * 10, dump_interval=10)
    frame = _frame()
    tracemalloc.start()
    try:
        for index in range(nframes):
            audit.observe(**{**frame, "time_fs": 10.0 * (index + 1)})
        report = audit.verify()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert report.nframes == nframes
    # The report's own float64 arrays (times + steps) and the float buffer are
    # a few MB; a retained Atoms/array per frame would be hundreds of MB.
    assert peak < 16 * 1024 * 1024
