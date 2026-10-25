"""Options, index specifications and the v1 scope gates (spec sections 5, 12.3)."""

from __future__ import annotations

import pytest

from mliport.confsearch.errors import (
    ConfsearchError,
    ConfsearchNotImplementedError,
)
from mliport.confsearch.index_spec import format_index_spec, parse_index_spec
from mliport.confsearch.options import ConfsearchOptions, steps_from_ps
from tests.mliport.mliport.confsearch.structures import (
    mode_a_options,
    mode_b_options,
)


def test_mode_a_requires_starts_and_indices():
    options = mode_a_options()
    assert options.mode == "A"
    with pytest.raises(ConfsearchError, match="cs_starts"):
        ConfsearchOptions(cs_starts=("a.xyz",))
    with pytest.raises(ConfsearchError, match="cs_adsorbate_indices"):
        ConfsearchOptions(cs_adsorbate_indices="0-1")


def test_mode_b_requires_slab_molecule_and_n_starts():
    options = mode_b_options(cs_n_starts=3)
    assert options.mode == "B"
    assert options.cs_n_starts == 3
    with pytest.raises(ConfsearchError, match="automatic placement"):
        ConfsearchOptions(cs_slab="s.vasp", cs_molecule="m.xyz")
    with pytest.raises(ConfsearchError, match="cannot be combined"):
        ConfsearchOptions(
            cs_starts=("a.xyz",),
            cs_adsorbate_indices="0-1",
            cs_slab="s.vasp",
            cs_molecule="m.xyz",
            cs_n_starts=1,
        )


def test_future_modes_are_rejected_by_name():
    with pytest.raises(ConfsearchNotImplementedError, match="bulk modes"):
        ConfsearchOptions(
            cs_mode="bulk_mobile", cs_starts=("a.xyz",), cs_adsorbate_indices="0-1"
        )
    with pytest.raises(ConfsearchNotImplementedError, match="melt-quench"):
        ConfsearchOptions(
            cs_mode="melt_quench", cs_starts=("a.xyz",), cs_adsorbate_indices="0-1"
        )
    with pytest.raises(ConfsearchError, match="cs_mode must be 'adsorbate'"):
        ConfsearchOptions(
            cs_mode="nonsense", cs_starts=("a.xyz",), cs_adsorbate_indices="0-1"
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cs_temperature_low": 600.0}, "cs_temperature_low"),
        ({"cs_temperature_high": 0.0}, "cs_temperature_high"),
        ({"cs_snapshot_interval_ps": 2.0}, "cs_snapshot_interval_ps"),
        ({"cs_snapshot_interval_ps": -0.5}, "cs_snapshot_interval_ps"),
        ({"cs_bond_scale": 1.0}, "cs_bond_scale"),
        ({"cs_bond_break_scale": 1.1}, "cs_bond_break_scale"),
        ({"cs_reaction_policy": "maybe"}, "cs_reaction_policy"),
        ({"cs_place_contact_scale": 1.1}, "cs_place_contact_scale"),
        ({"cs_place_other_min_scale": 1.1}, "cs_place_other_min_scale"),
        ({"cs_dedup_fp_tol": 1.5}, "cs_dedup_fp_tol"),
        ({"cs_refine_fmax": 0.2}, "cs_refine_fmax"),
        ({"cs_md_mobile": "everything"}, "cs_md_mobile"),
        ({"cs_fix_below": 0.0}, "cs_fix_below"),
        ({"cs_ionic_species": "Li,Xx"}, "not an element symbol"),
        ({"timestep": 0.0}, "timestep"),
        ({"cs_hold_ps": 0.0}, "cs_hold_ps"),
        ({"cs_fp_sigma": 9.0}, "cs_fp_sigma"),
        ({"cs_top_k": 0}, "cs_top_k"),
        ({"cs_cycles": -1}, "cs_cycles"),
    ],
)
def test_cross_key_validation_fails_closed(overrides, match):
    with pytest.raises(ConfsearchError, match=match):
        mode_a_options(**overrides)


def test_anchor_options_are_mode_b_only_and_parse_indices():
    options = mode_b_options(cs_place_anchor="0,2-3", cs_place_anchor_targets="P,S")
    assert options.anchor_indices == (0, 2, 3)
    assert options.anchor_target_symbols == ("P", "S")
    with pytest.raises(ConfsearchError, match="only valid in mode B"):
        mode_a_options(cs_place_anchor="0")
    with pytest.raises(ConfsearchError, match="requires cs_place_anchor"):
        mode_b_options(cs_place_anchor_targets="S")
    with pytest.raises(ConfsearchError, match="cannot parse"):
        mode_b_options(cs_place_anchor="0;1")


def test_steps_must_be_exact_integers():
    options = mode_a_options(cs_hold_ps=2.0, timestep=0.5)
    assert options.n_hold == 4000
    assert options.n_cool == 4000
    assert options.n_snapshot_interval == 1000
    assert options.snapshot_steps == (1000, 2000, 3000)
    assert options.total_md_steps_per_cycle == 8000
    assert options.total_quench_upper_bound == 4 * options.cs_cycles

    with pytest.raises(ConfsearchError, match="nearest valid values"):
        mode_a_options(cs_hold_ps=2.0001, timestep=0.5)


def test_snapshot_interval_zero_means_no_snapshots():
    options = mode_a_options(cs_snapshot_interval_ps=0.0)
    assert options.n_snapshot_interval == 0
    assert options.snapshot_steps == ()


def test_steps_from_ps_helper():
    assert steps_from_ps(2.0, 0.5, name="x") == 4000
    assert steps_from_ps(0.0, 0.5, name="x", allow_zero=True) == 0
    with pytest.raises(ConfsearchError, match="nearest valid values"):
        steps_from_ps(0.0011, 0.5, name="x")
    with pytest.raises(ConfsearchError, match="must be"):
        steps_from_ps(-1.0, 0.5, name="x")


def test_index_spec_parsing():
    assert parse_index_spec("300-302") == (300, 301, 302)
    assert parse_index_spec("0,5,7-9") == (0, 5, 7, 8, 9)
    assert parse_index_spec(" 3 , 1 ") == (1, 3)
    assert parse_index_spec(4) == (4,)
    assert parse_index_spec([7, 3]) == (3, 7)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("", "empty"),
        ("0,,1", "empty entry"),
        ("1-", "cannot parse"),
        ("a-3", "cannot parse"),
        ("9-7", "reversed range"),
        ("0,0", "repeats indices"),
        ("2-3,3-4", "repeats indices"),
        ("-1", "negative index"),
        ("0.5", "cannot parse"),
    ],
)
def test_index_spec_rejects_malformed_input(spec, match):
    with pytest.raises(ConfsearchError, match=match):
        parse_index_spec(spec)


def test_index_spec_bounds_are_checked_against_the_structure():
    with pytest.raises(ConfsearchError, match="outside the 4-atom structure"):
        parse_index_spec("0,4", n_atoms=4)
    with pytest.raises(ConfsearchError, match="0-based"):
        parse_index_spec("-1", n_atoms=4)
    with pytest.raises(ConfsearchError, match="negative indices"):
        parse_index_spec([-1], n_atoms=4)
    assert parse_index_spec("0-3", n_atoms=4) == (0, 1, 2, 3)


def test_format_index_spec_round_trips():
    assert format_index_spec([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert format_index_spec([]) == ""
    assert parse_index_spec(format_index_spec([1, 3, 4])) == (1, 3, 4)
