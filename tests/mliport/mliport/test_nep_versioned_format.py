"""Format-version 1 NEP header regressions (GPUMD upstream model format).

These files are HEADER fixtures, not complete trained potentials. They must
never be used as evidence that the subsequent GPUMD calculation succeeds.
"""

from __future__ import annotations

import pytest

from mliport.calculators.nep_calc import parse_nep_model_header
from mliport.gpumd.input import parse_nep_model

V1 = """nep 2 Li S
format_version 1
model nep
zbl 1 2
cutoff 6 5 274 158
n_max 10 8
basis_size 8 8
l_max 4 2 1 0 0 0 0
ann 50 0
energy_reference -3.5 -4.25
@ parameters 1
0.0
"""


def _file(tmp_path, text):
    path = tmp_path / "nep89.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_new_energy_nep_header_shared_between_ase_and_native(tmp_path):
    path = _file(tmp_path, V1)
    native = parse_nep_model(path)
    ase = parse_nep_model_header(path)
    assert native["model_format"] == "nep"
    assert native["model_version"] is None
    assert native["format_version"] == 1
    assert native["model_type"] == "nep"
    assert native["element_types"] == ["Li", "S"]
    assert native["cutoffs"] == {"radial_A": 6.0, "angular_A": 5.0}
    assert native["zbl"] == [1.0, 2.0]
    assert native["energy_reference_eV"] == [-3.5, -4.25]
    assert ase["format_version"] == 1
    assert ase["model_version"] is None
    assert ase["model_type"] == native["model_type"]
    assert ase["energy_reference_eV"] == native["energy_reference_eV"]
    assert ase["max_neighbours_radial"] == 274


def test_versioned_first_line_allows_inline_comment(tmp_path):
    text = V1.replace("nep 2 Li S", "nep 2 Li S # model elements")
    assert parse_nep_model(_file(tmp_path, text))["element_types"] == ["Li", "S"]


def test_new_nep_typewise_cutoffs_and_89_element_order(tmp_path):
    symbols = (
        "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca "
        "Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr "
        "Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
        "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
        "Tl Pb Bi Po At Rn"
    ).split()
    assert len(symbols) == 86  # standard H-through-Rn ordering
    symbols += ["Fr", "Ra", "Ac"]
    text = (
        f"nep 89 {' '.join(symbols)}\n"
        "format_version 1\nmodel nep\n"
        + "cutoff "
        + " ".join(["6", "5"] * 89)
        + " 274 158\n"
        + "n_max 10 8\nbasis_size 8 8\nl_max 4 2 1 0 0 0 0\n"
        + "ann 50 0\nenergy_reference "
        + " ".join(["-1.0"] * 89)
        + "\n@ parameters 0\n"
    )
    native = parse_nep_model(_file(tmp_path, text))
    assert native["element_types"] == symbols
    assert native["cutoff_by_element_A"][88] == {
        "element": "Ac",
        "radial_A": 6.0,
        "angular_A": 5.0,
    }


@pytest.mark.parametrize(
    "name",
    [
        "nep_spin",
        "nep_temperature",
        "qnep1",
        "qnep2",
        "qnep3",
        "nep_dipole",
        "nep_polarizability",
        "nep_vdw",
        "qnep3_vdw",
    ],
)
def test_new_unsupported_model_families_fail_closed(tmp_path, name):
    text = V1.replace("model nep\n", f"model {name}\n")
    for parser in (parse_nep_model, parse_nep_model_header):
        with pytest.raises(ValueError, match="Unsupported NEP model type"):
            parser(_file(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        V1.replace("format_version 1", "format_version 2"),
        V1.replace("energy_reference -3.5 -4.25", "energy_reference -3.5"),
        V1.replace("cutoff 6 5 274 158", "cutoff 6 x 274 158"),
        V1.replace("nep 2 Li S", "nep 3 Li S"),
        V1.replace("nep 2 Li S", "nep 2 Li Li"),
        "\n" + V1,
        "\ufeff" + V1,
    ],
)
def test_malformed_new_header_rejected(tmp_path, text):
    for parser in (parse_nep_model, parse_nep_model_header):
        with pytest.raises(ValueError):
            parser(_file(tmp_path, text))


def test_legacy_nep4_and_nep4_zbl_remain_supported(tmp_path):
    for first in ("nep4 2 Li S", "nep4_zbl 2 Li S"):
        text = first + "\nzbl 1 2\ncutoff 6 5 274 158\n"
        native = parse_nep_model(_file(tmp_path, text))
        assert native["model_version"] == 4
        assert native["format_version"] is None
        assert parse_nep_model_header(_file(tmp_path, text))["model_version"] == "nep4"
