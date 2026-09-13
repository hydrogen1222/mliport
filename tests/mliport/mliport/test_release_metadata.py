"""PR1 acceptance tests: release metadata identity (META-01).

`CITATION.cff` must describe *mliport*: the top-level authors are the project
maintainer identity, the repository URL points at mliport, and upstream
projects (fairchem/UMA, MACE, DeePMD/DPA, GRACE, ASE, kinisi, GEMDAT) are
credited as separate references / `docs/citations.md`, never as mliport's own
authors or DOI.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CFF = REPO / "CITATION.cff"
CITATIONS_DOC = REPO / "docs" / "citations.md"
README = REPO / "README.md"
UPSTREAM_DOI = "10.5281/zenodo.15587498"  # FAIRChem Zenodo DOI seen in the old CFF
REPO_OWNER_URL = "https://github.com/hydrogen1222"


def _cff() -> dict:
    return yaml.safe_load(CFF.read_text(encoding="utf-8"))


def _project_version() -> str:
    text = (REPO / "mliport" / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match, "pyproject.toml must declare a version"
    return match.group(1)


def test_citation_cff_is_valid_and_points_at_mliport():
    cff = _cff()
    assert cff["cff-version"] == "1.2.0"
    assert "mliport" in cff["title"]
    assert cff["version"] == _project_version()
    assert cff["type"] == "software"
    assert "MIT" in (
        cff["license"] if isinstance(cff["license"], list) else [cff["license"]]
    )
    url = cff.get("url", "")
    repo_code = cff.get("repository-code", "")
    assert "hydrogen1222/mliport" in url
    assert "hydrogen1222/mliport" in repo_code
    assert UPSTREAM_DOI not in url and UPSTREAM_DOI not in repo_code


def test_top_level_authors_are_the_mliport_maintainer_identity():
    """Structural rule: top-level authors belong to the mliport repository.

    No name blacklist: every top-level author must expose the repository
    owner's identity (website under the repo owner URL), while upstream
    contributors are only referenced through `references`.
    """
    cff = _cff()
    authors = cff.get("authors", [])
    assert authors, "CITATION.cff must name at least the maintainer identity"
    for author in authors:
        assert author.get("website", "").startswith(REPO_OWNER_URL), author
    referenced_names = set()
    for reference in cff.get("references", []):
        for author in reference.get("authors", []) or []:
            name = (
                author.get("family-names") or author.get("name") or author.get("alias")
            )
            if name:
                referenced_names.add(str(name))
    top_names = {
        str(author.get("family-names") or author.get("name") or author.get("alias"))
        for author in authors
    }
    assert not (
        top_names & referenced_names
    ), "upstream authors must not be re-used as mliport top-level authors"


def test_top_level_doi_is_absent_or_explicitly_mliports():
    cff = _cff()
    if "doi" not in cff:
        return
    assert (
        cff["doi"] != UPSTREAM_DOI
    ), "the FAIRChem DOI must never be mliport's top-level DOI"
    doc = CITATIONS_DOC.read_text(encoding="utf-8")
    assert str(cff["doi"]) in doc and "mliport" in doc


def test_upstream_projects_are_referenced():
    cff = _cff()
    references = cff.get("references", [])
    assert references, "CITATION.cff must carry upstream references"
    dumped = yaml.safe_dump(references, sort_keys=True).lower()
    for keyword in ("ase", "fairchem", "mace", "deepmd", "grace", "kinisi", "gemdat"):
        assert keyword in dumped, f"missing upstream reference for {keyword}"
    # the upstream DOI may only appear inside reference entries, never top-level
    top_level = {key: value for key, value in cff.items() if key != "references"}
    assert UPSTREAM_DOI not in yaml.safe_dump(top_level)


def test_citations_doc_covers_every_required_upstream():
    assert CITATIONS_DOC.is_file(), "docs/citations.md must exist"
    doc = CITATIONS_DOC.read_text(encoding="utf-8")
    lowered = doc.lower()
    for topic in (
        "mliport",
        "ase",
        "fairchem",
        "uma",
        "mace",
        "deepmd",
        "dpa",
        "grace",
        "kinisi",
        "gemdat",
    ):
        assert topic in lowered, f"docs/citations.md must mention {topic}"
    assert "citation" in lowered
    # and it must warn that upstream DOIs are not mliport's DOI
    assert "not" in lowered and ("doi" in lowered)


def test_readme_citation_section_points_to_the_citations_doc():
    text = README.read_text(encoding="utf-8")
    assert "docs/citations.md" in text
    assert re.search(
        r"^## Citation", text, flags=re.MULTILINE
    ), "README must have a Citation section"
