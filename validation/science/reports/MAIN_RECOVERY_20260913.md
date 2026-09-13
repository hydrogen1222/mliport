# main lineage recovery record (2026-09-13)

This note records the P0 Git/tree forensics performed for the
`mliport_main_recovery_release_bridge_plan_20260913.md` round. The observed
risk was that the remote `main` tree might have reverted to a pre-
productization lineage after a report-consolidation push. The measurements
below prove that this did **not** happen in this repository.

## Reference commits

| Name | Value |
|---|---|
| `BEFORE_MAIN` (`origin/main` at forensics time) | `a6030996310130a8b0aa71ceb9cf66896c723def` |
| `PRODUCTIZATION_RELEASE_CANDIDATE` (`v2.0.0b2` tag target) | `0f53dbb1cae3b46badd6671a2b1804e2a03734a2` |
| `REPORT_CONSOLIDATION` | `590d3ae4f032eb2ec1b456bb134b24655d3910dc` |
| old report-lineage parent | `a0d677b0660b76aee9a3e90ae08e5e24873124a4` |
| scientific evidence target | `5f8d91d4e5fbe8d8d0aacce39470757a72d5c74c` |

Recovery points created and pushed before any further work:

- `backup/current-main-20260913` -> `a603099`
- `backup/v2.0.0b2-tree-20260913` -> `0f53dbb`
- `backup/report-lineage-20260913` -> `590d3ae`

`git fetch --all --tags --prune` completed; the worktree contained only the
untracked plan document.

## Ancestry measurements

| Check | Result |
|---|---|
| `0f53dbb` ancestor of `origin/main` | yes |
| `590d3ae` ancestor of `origin/main` | yes |
| `590d3ae` ancestor of `0f53dbb` | yes |
| `0f53dbb` ancestor of `590d3ae` | no |
| `git merge-base 0f53dbb 590d3ae` | `590d3ae` |

Parents: `590d3ae` -> `a0d677b`; `0f53dbb` -> `e8edcea`; `origin/main`
(`a603099`) -> `0f53dbb`.

`git diff --stat 0f53dbb..origin/main` shows exactly one changed file:

```text
tests/mliport/mliport/test_validation_narrative.py | 17 ++++++-----
```

i.e. the productization release tree is contained in `main`; only the
formatter commit `a603099` sits on top.

## Tree-level assertions at `origin/main`

- implicit UMA default patterns (`engine_name = "uma"`,
  `ResolvedValue("uma"`, `(model_type or "uma")`): absent from `mliport/`;
- `CITATION.cff`: project-scoped authors, no top-level DOI, explicit "no
  upstream DOI is mliport's DOI" message;
- README/CONTRIBUTING/docs: no `.venv-uma`, no `pip install ./mliport`, no
  `MLIP eXtended`, no "UMA environment (default)";
- `docs/` tree: present (28 files);
- strict scientific configuration: enabled by default.

## Conclusion

No rollback and no lineage divergence exists. The report-consolidation commit
(`590d3ae`) predates productization and is an ancestor of the `v2.0.0b2`
release tree, which in turn is an ancestor of `main`. No repair branch,
history rewrite or force-push was required; `v2.0.0b2` remains immutable.

Future regressions are prevented by behavior-level productization invariant
tests (`tests/test_productization_invariants.py`), not by this note.

Final head of this recovery/release round: `v2.0.0b3` (see the tag and
release notes).
