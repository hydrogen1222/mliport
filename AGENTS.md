# mliport development contract

1. Scientific correctness has priority over speed and feature count.
2. Fail closed when physical assumptions are not satisfied.
3. Never silently change units, PBC semantics, model task/head, or dtype.
4. Never mix absolute energies from incompatible model tasks/reference levels.
5. Prefer ASE/upstream algorithms over reimplementing MD integrators.
6. Add a regression or known-answer test for scientific fixes when feasible.
7. `archive/` is historical reference only and must never be imported.
8. Active feature scope is defined by current public documentation, tests,
   issues/PRs, and the user's explicit task. Historical archives and transient
   audit/task-plan files are never normative requirements.
9. Transient agent briefs, audit scratchpads, local research data, benchmark
   outputs, and one-off development plans must not be committed.
10. Do not add features without an explicit request.
11. Preserve reproducible raw MLMD trajectories and provenance.
