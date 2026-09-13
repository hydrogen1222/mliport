# Batch processing

**Purpose.** Run the same calculation (default: single point) over many
structure files with one model, one backend and one configuration.

**Minimal command.**

```bash
.venv-mace/bin/mliport batch structures/ \
  --model mace-omat-0-medium.model --model-type mace \
  --calc-type sp --pattern "*.cif" \
  --output batch_results
```

**Important options.**

| Option | Meaning |
|---|---|
| `--calc-type` | sub-calculation to run per structure: `sp` (default), `opt` |
| `--pattern` | file glob; when omitted, discovers `*.cif`, `*.xyz`, `*.vasp`, `POSCAR*` |
| `--output` | batch output directory (default `batch_results`) |
| `--model`, `--model-type`, `--task`, `--head`, `--device` | model/backend selection |

**Inputs.** A directory of structures and one model/backend. All structures
share the same calculator and configuration; the batch runner loads the model
once and reuses the calculator for efficiency.

**Outputs.** One run directory per structure under `--output`, each with the
normal SP/OPT outputs, plus batch-level progress/failure reporting.

**Failure semantics.** A failing structure is reported as failed and does not
silently produce a zero-energy entry. Missing/empty inputs fail closed.
Batch results retain per-structure model identity and resolved configuration.

**Scientific caveats.** Batch mode is a throughput feature, not a
consistency guarantee: check that every structure matches the backend's task
semantics (periodic vs molecular). Do not mix structures requiring different
heads/tasks in one batch — use separate batch runs and say so.

**Example.**

```bash
.venv-dpa/bin/mliport batch structures/ \
  --model DPA-3.1-3M.pt --model-type dpa --head Omat24 \
  --calc-type sp --output runs/dpa-batch
```
