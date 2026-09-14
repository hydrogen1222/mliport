"""Regenerate machine-generated documentation from repository code.

Generated artifacts (never edit them by hand):

- ``docs/installation.md`` -- environment/Python matrix from the
  compatibility registry;
- ``docs/backends/<engine>.md`` -- supported profiles from the pinned model
  manifest, runtime facts from the compatibility registry and V100
  capability evidence from the packaged capability records;
- ``mliport/README.md`` -- PyPI/package README generated from the root
  README with repository-relative links rewritten to GitHub URLs.

Usage::

    python scripts/generate_docs.py [--check]

``--check`` exits non-zero when a generated file is out of date instead of
rewriting it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mliport"))

from mliport.install.compatibility import (  # noqa: E402
    BACKENDS,
    python_versions_for,
)
from mliport.install.docs import (  # noqa: E402
    render_installation_markdown,
)

MANIFEST_PATH = REPO / "validation" / "science" / "model_manifest.json"
CAPABILITY_DIR = REPO / "mliport" / "mliport" / "data" / "validation"
GITHUB_URL = "https://github.com/hydrogen1222/mliport"

_BACKEND_INTRO = {
    "uma": (
        "UMA checkpoints from Meta FAIR are loaded through the upstream "
        "`fairchem-core` stack. `uma` and the upstream package alias "
        "`fairchem` are the same runtime; `omol`/`oc20`/`omat` task families "
        "are selected with `TASK`/`--task`."
    ),
    "mace": (
        "MACE checkpoints are loaded through `mace-torch`. MACE profiles are "
        "selected by checkpoint plus dtype (`float64` is the primary "
        "accuracy-first profile; `float32` is a separate identity and is "
        "never mixed with float64 results)."
    ),
    "dpa": (
        "DPA checkpoints are loaded through `deepmd-kit`. Multi-task "
        "checkpoints require an explicit branch `HEAD`; mliport fails closed "
        "instead of guessing a head."
    ),
    "grace": (
        "GRACE models are exported SavedModels loaded through "
        "`tensorpotential`. `MODEL_PATH` points at the SavedModel directory, "
        "not a single file."
    ),
}

_MODEL_ACQUISITION = {
    "uma": (
        "Download UMA checkpoints from the upstream fairchem/UMA release "
        "channels and verify the SHA-256 before use. mliport never "
        "redistributes model weights; see [citations](../citations.md)."
    ),
    "mace": (
        "Download MACE checkpoints from the upstream MACE releases. The "
        "pinned `mace_omat` profiles use "
        "`mace-omat-0-medium.model` and are referenced by SHA-256 in "
        "`validation/science/model_manifest.json`."
    ),
    "dpa": (
        "Download DPA checkpoints from the upstream DeePMD-kit/DPA release "
        "channels. Read the available branches with `dp --pt show <model> "
        "model-branch` and pass the intended branch as `HEAD`."
    ),
    "grace": (
        "Download official GRACE exported SavedModels from the upstream "
        "AMS-ICAMS-RUB release channels; the model path is the SavedModel "
        "directory."
    ),
}

_LIMITATIONS = {
    "uma": [
        "UMA task families are not interchangeable: a molecule run must use "
        "an `omol` task and a periodic bulk run an `omat` task.",
        "Upstream precision is model-defined; mliport reports the framework "
        "version rather than pretending to control model weights.",
        "`fairchem` is an alias, not a second backend.",
    ],
    "mace": [
        "The float32 profile is a separate identity; never compare its "
        "energies with float64 results without saying so.",
        "Multi-head checkpoints require an explicit `HEAD`; mliport refuses "
        "to guess.",
        "MACE stress support depends on the checkpoint; check the T1 "
        "`stress_supported` record.",
    ],
    "dpa": [
        "Multi-task checkpoints require an explicit `HEAD`; the product "
        "fails closed rather than using an arbitrary branch.",
        "DPA has no `inference_mode` concept; UMA-only options are rejected "
        "in strict config mode.",
        "Stress support depends on the checkpoint and its branch.",
        "DPA needs process-level CUDA isolation. The CLI restarts a command "
        "with `CUDA_VISIBLE_DEVICES` when it is unset; the Python API needs "
        "the variable set in the caller.",
        "`deepmd-kit[torch]` metadata is required for the model loader; the "
        "installer carries the matching `mpich` package.",
    ],
    "grace": [
        "`MODEL_PATH` must be the SavedModel directory.",
        "TensorFlow GPU visibility must be isolated per process. The CLI "
        "restarts a command with `CUDA_VISIBLE_DEVICES` when it is unset; "
        "the Python API and the queue worker need the same isolation.",
        "Stress support depends on the checkpoint.",
    ],
}

_MINIMAL_COMMAND = {
    "uma": (
        ".venv/bin/mliport sp structure.cif --model /path/to/uma.pt "
        "--model-type uma --task omat"
    ),
    "mace": (
        ".venv-mace/bin/mliport sp structure.cif "
        "--model /path/to/mace-omat-0-medium.model --model-type mace"
    ),
    "dpa": (
        ".venv-dpa/bin/mliport sp structure.cif --model /path/to/DPA-3.1-3M.pt "
        "--model-type dpa --head Omat24"
    ),
    "grace": (
        ".venv-grace/bin/mliport sp structure.cif --model /path/to/saved_model "
        "--model-type grace"
    ),
}


def _profiles_for(engine: str) -> dict[str, dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        key: profile
        for key, profile in manifest["profiles"].items()
        if profile.get("engine") == engine
    }


def _capability_evidence(engine: str) -> list[dict]:
    evidence = []
    for path in sorted(CAPABILITY_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_identity", {}).get("model_type") == engine:
            evidence.append(payload)
    return evidence


def _sha_short(sha: object) -> str:
    text = str(sha or "")
    return text[:12] + "..." if len(text) > 12 else text


def render_backend_page(engine: str) -> str:
    """One backend page: facts from registry + manifest + capability records."""
    backend = BACKENDS[engine]
    versions = python_versions_for(engine)
    profiles = _profiles_for(engine)
    evidence = _capability_evidence(engine)
    lines = [
        f"# {backend.label}",
        "",
        _BACKEND_INTRO[engine],
        "",
        "## Install environment",
        "",
        "| Environment | Python | Runtime stack |",
        "|---|---|---|",
        f"| `{backend.venv_name}` | {', '.join(versions)} | "
        f"`{backend.requirement}` ({backend.framework} "
        f"{'/'.join(sorted({p.framework_version for p in backend.arch_profiles.values()}))}) |",
        "",
        "```bash",
        f"./scripts/install_mliport.sh --engines {engine} --device auto",
        "```",
        "",
        "## Supported profiles",
        "",
        "| Profile | Identity | dtype | task | head | model SHA-256 |",
        "|---|---|---|---|---|---|",
    ]
    for key, profile in sorted(profiles.items()):
        lines.append(
            f"| `{key}` | {profile.get('identity')} | {profile.get('dtype')} "
            f"| {profile.get('task')} | {profile.get('head')} "
            f"| `{_sha_short(profile.get('model_sha256'))}` |"
        )
    lines += [
        "",
        "## Minimal command",
        "",
        "```bash",
        _MINIMAL_COMMAND[engine],
        "```",
        "",
        "## Precision",
        "",
        "The dtype is part of the profile identity. "
        + (
            "MACE profiles declare `float64` (primary) and `float32` "
            "(separate identity) explicitly."
            if engine == "mace"
            else "This backend's precision is model-defined; mliport records "
            "the effective dtype from the backend, never a value it assumes."
        ),
        "",
        "## Stress",
        "",
        "mliport reports stress only if the loaded checkpoint exposes it "
        "(`implemented_properties`). Never assume stress support from the "
        "engine name: the T1 evidence records `stress_supported` per profile, "
        "and `mliport doctor` reports what the current model provides.",
        "",
        "## Head / task semantics",
        "",
    ]
    if engine == "uma":
        lines.append(
            "`TASK` selects the UMA task family (`omat` for periodic bulk, "
            "`omol` for molecules, `oc20` for catalysis). There is no "
            "separate `HEAD` field for UMA."
        )
    elif engine == "dpa":
        lines.append(
            "Multi-task checkpoints expose named branches. Pass the intended "
            "branch with `--head`/`HEAD`; omitting it on a multi-task "
            "checkpoint is a fatal configuration error, never a silent "
            "default branch."
        )
    elif engine == "mace":
        lines.append(
            "`HEAD` selects a head when the checkpoint provides several; "
            "single-head checkpoints do not need it. `TASK` is the explicit "
            "PBC semantic (`bulk`/`molecule`)."
        )
    else:
        lines.append(
            "GRACE models do not expose mliport-level heads; `TASK` is the "
            "explicit PBC semantic (`bulk`/`molecule`)."
        )
    lines += [
        "",
        "## Model acquisition",
        "",
        _MODEL_ACQUISITION[engine],
        "",
        "## V100 capability evidence",
        "",
    ]
    if evidence:
        lines += [
            "| evidence id | dtype | head | backend | workloads |",
            "|---|---|---|---|---|",
        ]
        for item in evidence:
            identity = item["model_identity"]
            workloads = ", ".join(
                f"{name}:{entry.get('status')}"
                for name, entry in sorted(item["workloads"].items())
            )
            lines.append(
                f"| `{item['evidence_id']}` | {identity.get('dtype_effective')} "
                f"| {identity.get('head_effective')} "
                f"| {identity.get('backend_version')} "
                f"({identity.get('framework_versions')}) | {workloads} |"
            )
    else:
        lines.append(
            "No packaged capability evidence for this backend yet; see the "
            "validation report for the current campaign records."
        )
    lines += [
        "",
        "## License and citation",
        "",
        "mliport is MIT-licensed; the upstream model/software must be cited "
        "separately. See [citations](../citations.md) for the authoritative "
        "upstream references.",
        "",
        "## Known limitations",
        "",
    ]
    lines += [f"- {item}" for item in _LIMITATIONS[engine]]
    lines.append("")
    return "\n".join(lines)


CAPABILITY_MATRIX_PATH = REPO / "validation" / "science" / "capability_matrix.json"


def render_neb_capability_table() -> str:
    """Interface expressibility table generated from the capability matrix."""
    matrix = json.loads(CAPABILITY_MATRIX_PATH.read_text(encoding="utf-8"))
    lines = [
        "| Feature | API | direct CLI | INCAR | TUI |",
        "|---|---|---|---|---|",
    ]
    for feature in matrix["features"]:
        if not feature["feature"].startswith("neb_"):
            continue
        name = feature["feature"].removeprefix("neb_").replace("_", " ")
        lines.append(
            f"| {name} | {feature['api']} | {feature['direct_cli']} "
            f"| {feature['incar']} | {feature['tui']} |"
        )
    lines.append("")
    lines.append(
        "This matrix is generated from "
        "`validation/science/capability_matrix.json`. A `yes` means the entry "
        "point can express the feature; it is not an equivalence or "
        "validation claim. A `no`/`partial` entry carries a note in that file."
    )
    return "\n".join(lines)


def render_neb_page() -> str:
    """NEB documentation: curated guidance + generated capability matrix."""
    return f"""# NEB / CI-NEB

**Purpose.** Find a minimum-energy path between two endpoints using
fixed-cell nudged elastic band (with optional climbing image), and report the
forward barrier.

**Minimal command.**

```bash
.venv-mace/bin/mliport neb --initial initial.vasp --final final.vasp \
  --model mace-omat-0-medium.model --model-type mace --images 7 \
  --allow-unvalidated-neb --output runs/neb
```

The current checkpoints are outside the validated NEB envelope: mliport
cannot yet certify their energy-gradient consistency for this exact
model/runtime, so it refuses to run without the explicit
`--allow-unvalidated-neb` opt-in. Such runs are recorded as
`workflow_smoke_only` and do not claim a physical barrier. Once a runtime has
a validated capability record the flag is no longer required.

**Important options.**

| Option | Meaning |
|---|---|
| `--initial`, `--final` | endpoint structures (mandatory) |
| `--images` | number of intermediate images (endpoints additional) |
| `--climb` / `NEB_CLIMB` | climbing-image refinement |
| `--atom-map` | 0-based mapping of final indices in initial-atom order |
| `--image-shifts` | integer lattice shifts per atom (`0,0,0;1,0,0`) |
| `--neb-interpolation`, `--neb-path-convention` | IDPP interpolation and path convention |
| `--neb-pre-fmax`, `--neb-pre-max-steps` | endpoint pre-relaxation before interpolation |
| `--endpoint-policy`, `--endpoint-fmax`, `--endpoint-steps` | how endpoint images are constrained |
| `--neb-spring`, `--neb-maxstep`, `--neb-min-distance` | band/spring parameters |
| `--checkpoint-interval` | band checkpoint frequency |
| `--resume` | resume from a checkpoint/directory |
| `--allow-unvalidated-neb` | opt-in for NEB configurations outside the validated envelope (recorded, not silent) |

**Inputs.** Two structures with consistent atom order (or an explicit
`--atom-map`), a model, an explicit backend, and enough memory for
`images + 2` calculators.

**Outputs.** `NEB_BAND`/band trajectory, per-image energies/forces,
`resolved_config.json`, checkpoint files, and a result JSON with `barrier`,
convergence state and the sampled barrier value.

**Failure semantics.** Non-convergence is reported with the best band and a
failure reason; it is not a converged barrier. Bad atom mapping or
inconsistent endpoints fail before the band is built. Checkpoint/resume
locks the original run directory so the band cannot be silently restarted
with different settings.

**Scientific caveats.**

- A NEB saddle candidate is **not** a verified transition state. Hessian
  (frequency) verification is a separate analysis and is not implied by a
  converged band.
- The barrier is only comparable within one model/head identity and one NEB
  protocol (images, interpolation, endpoint policy, spring).
- Endpoint pre-relaxation changes the reference; record it.
- The saddle is meaningful only if the endpoints are the intended states and
  the atom mapping is chemically correct.

**Interface capability matrix**

{render_neb_capability_table()}

**Example (resume).**

```bash
.venv-mace/bin/mliport neb --resume runs/neb --output runs/neb
```

Resume continues in the original run directory and checks the recorded
identity (model, task/head/dtype, cell/PBC, constraints, band setup and
options). Step budgets are part of that identity in this beta: a resume that
changes `--max-steps` is rejected with `Resume fingerprint is incompatible`;
the explicit error leaves the checkpoint untouched.
"""


def render_package_readme(root_readme: str) -> str:
    """PyPI/package README generated from the root README.

    Repository-relative links are rewritten to absolute GitHub URLs so the
    rendered PyPI page keeps working outside a checkout.
    """
    body = re.sub(
        r"\]\((docs/[^)]+|CITATION\.cff|LICENSE\.md|validation/[^)]+)\)",
        lambda match: f"]({GITHUB_URL}/blob/main/{match.group(1)})",
        root_readme,
    )
    header = (
        "<!-- Generated from the repository root README.md by "
        "scripts/generate_docs.py; DO NOT EDIT. -->\n\n"
    )
    return header + body


def _generated_files() -> dict[Path, callable]:
    files: dict[Path, callable] = {
        REPO / "docs" / "installation.md": render_installation_markdown,
        REPO / "mliport" / "README.md": lambda: render_package_readme(
            (REPO / "README.md").read_text(encoding="utf-8")
        ),
    }
    for engine in BACKENDS:
        files[REPO / "docs" / "backends" / f"{engine}.md"] = (
            lambda engine=engine: render_backend_page(engine)
        )
    files[REPO / "docs" / "workflows" / "neb.md"] = render_neb_page
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of rewriting when a generated file is stale.",
    )
    args = parser.parse_args()
    failures = 0
    for path, render in _generated_files().items():
        content = render()
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == content:
            print(f"[docs] up to date: {path.relative_to(REPO)}")
            continue
        if args.check:
            print(
                f"[docs] STALE: {path.relative_to(REPO)} "
                "(run python scripts/generate_docs.py)",
                file=sys.stderr,
            )
            failures += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[docs] wrote {path.relative_to(REPO)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
