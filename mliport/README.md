# mliport

VASP-style CLI / TUI / Python API for reliable machine-learning interatomic-potential (MLIP) calculations and molecular dynamics with **UMA (FAIRChem)**, **MACE**, **DPA (DeepMD-kit)**, and **GRACE**.

> **Full documentation:** see the repository root [`README.md`](../README.md) (English) and [`README_CN.md`](../README_CN.md) (中文).

## Quick Start

From the repository root:

```bash
# One-command installer (auto-detects GPU, installs all four engines)
./scripts/install_mliport.sh

# UMA single point
.venv/bin/mliport sp structure.cif --model uma-s-1.pt --task omat --device cpu

# MACE
.venv-mace/bin/mliport sp bulk.cif --model mace.model \
  --model-type mace --task bulk --head default --device cuda:0

# DPA
.venv-dpa/bin/mliport opt bulk.cif --model dpa.pt \
  --model-type dpa --task bulk --head Domains_SSE_PBE --device cuda:0

# GRACE
.venv-grace/bin/mliport sp bulk.cif --model grace_model/ \
  --model-type grace --task bulk --device cuda:0
```

## Package Layout

```
mliport/
├── mliport/                # Python package
│   ├── cli.py            # CLI entry point
│   ├── engine.py         # CalculationEngine (unified execution)
│   ├── calculators/      # MACE/DPA/GRACE wrappers + CalculatorFactory
│   ├── runners/          # SinglePoint, Optimization, MD, Batch
│   ├── analysis/         # Trajectory analysis (calculator-independent)
│   ├── install/          # Installation / compatibility matrix
│   ├── tui/              # Textual TUI
│   └── writers/          # OUTCAR, CONTCAR, XDATCAR, OSZICAR, JSON
├── templates/            # INCAR templates
└── examples/             # Example scripts
```

## License

MIT License. See [`LICENSE.md`](../LICENSE.md) and [`LICENSE`](LICENSE).
