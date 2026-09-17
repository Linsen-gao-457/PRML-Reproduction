# PRML-Reproduction

Implementations and reproducible figures from *Pattern Recognition and Machine
Learning*.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Reproduce a figure

Run figure scripts from the repository root. Configuration lives in the matching
file under `configs/` and generated images are written to `outputs/figures/`.

```bash
python -m chapters.ch01 1
python -m chapters.ch01 4
python -m chapters.ch01 all
```

## Layout

- `src/prml/`: reusable distributions, models, and utilities
- `chapters/`: scripts and chapter notes
- `configs/`: reproducibility parameters for each experiment
- `tests/`: unit tests for reusable implementations
- `outputs/`: generated figures and numerical results
