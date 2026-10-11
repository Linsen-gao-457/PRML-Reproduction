# PRML-Reproduction

Implementations and reproducible figures from _Pattern Recognition and Machine
Learning_.

## Setup

For normal development, install versions allowed by pyproject.toml:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

For an environment matching the tested dependency versions, use the lock file:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]' -c requirements.lock.txt
```

## Reproduce a figure

Run figure scripts from the repository root. Configuration lives in the matching
file under `configs/`, and generated images are grouped by chapter under
`outputs/figures/chXX/`.

```bash
python -m chapters.ch01 all
python -m chapters.ch02 all
python -m chapters.ch03 all

```

## Layout

- `src/prml/`: reusable distributions, models, and utilities
- `chapters/`: scripts and chapter notes
- `configs/`: reproducibility parameters for each experiment
- `outputs/`: generated figures and numerical results
